------------------------------ MODULE KvDiskTier -----------------------------
\* State and staging-fence model for the Layer 2 `--kv-disk` tier
\* (`src/engine.rs:67-90`, `3488-3713`):
\*   - `tier_enter` spills older positions to `file` up to `cold = min(n, lo + Kps + 1)`
\*     (one split ahead of `lo = n - HotCap`) so cache-fixed split boundary
\*     `b = Ceil(lo, Kps) * Kps` always lies in the overlap `lo..cold` held by BOTH
\*     `file` and `hotRing`!
\*   - Each decode `step(pos)` spills positions about to be overwritten in `hotRing`
\*     (`pos % HotCap`) before writing `pos`, plans cold splits `[0..b)` into two
\*     ping-pong staging halves `half \in {0, 1}`, and waits on `pending[half]`
\*     before CPU `tier_read` refills that staging half.

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
    MaxPos,         \* Maximum context length
    HotCap,         \* Hot ring capacity in positions (>= Kps + 2)
    Kps,            \* Keys per cache-fixed attention split (>= 2)
    HalfCap,        \* Positions per staging half (>= Kps)
    MinN,           \* Minimum context to enter `--kv-disk` tier
    Bug             \* "none" | "no_overlap" | "no_fence"

ASSUME
    /\ Kps >= 2
    /\ HalfCap >= Kps
    /\ HotCap >= Kps + 2
    /\ MinN >= HotCap
    /\ MaxPos > MinN
    /\ Bug \in {"none", "no_overlap", "no_fence"}

CeilDiv(n, d) == (n + d - 1) \div d
Max(a, b) == IF a >= b THEN a ELSE b
Min(a, b) == IF a <= b THEN a ELSE b

KeyVal(p) == p + 100

VARIABLES
    phase,          \* "untiered" | "idle" | "stepping" | "done"
    pos,            \* Cached positions `0..(pos - 1)`
    tierOn,         \* `Tier.on`
    lo,             \* `Tier.lo`: oldest position guaranteed in `hotRing`
    cold,           \* `Tier.cold`: exclusive upper bound written to `file`
    filed,          \* `Tier.filed`: high-water mark written to `file`
    hotRing,        \* [0..(HotCap - 1) -> Int]
    diskFile,       \* [0..(MaxPos - 1) -> Int]
    stgHalf,        \* [0..1 -> [0..(HalfCap - 1) -> Int]]
    curHalf,        \* Active staging half `0` or `1`
    gpuPending,     \* [0..1 -> BOOLEAN]: TRUE while an in-flight CmdBuf still reads half `h`
    splitB,         \* Planned split boundary `b` between `diskFile` (`0..b-1`) and `hotRing` (`b..pos`)
    alignedPlan,    \* BOOLEAN: whether `b` aligned cleanly to a `Kps` multiple `<= cold`
    planChunks,     \* Seq of `[c0, c1, round]` file chunks for the active step
    chunkIdx,       \* Next chunk index `1..(Len(planChunks) + 1)`
    loadedRound,    \* Currently staged round (-1 = none)
    stgRace,        \* BOOLEAN: TRUE if CPU overwrote a staging half while `gpuPending[h]` was TRUE
    badRead         \* BOOLEAN: TRUE if attention read a wrong/missing key from `stgHalf` or `hotRing`

vars == <<
    phase, pos, tierOn, lo, cold, filed, hotRing, diskFile,
    stgHalf, curHalf, gpuPending, splitB, alignedPlan, planChunks,
    chunkIdx, loadedRound, stgRace, badRead
>>

Init ==
    /\ phase       = "untiered"
    /\ pos         = MinN
    /\ tierOn      = FALSE
    /\ lo          = 0
    /\ cold        = 0
    /\ filed       = 0
    /\ hotRing     = [i \in 0..(HotCap - 1) |-> 0]
    /\ diskFile    = [p \in 0..(MaxPos - 1) |-> 0]
    /\ stgHalf     = [h \in 0..1 |-> [i \in 0..(HalfCap - 1) |-> 0]]
    /\ curHalf     = 0
    /\ gpuPending  = [h \in 0..1 |-> FALSE]
    /\ splitB      = 0
    /\ alignedPlan = TRUE
    /\ planChunks  = << >>
    /\ chunkIdx    = 1
    /\ loadedRound = -1
    /\ stgRace     = FALSE
    /\ badRead     = FALSE

\* `Engine::tier_enter(pos)` (`src/engine.rs:3488-3516`):
\* writes `[0..cold0)` to `diskFile` and copies `[lo0..pos)` into `hotRing[p % HotCap]`.
TierEnter ==
    /\ phase = "untiered"
    /\ ~tierOn
    /\ pos >= MinN
    /\ LET lo0   == pos - HotCap
           cold0 == IF Bug = "no_overlap"
                    THEN lo0
                    ELSE Min(pos, lo0 + Kps + 1)
       IN /\ tierOn'      = TRUE
          /\ lo'          = lo0
          /\ cold'        = cold0
          /\ filed'       = cold0
          /\ diskFile'    = [p \in 0..(MaxPos - 1) |-> IF p < cold0 THEN KeyVal(p) ELSE 0]
          /\ hotRing'     = [i \in 0..(HotCap - 1) |->
                                IF \E p \in lo0..(pos - 1) : p % HotCap = i
                                THEN LET p == CHOOSE q \in lo0..(pos - 1) : q % HotCap = i
                                     IN KeyVal(p)
                                ELSE 0]
          /\ curHalf'     = 0
          /\ loadedRound' = -1
          /\ phase'       = "idle"
    /\ UNCHANGED <<pos, stgHalf, gpuPending, splitB, alignedPlan, planChunks, chunkIdx, stgRace, badRead>>

RECURSIVE BuildChunks(_, _, _, _)
BuildChunks(c, bLim, round, room) ==
    IF c >= bLim THEN << >>
    ELSE LET nextB == ((c \div Kps) + 1) * Kps
             c1    == Min(bLim, nextB)
             len   == c1 - c
             rUse  == IF room < len THEN round + 1 ELSE round
             rem   == (IF room < len THEN HalfCap ELSE room) - len
             off   == (IF room < len THEN HalfCap ELSE room) - rem - len
         IN <<[c0 |-> c, c1 |-> c1, round |-> rUse, off |-> off]>>
            \o BuildChunks(c1, bLim, rUse, rem)

\* Start of `Engine::step(pos)` (`src/engine.rs:3591-3669`):
\* spills ring victims up to `want` into `diskFile`, plans chunks `[0..b)`, and enters `"stepping"`.
BeginDecodeStep ==
    /\ phase = "idle"
    /\ tierOn
    /\ pos < MaxPos
    /\ LET need   == IF pos + 1 > HotCap THEN pos + 1 - HotCap ELSE 0
           lo1    == Max(lo, need)
           done   == IF pos >= 1 THEN pos - 1 ELSE 0
           want   == IF Bug = "no_overlap"
                     THEN lo1
                     ELSE Max(need, Min(done, lo1 + Kps))
           cold1  == Max(cold, want)
           filed1 == Max(filed, cold1)
           nDisk  == [p \in 0..(MaxPos - 1) |->
                        IF p >= cold /\ p < cold1
                        THEN hotRing[p % HotCap]
                        ELSE diskFile[p]]
           bCand  == IF lo1 = 0 THEN 0 ELSE CeilDiv(lo1, Kps) * Kps
           isAln  == bCand <= pos /\ bCand <= cold1 /\ Kps <= HalfCap
           bUse   == IF isAln THEN bCand ELSE lo1
           chunks == BuildChunks(0, bUse, 0, HalfCap)
       IN /\ lo'          = lo1
          /\ cold'        = cold1
          /\ filed'       = filed1
          /\ diskFile'    = nDisk
          /\ splitB'      = bUse
          /\ alignedPlan' = isAln
          /\ planChunks'  = chunks
          /\ chunkIdx'    = 1
          /\ loadedRound' = -1
          /\ phase'       = "stepping"
    /\ UNCHANGED <<pos, tierOn, hotRing, stgHalf, curHalf, gpuPending, stgRace, badRead>>

\* `Engine::tier_load` (`src/engine.rs:3683-3713`): switches staging half when `round > 0`,
\* waits for `gpuPending[h]` to complete (`Bug = "no_fence"` skips the wait!), and loads round chunks.
LoadAndAttendChunk ==
    /\ phase = "stepping"
    /\ chunkIdx <= Len(planChunks)
    /\ LET ch       == planChunks[chunkIdx]
           needLoad == loadedRound # ch.round
           switchH  == needLoad /\ ch.round > 0
           nextH    == IF switchH THEN 1 - curHalf ELSE curHalf
           pendAfterSwitch == IF switchH
                              THEN [gpuPending EXCEPT ![curHalf] = TRUE]
                              ELSE gpuPending
           raced    == needLoad /\ Bug = "no_fence" /\ pendAfterSwitch[nextH]
           pendAfterFence == IF needLoad /\ Bug # "no_fence"
                             THEN [pendAfterSwitch EXCEPT ![nextH] = FALSE]
                             ELSE pendAfterSwitch
           nextStg  == IF needLoad
                       THEN [stgHalf EXCEPT ![nextH] =
                                [i \in 0..(HalfCap - 1) |->
                                    IF \E k \in 1..Len(planChunks) :
                                        LET ck == planChunks[k]
                                        IN ck.round = ch.round /\ i >= ck.off /\ i < ck.off + (ck.c1 - ck.c0)
                                    THEN LET k == CHOOSE x \in 1..Len(planChunks) :
                                                    LET ck == planChunks[x]
                                                    IN ck.round = ch.round /\ i >= ck.off /\ i < ck.off + (ck.c1 - ck.c0)
                                             ck == planChunks[k]
                                         IN diskFile[ck.c0 + (i - ck.off)]
                                    ELSE stgHalf[nextH][i]]]
                       ELSE stgHalf
           readOk   == \A p \in ch.c0..(ch.c1 - 1) :
                        nextStg[nextH][ch.off + (p - ch.c0)] = KeyVal(p)
       IN /\ curHalf'     = nextH
          /\ gpuPending'  = pendAfterFence
          /\ stgHalf'     = nextStg
          /\ loadedRound' = ch.round
          /\ chunkIdx'    = chunkIdx + 1
          /\ stgRace'     = (stgRace \/ raced)
          /\ badRead'     = (badRead \/ ~readOk)
    /\ UNCHANGED <<phase, pos, tierOn, lo, cold, filed, hotRing, diskFile, splitB, alignedPlan, planChunks>>

\* Hot-ring splits `[splitB..pos]` + new key write at `pos % HotCap` + commit CmdBuf on `curHalf`.
FinishDecodeStep ==
    /\ phase = "stepping"
    /\ chunkIdx > Len(planChunks)
    /\ LET hotOk == /\ splitB >= lo
                    /\ \A p \in splitB..(pos - 1) : hotRing[p % HotCap] = KeyVal(p)
       IN /\ hotRing'    = [hotRing EXCEPT ![pos % HotCap] = KeyVal(pos)]
          /\ pos'        = pos + 1
          /\ gpuPending' = [gpuPending EXCEPT ![curHalf] = TRUE]
          /\ curHalf'    = 1 - curHalf
          /\ badRead'    = (badRead \/ ~hotOk)
          /\ phase'      = IF pos + 1 = MaxPos THEN "done" ELSE "idle"
    /\ UNCHANGED <<tierOn, lo, cold, filed, diskFile, stgHalf, splitB, alignedPlan,
                   planChunks, chunkIdx, loadedRound, stgRace>>

\* Background GPU completion of an in-flight command buffer on staging half `h`.
GpuCompleteHalf(h) ==
    /\ h \in 0..1
    /\ gpuPending[h]
    /\ gpuPending' = [gpuPending EXCEPT ![h] = FALSE]
    /\ UNCHANGED <<phase, pos, tierOn, lo, cold, filed, hotRing, diskFile,
                   stgHalf, curHalf, splitB, alignedPlan, planChunks, chunkIdx,
                   loadedRound, stgRace, badRead>>

\* `Engine::tier_leave` (`src/engine.rs:3518-3552`): waits for GPU idle and verifies
\* `[0..lo)` from `diskFile` + `[lo..pos)` from `hotRing` reconstructs `[0..pos)` intact.
TierLeave ==
    /\ phase \in {"idle", "done"}
    /\ tierOn
    /\ LET fullOk == /\ \A p \in 0..(lo - 1) : diskFile[p] = KeyVal(p)
                     /\ \A p \in lo..(pos - 1) : hotRing[p % HotCap] = KeyVal(p)
       IN /\ tierOn'     = FALSE
          /\ gpuPending' = [h \in 0..1 |-> FALSE]
          /\ badRead'    = (badRead \/ ~fullOk)
          /\ phase'      = "done"
    /\ UNCHANGED <<pos, lo, cold, filed, hotRing, diskFile, stgHalf, curHalf,
                   splitB, alignedPlan, planChunks, chunkIdx, loadedRound, stgRace>>

DoneStutter ==
    /\ phase = "done" /\ ~tierOn
    /\ UNCHANGED vars

Next ==
    \/ TierEnter
    \/ BeginDecodeStep
    \/ LoadAndAttendChunk
    \/ FinishDecodeStep
    \/ (\E h \in 0..1 : GpuCompleteHalf(h))
    \/ TierLeave
    \/ DoneStutter

Spec == Init /\ [][Next]_vars

\* ---- Invariants -----------------------------------------------------------

TypeOK ==
    /\ phase \in {"untiered", "idle", "stepping", "done"}
    /\ pos \in MinN..MaxPos
    /\ tierOn \in BOOLEAN
    /\ lo \in 0..pos
    /\ cold \in 0..pos
    /\ curHalf \in 0..1
    /\ gpuPending \in [0..1 -> BOOLEAN]

\* Every step's cold/hot partition lands on an exact `Kps` split boundary (`alignedPlan = TRUE`)
\* because `cold` stays one split ahead of `lo`.
SplitBoundaryAligned ==
    /\ alignedPlan = TRUE
    /\ (phase = "stepping" => (splitB % Kps = 0 /\ splitB >= lo /\ splitB <= cold))

\* CPU `tier_read` never overwrites a staging half while an in-flight GPU command buffer reads it.
StagingHalfSafe == ~stgRace

\* Every cold file read, every hot ring read, and `tier_leave` reconstruction match exact `KeyVal(p)`.
EveryKeyExact ==
    /\ ~badRead
    /\ (tierOn /\ phase \in {"idle", "done"} =>
          /\ \A p \in 0..(cold - 1) : diskFile[p] = KeyVal(p)
          /\ \A p \in lo..(pos - 1) : hotRing[p % HotCap] = KeyVal(p))

=============================================================================
