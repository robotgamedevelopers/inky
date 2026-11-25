------------------------------ MODULE VecAlloc ------------------------------
\* Exact state and liveness model for `VecAlloc` (`src/engine.rs:460, 495-505, 2281-2341, 2677-2719`)
\* across the 24-layer looped architecture (`BLOCK = 6`, `LOOP_START = 6`, `LOOP_END = 18`, `N_PASS = 3`),
\* including the `LoopCD` dry-state weak coda (`pass = 0`, `ip \in 18..23`) that runs while
\* `prelude`, `prelude_blocks`, `bank.0`, and `bank.1` must stay intact for `pass = 1..2`.

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
    NVec,       \* Vector pool size (`B_NVEC = 10` for batch arena, `N_VEC = 16` for solo)
    RunLoopCd,  \* BOOLEAN: whether `pass == 0` runs the weak reference coda
    Bug         \* "none" | "unpinned_weak"

ASSUME
    /\ NVec >= 2
    /\ RunLoopCd \in BOOLEAN
    /\ Bug \in {"none", "unpinned_weak"}

NPhys     == 24
ResBlock  == 6
LoopStart == 6
LoopEnd   == 18
NPass     == 3

Stages == {"prelude", "loop", "weak_coda", "coda", "done"}

\* Round-robin `VecAlloc::alloc` (`src/engine.rs:497-504`): searches `0..(NVec - 1)` steps
\* starting at `next`, skipping vector `0` (reserved initial `k_embed` slot) and `live`.
RECURSIVE FindSlot(_, _, _)
FindSlot(curNext, stepsLeft, liveSet) ==
    IF stepsLeft = 0 THEN -1
    ELSE LET c == curNext
             nxt == (curNext + 1) % NVec
         IN IF c # 0 /\ c \notin liveSet
            THEN c
            ELSE FindSlot(nxt, stepsLeft - 1, liveSet)

NextAfterAlloc(curNext, chosen) == (chosen + 1) % NVec

SeqSet(seq) == {seq[i] : i \in 1..Len(seq)}

VARIABLES
    stage,          \* Current schedule phase in Stages
    pass,           \* Current loop pass (0..2)
    ip,             \* Current physical layer index (0..23)
    substep,        \* "mid" (attn wo) or "out" (ffn down)
    vaNext,         \* `VecAlloc.next` cursor in 0..(NVec - 1)
    cur,            \* Vector index holding current residual stream
    mid,            \* Vector index holding intermediate post-attn residual
    blocks,         \* Seq of depth-residual block-start vector indices
    prelude,        \* Saved `cur` after layer 5
    preludeBlocks,  \* Saved `blocks` after layer 5
    bank0,          \* Loop message bank 0 (`blocks[preludeBlocks + 1]` at end of pass)
    bank1,          \* Loop message bank 1 (`cur` at end of pass)
    curW,           \* Weak-coda `cur_w` cursor
    blocksW,        \* Weak-coda `blocks_w` sequence
    writeStamp,     \* Monotone stamp written to a vector slot when produced
    slotStamp,      \* [0..(NVec - 1) -> Nat] actual stamp currently residing in each slot
    curStamp,       \* Expected stamp for `cur`
    midStamp,       \* Expected stamp for `mid`
    blkStamps,      \* Expected stamps aligned 1-to-1 with `blocks`
    preludeStamp,   \* Expected stamp for `prelude`
    preBlkStamps,   \* Expected stamps aligned 1-to-1 with `preludeBlocks`
    bank0Stamp,     \* Expected stamp for `bank0`
    bank1Stamp,     \* Expected stamp for `bank1`
    curWStamp,      \* Expected stamp for `curW`
    blkWStamps,     \* Expected stamps aligned 1-to-1 with `blocksW`
    exhausted,      \* BOOLEAN: TRUE if `VecAlloc::alloc` ever ran out of free vectors
    corrupted       \* BOOLEAN: TRUE if any read observed an overwritten live vector

vars == <<
    stage, pass, ip, substep, vaNext, cur, mid, blocks,
    prelude, preludeBlocks, bank0, bank1, curW, blocksW,
    writeStamp, slotStamp, curStamp, midStamp, blkStamps,
    preludeStamp, preBlkStamps, bank0Stamp, bank1Stamp, curWStamp, blkWStamps,
    exhausted, corrupted
>>

Init ==
    /\ stage         = "prelude"
    /\ pass          = 0
    /\ ip            = 0
    /\ substep       = "mid"
    /\ vaNext        = 1
    /\ cur           = 0
    /\ mid           = 0
    /\ blocks        = << >>
    /\ prelude       = 0
    /\ preludeBlocks = << >>
    /\ bank0         = 0
    /\ bank1         = 0
    /\ curW          = 0
    /\ blocksW       = << >>
    /\ writeStamp    = 1
    /\ slotStamp     = [i \in 0..(NVec - 1) |-> IF i = 0 THEN 1 ELSE 0]
    /\ curStamp      = 1
    /\ midStamp      = 0
    /\ blkStamps     = << >>
    /\ preludeStamp  = 0
    /\ preBlkStamps  = << >>
    /\ bank0Stamp    = 0
    /\ bank1Stamp    = 0
    /\ curWStamp     = 0
    /\ blkWStamps    = << >>
    /\ exhausted     = FALSE
    /\ corrupted     = FALSE

SeqIntact(slots, stamps) ==
    /\ Len(slots) = Len(stamps)
    /\ \A i \in 1..Len(slots) : slotStamp[slots[i]] = stamps[i]

PinnedForStage ==
    IF Bug = "unpinned_weak" THEN {}
    ELSE IF stage = "loop" THEN
        IF pass = 0 THEN {prelude} ELSE {prelude, bank0, bank1}
    ELSE IF stage = "weak_coda" THEN
        {prelude, bank0, bank1}
    ELSE {}

ReadInputsIntact ==
    IF stage = "weak_coda" THEN
        /\ slotStamp[curW] = curWStamp
        /\ SeqIntact(blocksW, blkWStamps)
    ELSE
        /\ slotStamp[cur] = curStamp
        /\ SeqIntact(blocks, blkStamps)
        /\ (stage = "loop" /\ pass > 0 =>
              /\ slotStamp[bank0] = bank0Stamp
              /\ slotStamp[bank1] = bank1Stamp)

\* Substep 1 of `layer_b_ex`: allocate `mid` with `live = blocks \cup {cur} \cup pinned`,
\* write `mid`, and if `ip % 6 == 0` push `cur` onto `blocks`.
StepMid ==
    /\ stage \in {"prelude", "loop", "weak_coda", "coda"}
    /\ substep = "mid"
    /\ ~exhausted
    /\ LET isWeak    == stage = "weak_coda"
           cSlot     == IF isWeak THEN curW ELSE cur
           cStmp     == IF isWeak THEN curWStamp ELSE curStamp
           bSeq      == IF isWeak THEN blocksW ELSE blocks
           bStmp     == IF isWeak THEN blkWStamps ELSE blkStamps
           live      == SeqSet(bSeq) \cup {cSlot} \cup PinnedForStage
           mSlot     == FindSlot(vaNext, NVec, live)
           bStart    == (ip % ResBlock) = 0
           nextStamp == writeStamp + 1
           inputsOk  == ReadInputsIntact
       IN IF mSlot = -1
          THEN /\ exhausted' = TRUE
               /\ UNCHANGED <<stage, pass, ip, substep, vaNext, cur, mid, blocks,
                              prelude, preludeBlocks, bank0, bank1, curW, blocksW,
                              writeStamp, slotStamp, curStamp, midStamp, blkStamps,
                              preludeStamp, preBlkStamps, bank0Stamp, bank1Stamp,
                              curWStamp, blkWStamps, corrupted>>
          ELSE /\ vaNext'     = NextAfterAlloc(vaNext, mSlot)
               /\ mid'        = mSlot
               /\ midStamp'   = nextStamp
               /\ writeStamp' = nextStamp
               /\ slotStamp'  = [slotStamp EXCEPT ![mSlot] = nextStamp]
               /\ corrupted'  = (corrupted \/ ~inputsOk)
               /\ substep'    = "out"
               /\ IF isWeak
                  THEN /\ blocksW'    = IF bStart THEN Append(bSeq, cSlot) ELSE bSeq
                       /\ blkWStamps' = IF bStart THEN Append(bStmp, cStmp) ELSE bStmp
                       /\ UNCHANGED <<blocks, blkStamps>>
                  ELSE /\ blocks'     = IF bStart THEN Append(bSeq, cSlot) ELSE bSeq
                       /\ blkStamps'  = IF bStart THEN Append(bStmp, cStmp) ELSE bStmp
                       /\ UNCHANGED <<blocksW, blkWStamps>>
               /\ UNCHANGED <<stage, pass, ip, cur, curStamp, prelude, preludeBlocks,
                              bank0, bank1, curW, curWStamp, preludeStamp, preBlkStamps,
                              bank0Stamp, bank1Stamp, exhausted>>

\* Substep 2 of `layer_b_ex`: FFN depth-mix reads `blocks \cup {mid}` (plus loop banks),
\* allocates `out` with `live = blocks \cup {mid} \cup pinned`, and advances the layer schedule.
StepOut ==
    /\ stage \in {"prelude", "loop", "weak_coda", "coda"}
    /\ substep = "out"
    /\ ~exhausted
    /\ LET isWeak    == stage = "weak_coda"
           bSeq      == IF isWeak THEN blocksW ELSE blocks
           bStmp     == IF isWeak THEN blkWStamps ELSE blkStamps
           live      == SeqSet(bSeq) \cup {mid} \cup PinnedForStage
           oSlot     == FindSlot(vaNext, NVec, live)
           nextStamp == writeStamp + 1
           inputsOk  == /\ slotStamp[mid] = midStamp
                        /\ SeqIntact(bSeq, bStmp)
                        /\ (stage = "loop" /\ pass > 0 =>
                              /\ slotStamp[bank0] = bank0Stamp
                              /\ slotStamp[bank1] = bank1Stamp)
       IN IF oSlot = -1
          THEN /\ exhausted' = TRUE
               /\ UNCHANGED <<stage, pass, ip, substep, vaNext, cur, mid, blocks,
                              prelude, preludeBlocks, bank0, bank1, curW, blocksW,
                              writeStamp, slotStamp, curStamp, midStamp, blkStamps,
                              preludeStamp, preBlkStamps, bank0Stamp, bank1Stamp,
                              curWStamp, blkWStamps, corrupted>>
          ELSE LET newSlotStamp == [slotStamp EXCEPT ![oSlot] = nextStamp]
               IN /\ vaNext'     = NextAfterAlloc(vaNext, oSlot)
                  /\ writeStamp' = nextStamp
                  /\ slotStamp'  = newSlotStamp
                  /\ substep'    = "mid"
                  /\ IF stage = "prelude" THEN
                        IF ip + 1 < LoopStart THEN
                            /\ ip' = ip + 1
                            /\ cur' = oSlot /\ curStamp' = nextStamp
                            /\ corrupted' = (corrupted \/ ~inputsOk)
                            /\ UNCHANGED <<stage, pass, blocks, blkStamps, prelude, preludeBlocks,
                                           preludeStamp, preBlkStamps, bank0, bank1, bank0Stamp,
                                           bank1Stamp, curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>
                        ELSE
                            /\ stage'         = "loop"
                            /\ pass'          = 0
                            /\ ip'            = LoopStart
                            /\ cur'           = oSlot /\ curStamp' = nextStamp
                            /\ prelude'       = oSlot /\ preludeStamp' = nextStamp
                            /\ preludeBlocks' = blocks /\ preBlkStamps' = blkStamps
                            /\ corrupted'     = (corrupted \/ ~inputsOk)
                            /\ UNCHANGED <<blocks, blkStamps, bank0, bank1, bank0Stamp, bank1Stamp,
                                           curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>
                     ELSE IF stage = "loop" THEN
                        IF ip + 1 < LoopEnd THEN
                            /\ ip' = ip + 1
                            /\ cur' = oSlot /\ curStamp' = nextStamp
                            /\ corrupted' = (corrupted \/ ~inputsOk)
                            /\ UNCHANGED <<stage, pass, blocks, blkStamps, prelude, preludeBlocks,
                                           preludeStamp, preBlkStamps, bank0, bank1, bank0Stamp,
                                           bank1Stamp, curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>
                        ELSE
                            LET nb0  == blocks[Len(preludeBlocks) + 2]
                                nb0S == blkStamps[Len(preludeBlocks) + 2]
                                nb1  == oSlot
                                nb1S == nextStamp
                            IN IF pass = 0 /\ RunLoopCd THEN
                                  /\ stage'      = "weak_coda"
                                  /\ ip'         = LoopEnd
                                  /\ cur'        = oSlot /\ curStamp' = nextStamp
                                  /\ bank0'      = nb0   /\ bank0Stamp' = nb0S
                                  /\ bank1'      = nb1   /\ bank1Stamp' = nb1S
                                  /\ curW'       = oSlot /\ curWStamp' = nextStamp
                                  /\ blocksW'    = blocks /\ blkWStamps' = blkStamps
                                  /\ corrupted'  = (corrupted \/ ~inputsOk)
                                  /\ UNCHANGED <<pass, blocks, blkStamps, prelude, preludeBlocks,
                                                 preludeStamp, preBlkStamps, mid, midStamp, exhausted>>
                               ELSE IF pass + 1 < NPass THEN
                                  LET rewindOk == /\ newSlotStamp[prelude] = preludeStamp
                                                  /\ \A i \in 1..Len(preludeBlocks) :
                                                        newSlotStamp[preludeBlocks[i]] = preBlkStamps[i]
                                  IN /\ pass'       = pass + 1
                                     /\ ip'         = LoopStart
                                     /\ cur'        = prelude       /\ curStamp'  = preludeStamp
                                     /\ blocks'     = preludeBlocks /\ blkStamps' = preBlkStamps
                                     /\ bank0'      = nb0           /\ bank0Stamp' = nb0S
                                     /\ bank1'      = nb1           /\ bank1Stamp' = nb1S
                                     /\ corrupted'  = (corrupted \/ ~inputsOk \/ ~rewindOk)
                                     /\ UNCHANGED <<stage, prelude, preludeBlocks, preludeStamp, preBlkStamps,
                                                    curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>
                               ELSE
                                  /\ stage'      = "coda"
                                  /\ ip'         = LoopEnd
                                  /\ cur'        = oSlot /\ curStamp' = nextStamp
                                  /\ bank0'      = nb0   /\ bank0Stamp' = nb0S
                                  /\ bank1'      = nb1   /\ bank1Stamp' = nb1S
                                  /\ corrupted'  = (corrupted \/ ~inputsOk)
                                  /\ UNCHANGED <<pass, blocks, blkStamps, prelude, preludeBlocks,
                                                 preludeStamp, preBlkStamps, curW, blocksW, curWStamp,
                                                 blkWStamps, mid, midStamp, exhausted>>
                     ELSE IF stage = "weak_coda" THEN
                        IF ip + 1 < NPhys THEN
                            /\ ip' = ip + 1
                            /\ curW' = oSlot /\ curWStamp' = nextStamp
                            /\ corrupted' = (corrupted \/ ~inputsOk)
                            /\ UNCHANGED <<stage, pass, cur, curStamp, blocks, blkStamps,
                                           prelude, preludeBlocks, preludeStamp, preBlkStamps,
                                           bank0, bank1, bank0Stamp, bank1Stamp, blocksW, blkWStamps,
                                           mid, midStamp, exhausted>>
                        ELSE
                            \* Weak output mix reads `blocksW \cup {oSlot}`, then rewinds to `pass = 1`
                            \* reading `prelude`, `preludeBlocks`, `bank0`, and `bank1`.
                            LET weakMixOk == \A i \in 1..Len(blocksW) : newSlotStamp[blocksW[i]] = blkWStamps[i]
                                rewindOk  == /\ newSlotStamp[prelude] = preludeStamp
                                             /\ newSlotStamp[bank0]   = bank0Stamp
                                             /\ newSlotStamp[bank1]   = bank1Stamp
                                             /\ \A i \in 1..Len(preludeBlocks) :
                                                    newSlotStamp[preludeBlocks[i]] = preBlkStamps[i]
                            IN /\ stage'      = "loop"
                               /\ pass'       = 1
                               /\ ip'         = LoopStart
                               /\ cur'        = prelude       /\ curStamp'   = preludeStamp
                               /\ blocks'     = preludeBlocks /\ blkStamps'  = preBlkStamps
                               /\ curW'       = oSlot         /\ curWStamp'  = nextStamp
                               /\ corrupted'  = (corrupted \/ ~inputsOk \/ ~weakMixOk \/ ~rewindOk)
                               /\ UNCHANGED <<prelude, preludeBlocks, preludeStamp, preBlkStamps,
                                              bank0, bank1, bank0Stamp, bank1Stamp, blocksW, blkWStamps,
                                              mid, midStamp, exhausted>>
                     ELSE \* stage = "coda"
                        IF ip + 1 < NPhys THEN
                            /\ ip' = ip + 1
                            /\ cur' = oSlot /\ curStamp' = nextStamp
                            /\ corrupted' = (corrupted \/ ~inputsOk)
                            /\ UNCHANGED <<stage, pass, blocks, blkStamps, prelude, preludeBlocks,
                                           preludeStamp, preBlkStamps, bank0, bank1, bank0Stamp,
                                           bank1Stamp, curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>
                        ELSE
                            LET outMixOk == /\ Len(blocks) = 4
                                            /\ \A i \in 1..Len(blocks) : newSlotStamp[blocks[i]] = blkStamps[i]
                            IN /\ stage'     = "done"
                               /\ cur'       = oSlot /\ curStamp' = nextStamp
                               /\ corrupted' = (corrupted \/ ~inputsOk \/ ~outMixOk)
                               /\ UNCHANGED <<pass, ip, blocks, blkStamps, prelude, preludeBlocks,
                                              preludeStamp, preBlkStamps, bank0, bank1, bank0Stamp,
                                              bank1Stamp, curW, blocksW, curWStamp, blkWStamps, mid, midStamp, exhausted>>

DoneStutter ==
    /\ (stage = "done" \/ exhausted)
    /\ UNCHANGED vars

Next == StepMid \/ StepOut \/ DoneStutter

Spec == Init /\ [][Next]_vars

\* ---- Invariants -----------------------------------------------------------

TypeOK ==
    /\ stage \in Stages
    /\ pass \in 0..(NPass - 1)
    /\ ip \in 0..(NPhys - 1)
    /\ substep \in {"mid", "out"}
    /\ vaNext \in 0..(NVec - 1)
    /\ cur \in 0..(NVec - 1)
    /\ mid \in 0..(NVec - 1)
    /\ exhausted \in BOOLEAN
    /\ corrupted \in BOOLEAN

NoPoolExhaustion == ~exhausted

NeverOverwritesLive ==
    /\ ~corrupted
    /\ (stage = "done" =>
          /\ Len(blocks) = 4
          /\ slotStamp[cur] = curStamp
          /\ SeqIntact(blocks, blkStamps)
          /\ (RunLoopCd => Len(blocksW) = 4))

=============================================================================
