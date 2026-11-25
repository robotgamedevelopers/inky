---------------------------- MODULE CacheScheduler ----------------------------
\* Coupled model of the block cache and lane scheduler (docs/DESIGN.md,
\* sections "Blocks", "Scheduler", and "Models").
\*
\* Vocabulary matches specs/BlockLayout.tla:
\*   - Kinds == {"full", "window"}: the two cache block kinds;
\*   - BlockSize (B in BlockLayout), WindowSize (W in BlockLayout): positions
\*     per block and sliding-window width in positions;
\*   - Window, History, SpanWindow, Full, SpanBound: exact logical block sets;
\*   - start point (KV blocks + recurrent `stateTier`), multi-block prefill span,
\*     pin, prefill-only sealed blocks, decode-private unsealed blocks, RAM/SSD tiers;
\*   - MaxBypass + independent unclamped `ghostBypass`: starvation bound W covering
\*     Ready, Restoring, Admitted, and Queued lanes.

EXTENDS Integers, FiniteSets, Sequences, TLC

CONSTANTS
    Lanes,          \* Set of lane slots, e.g. {a, b}
    Prefixes,       \* Set of token prefixes, e.g. {p1, p2}
    RequestLengths, \* Allowed request target lengths, e.g. {2, 4}
    BlockSize,      \* Positions per block (64 in production, 1 or 2 here)
    WindowSize,     \* Sliding-window positions (1024 in production, 1..3 here)
    MaxPos,         \* Maximum context positions (4 or 8 here)
    MaxSpan,        \* Maximum tokens per prefill span (may cross block boundaries!)
    StepTokenBudget,\* Aggregate token budget across all spans in one step
    MemBudget,      \* Memory budget in KV blocks
    DiskBudget,     \* Disk budget in KV blocks
    MaxQueue,       \* Queue limit before 429 refusal
    MaxBypass,      \* Bypass bound W for Ready, Restoring, Admitted, and Queued lanes
    Bug             \* "none" or reversed-witness selector (see ASSUME below)

ASSUME
    /\ BlockSize > 0
    /\ WindowSize >= BlockSize
    /\ MaxPos >= BlockSize
    /\ MaxPos % BlockSize = 0
    /\ RequestLengths # {}
    /\ \A r \in RequestLengths : r \in 1..MaxPos /\ r % BlockSize = 0
    /\ MaxSpan \in 1..MaxPos
    /\ StepTokenBudget >= MaxSpan
    /\ MemBudget >= 2
    /\ DiskBudget >= 2
    /\ MaxQueue >= 0
    /\ MaxBypass >= 1
    /\ Bug \in {
           "none",
           "cancel_in_step",
           "cancel_in_restore",
           "restore_split",
           "boundary_window_dropped",
           "unpruned_window",
           "no_decode_priority",
           "queued_bypass",
           "deferred_tied_429",
           "eager_hot_writeback",
           "drop_start_state",
           "seal_decode",
           "writeback_clobber",
           "no_spill",
           "unsealed_tree",
           "dirty_priv_tail",
           "stale_refcount",
           "over_step_budget",
           "queue_overflow",
           "mutate_sealed"
       }

\* Default false so existing atomic-restore configs keep identical state graphs;
\* overridden via `AsyncRestore = TRUE` in `CacheSchedulerCancelRestore.cfg`,
\* `CancelInRestore.cfg`, and `TraceCheck.cfg` to model async disk restore I/O
\* and cancellation during restore (`U88`).
AsyncRestore == FALSE

Kinds == {"full", "window"}
Tiers == {"none", "memory", "disk", "both"}
BlkIdx == 0..((MaxPos - 1) \div BlockSize)
Boundaries == {k \in 1..MaxPos : k % BlockSize = 0}
DefaultPrefix == CHOOSE p \in Prefixes : TRUE
DefaultTarget == CHOOSE r \in RequestLengths : TRUE
ProbeLane == CHOOSE l \in Lanes : TRUE

Max(a, b) == IF a >= b THEN a ELSE b
Min(a, b) == IF a <= b THEN a ELSE b

\* Canonical block identifiers:
\*   - TreeIds: sealed prefill blocks in the shared prefix tree, keyed by (prefix, kind, idx);
\*   - PrivIds: private unsealed blocks written by one lane (lane, kind, idx), supporting
\*     both multi-block prefill spans and unsealed decode-written blocks (Decision 4).
TreeIds == {<<"tree", p, k, i>> : p \in Prefixes, k \in Kinds, i \in BlkIdx}
PrivIds == {<<"priv", l, k, i>> : l \in Lanes, k \in Kinds, i \in BlkIdx}
BlockIds == TreeIds \cup PrivIds
NoBlock == <<"none">>

BlockKind(b) == b[3]

\* Logical block-range helpers (identical to specs/BlockLayout.tla and serve/block_cache.rs:100).
Ceil(n, d) == (n + d - 1) \div d
SpanBound(c) == Ceil(WindowSize - 1 + c, BlockSize) + 1
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div BlockSize : p \in lo..hi}
History(n) == Window(IF n >= WindowSize THEN n - WindowSize + 1 ELSE 0, n - 1)
SpanWindow(n, c) == Window(IF n >= WindowSize THEN n - WindowSize + 1 ELSE 0, n + c - 1)
Full(n) == Window(0, n - 1)

TotalFullBlocks(m) == Cardinality(Full(m))
MaxWinBlocks(m) ==
    LET bound == SpanBound(MaxSpan)
        allW  == Cardinality(Full(m))
    IN IF bound < allW THEN bound ELSE allW
WorstCaseBlocks(m) == TotalFullBlocks(m) + MaxWinBlocks(m)

\* Dedicated RAM and SSD budgets for kept start-point recurrent states (`cs`).
MaxSpanBoundaries == Ceil(MaxSpan, BlockSize)
StateMemBudget == Max(2, (Cardinality(Lanes) - 1) + MaxSpanBoundaries)
StateDiskBudget == Max(DiskBudget, StateMemBudget)

EmptyData == [i \in 0..(BlockSize - 1) |-> 0]
FullData  == [i \in 0..(BlockSize - 1) |-> 1]
EmptyBlock == [
    tier     |-> "none",
    sealed   |-> FALSE,
    idx      |-> 0,
    length   |-> 0,
    data     |-> EmptyData,
    refCount |-> 0
]
EmptyBlkMap == [i \in BlkIdx |-> NoBlock]
EmptyStart == [
    kept      |-> FALSE,
    stateTier |-> "none",
    full      |-> {},
    win       |-> {}
]

LaneLifecycleStates == {
    "Queued", "Admitted", "Restoring", "Ready",
    "InStep", "Cancelling", "Cancelled", "Finished"
}
SlotStates == {"Idle", "Refused"} \cup LaneLifecycleStates
MemHoldingStates == {"Restoring", "Ready", "InStep", "Cancelling"}
AdmittedStates == {"Admitted"} \cup MemHoldingStates
WaitingStates == {"Queued", "Admitted", "Restoring", "Ready"}

RECURSIVE SumBy(_, _)
SumBy(S, f) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE v \in S : TRUE IN f[x] + SumBy(S \ {x}, f)

VARIABLES
    blocks,         \* [BlockIds -> BlockRecord]
    startPoints,    \* [Prefixes \X Boundaries -> StartPointRecord]
    laneState,      \* [Lanes -> SlotStates]
    lanePrefix,     \* [Lanes -> Prefixes]
    laneTarget,     \* [Lanes -> RequestLengths]
    lanePos,        \* [Lanes -> 0..MaxPos]
    laneFull,       \* [Lanes -> [BlkIdx -> BlockIds \cup {NoBlock}]]
    laneWin,        \* [Lanes -> [BlkIdx -> BlockIds \cup {NoBlock}]]
    reservation,    \* [Lanes -> 0..MemBudget]
    admitBudget,    \* [Lanes -> 0..MemBudget]
    waitTicks,      \* [Lanes -> 0..(MaxBypass + 1)] scheduler's own counter
    ghostBypass,    \* [Lanes -> Nat] independent unclamped pass-over ghost counter
    queueSeq,       \* Seq(Lanes): FIFO arrival order of currently Queued lanes (no duplicates)
    stepActive,     \* BOOLEAN: one step in flight
    stepLanes,      \* Subset of Lanes in the current step
    stepSpan,       \* [Lanes -> 0..MaxSpan]
    stepPins,       \* Subset of BlockIds pinned in memory by the step's spans
    stepIO,         \* Subset of BlockIds with pending disk I/O
    gpuDone,        \* BOOLEAN: GPU work of the in-flight step has completed
    restoreIO       \* [Lanes -> BlockIds \cup {NoBlock}]: in-flight async disk restore read (`U88`)

vars == <<
    blocks, startPoints,
    laneState, lanePrefix, laneTarget, lanePos, laneFull, laneWin,
    reservation, admitBudget, waitTicks, ghostBypass, queueSeq,
    stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO
>>

RemoveLane(seq, x) == SelectSeq(seq, LAMBDA y : y # x)
QueueIdx(l)        == CHOOSE i \in 1..Len(queueSeq) : queueSeq[i] = l
OlderQueued(a, b)  == QueueIdx(a) < QueueIdx(b)

\* ---- Derived Queries & Referrer Accounting --------------------------------

InMemory(b) == blocks[b].tier \in {"memory", "both"}
OnDisk(b)   == blocks[b].tier \in {"disk", "both"}
Present(b)  == blocks[b].tier # "none"

StateInMem(pk)   == IF pk \in DOMAIN startPoints THEN startPoints[pk].stateTier \in {"memory", "both"} ELSE FALSE
StateOnDisk(pk)  == IF pk \in DOMAIN startPoints THEN startPoints[pk].stateTier \in {"disk", "both"} ELSE FALSE
StatePresent(pk) == IF pk \in DOMAIN startPoints THEN startPoints[pk].stateTier # "none" ELSE FALSE

MemUsed       == Cardinality({b \in BlockIds : InMemory(b)})
DiskUsed      == Cardinality({b \in BlockIds : OnDisk(b)})
StateMemUsed  == Cardinality({pk \in Prefixes \X Boundaries : startPoints[pk].kept /\ StateInMem(pk)})
StateDiskUsed == Cardinality({pk \in Prefixes \X Boundaries : startPoints[pk].kept /\ StateOnDisk(pk)})

LaneBlockSet(f, w) ==
    ({f[i] : i \in BlkIdx} \cup {w[i] : i \in BlkIdx}) \ {NoBlock}

LaneBlocks(l) == LaneBlockSet(laneFull[l], laneWin[l])

OwningLaneStates == AdmittedStates
LaneReferrersOf(b, st, lf, lw) ==
    {<<"lane", l>> : l \in {q \in Lanes : st[q] \in OwningLaneStates /\ b \in LaneBlockSet(lf[q], lw[q])}}

StartReferrersOf(b, sp) ==
    {<<"start", pk[1], pk[2]>> : pk \in {x \in DOMAIN sp :
        sp[x].kept /\ (b \in sp[x].full \/ b \in sp[x].win)}}

AllReferrersOf(b, st, lf, lw, sp) ==
    LaneReferrersOf(b, st, lf, lw) \cup StartReferrersOf(b, sp)

LaneReferrers(b)  == LaneReferrersOf(b, laneState, laneFull, laneWin)
StartReferrers(b) == StartReferrersOf(b, startPoints)
Referrers(b)      == AllReferrersOf(b, laneState, laneFull, laneWin, startPoints)

SyncRefCounts(blks, st, lf, lw, sp) ==
    [b \in BlockIds |->
        IF blks[b].tier = "none" THEN EmptyBlock
        ELSE LET refs == Cardinality(AllReferrersOf(b, st, lf, lw, sp))
             IN IF ~blks[b].sealed /\ refs = 0 THEN EmptyBlock
                ELSE [blks[b] EXCEPT !.refCount =
                        IF Bug = "stale_refcount" /\ refs > 1 THEN 1 ELSE refs]]

ReservedTotal       == SumBy({l \in Lanes : laneState[l] \in MemHoldingStates}, reservation)
AdmittedTotal       == SumBy({l \in Lanes : laneState[l] \in AdmittedStates}, admitBudget)
QueuedLanes         == {l \in Lanes : laneState[l] = "Queued"}
ReadyLanes          == {l \in Lanes : laneState[l] = "Ready"}
RestoringLanes      == {l \in Lanes : laneState[l] = "Restoring"}
RestoringStartNeed  == {<<lanePrefix[r], lanePos[r]>> : r \in {q \in RestoringLanes : lanePos[q] > 0}}
ActiveRestoring     == {r \in RestoringLanes :
                            (\E rb \in LaneBlocks(r) : blocks[rb].tier = "disk")
                            \/ (StateOnDisk(<<lanePrefix[r], lanePos[r]>>)
                                /\ ~StateInMem(<<lanePrefix[r], lanePos[r]>>))}
SteppableReady      == {l \in ReadyLanes : lanePos[l] < laneTarget[l]}
WindowPruned(l)     == \A i \in BlkIdx : laneWin[l][i] # NoBlock => i \in History(lanePos[l])

\* Positions below PromptEnd(t) are prompt prefill (spans up to MaxSpan);
\* positions at or above PromptEnd(t) are token-by-token decode (1-token spans).
\* Decision 4: decode-written blocks are NEVER sealed into TreeIds or kept as start points.
PromptEnd(t)    == IF t > BlockSize THEN t - BlockSize ELSE 0
MatchCeiling(t) == IF Bug = "seal_decode" THEN t ELSE PromptEnd(t)
IsPrefilling(l) == laneState[l] = "Ready" /\ lanePos[l] < PromptEnd(laneTarget[l])
IsDecoding(l)   == laneState[l] = "Ready" /\ lanePos[l] >= PromptEnd(laneTarget[l]) /\ lanePos[l] < laneTarget[l]

NeedFullForSpan(l, c) == {i \in Full(lanePos[l] + c) : laneFull[l][i] = NoBlock}
NeedWinForSpan(l, c)  == {i \in SpanWindow(lanePos[l], c) : laneWin[l][i] = NoBlock}
SpanAllocCount(l, c)  == Cardinality(NeedFullForSpan(l, c)) + Cardinality(NeedWinForSpan(l, c))

MaxLegalSpan(l) ==
    IF IsDecoding(l) THEN 1
    ELSE Min(MaxSpan, Min(StepTokenBudget, PromptEnd(laneTarget[l]) - lanePos[l]))

NeedAllocMap == [l \in Lanes |->
    IF laneState[l] = "Ready" /\ lanePos[l] < laneTarget[l]
    THEN SpanAllocCount(l, 1)
    ELSE 0]

MaxSpanAllocMap == [l \in Lanes |->
    IF laneState[l] = "Ready" /\ lanePos[l] < laneTarget[l]
    THEN SpanAllocCount(l, MaxLegalSpan(l))
    ELSE 0]

CurrentWorkingSet(l) ==
    Cardinality(LaneBlocks(l)) +
        (IF lanePos[l] < laneTarget[l] THEN
            (IF laneFull[l][lanePos[l] \div BlockSize] = NoBlock THEN 1 ELSE 0) +
            (IF laneWin[l][lanePos[l] \div BlockSize] = NoBlock THEN 1 ELSE 0)
         ELSE 0)

ColdMemBlocks ==
    {b \in BlockIds : InMemory(b) /\ LaneReferrers(b) = {} /\ b \notin stepIO}

ColdDiskBlocks ==
    {b \in BlockIds : OnDisk(b) /\ LaneReferrers(b) = {} /\ b \notin stepIO}

ColdMemStates ==
    {pk \in Prefixes \X Boundaries :
        startPoints[pk].kept /\ StateInMem(pk) /\ pk \notin RestoringStartNeed}

ColdDiskStates ==
    {pk \in Prefixes \X Boundaries :
        startPoints[pk].kept /\ StateOnDisk(pk) /\ pk \notin RestoringStartNeed}

MatchingBoundaries(p, maxP) ==
    {k \in Boundaries : k <= maxP /\ startPoints[<<p, k>>].kept}

BestBoundary(p, maxP) ==
    LET ms == MatchingBoundaries(p, maxP)
    IN IF ms = {} THEN 0
       ELSE CHOOSE k \in ms : \A j \in ms : k >= j

\* ---- Initial State --------------------------------------------------------

Init ==
    /\ blocks      = [b \in BlockIds |-> EmptyBlock]
    /\ startPoints = [pk \in Prefixes \X Boundaries |-> EmptyStart]
    /\ laneState   = [l \in Lanes |-> "Idle"]
    /\ lanePrefix  = [l \in Lanes |-> DefaultPrefix]
    /\ laneTarget  = [l \in Lanes |-> DefaultTarget]
    /\ lanePos     = [l \in Lanes |-> 0]
    /\ laneFull    = [l \in Lanes |-> EmptyBlkMap]
    /\ laneWin     = [l \in Lanes |-> EmptyBlkMap]
    /\ reservation = [l \in Lanes |-> 0]
    /\ admitBudget = [l \in Lanes |-> 0]
    /\ waitTicks   = [l \in Lanes |-> 0]
    /\ ghostBypass = [l \in Lanes |-> 0]
    /\ queueSeq    = << >>
    /\ stepActive  = FALSE
    /\ stepLanes   = {}
    /\ stepSpan    = [l \in Lanes |-> 0]
    /\ stepPins    = {}
    /\ stepIO      = {}
    /\ gpuDone     = FALSE
    /\ restoreIO   = [l \in Lanes |-> NoBlock]

\* ---- Request Arrival, Queueing, Admission & 429 Refusal -------------------
\* Note on `serve.rs` 429 paths (`U60`):
\*   1. Capacity / queue-limit refusal: modelled below in `Arrive` (`wc > MemBudget`
\*      or `Cardinality(QueuedLanes) = MaxQueue` without direct-admit headroom -> `"Refused"`).
\*      Inside this untimed scheduler model, `ActionNoQueuedRefusal` proves that capacity
\*      accounting never retroactively evicts an accepted `"Queued"` request to `"Refused"`.
\*   2. Wall-clock queue-wait timeout: `src/serve.rs` (`AdmissionGate`, `QUEUE_WAIT_TIMEOUT = 60s`)
\*      also returns HTTP 429 (`QUEUE_TIMED_OUT`) if a waiter's 60 s wall-clock deadline expires
\*      before a lane permit opens; real-time HTTP deadlines live outside this untimed spec
\*      (and drop the waiter from the queue just like client `Cancel`).

DiskStartCount(p, reqLen) ==
    LET k == BestBoundary(p, MatchCeiling(reqLen))
    IN IF k = 0 THEN 0
       ELSE Cardinality({b \in startPoints[<<p, k>>].full \cup startPoints[<<p, k>>].win : blocks[b].tier = "disk"})

CanFitRequest(p, reqLen) ==
    LET wc       == WorstCaseBlocks(reqLen)
        splitBug == Bug = "restore_split" /\ DiskStartCount(p, reqLen) >= 2
    IN IF splitBug
       THEN AdmittedTotal + 1 <= MemBudget
       ELSE AdmittedTotal + wc <= MemBudget

CanFitAdmit(l) == CanFitRequest(lanePrefix[l], laneTarget[l])

CanAdmitNow(l) ==
    /\ CanFitAdmit(l)
    /\ \A q \in (QueuedLanes \ {l}) :
           CanFitAdmit(q) =>
               (waitTicks[l] > waitTicks[q] \/ (waitTicks[l] = waitTicks[q] /\ OlderQueued(l, q)))
    /\ \A q \in (QueuedLanes \ {l}) :
           waitTicks[q] >= MaxBypass =>
               (OlderQueued(l, q) \/ waitTicks[l] > waitTicks[q])

CanAdvanceAdmitted(l) ==
    /\ laneState[l] = "Admitted"
    /\ (   (Bug = "restore_split" /\ DiskStartCount(lanePrefix[l], laneTarget[l]) >= 2)
        \/ ReservedTotal + WorstCaseBlocks(laneTarget[l]) <= MemBudget)

StartStateReady(l) ==
    IF lanePos[l] = 0 THEN TRUE ELSE StateInMem(<<lanePrefix[l], lanePos[l]>>)

CanAdvanceRestoring(l) ==
    /\ laneState[l] = "Restoring"
    /\ (   \A b \in LaneBlocks(l) : InMemory(b)
        \/ MemUsed < MemBudget
        \/ ColdMemBlocks # {})
    /\ (   StartStateReady(l)
        \/ StateMemUsed < StateMemBudget
        \/ ColdMemStates # {})
    /\ (((\A b \in LaneBlocks(l) : InMemory(b)) /\ StartStateReady(l) /\ Bug = "restore_split")
           => ReservedTotal + (WorstCaseBlocks(laneTarget[l]) - reservation[l]) <= MemBudget)

ActionableWaiting ==
    SteppableReady
        \cup {l \in Lanes : CanAdvanceAdmitted(l)}
        \cup {l \in Lanes : CanAdvanceRestoring(l)}
        \cup {q \in QueuedLanes : CanAdmitNow(q)}

StepBypassed(chosen) == ActionableWaiting \ chosen

Arrive(l, p, reqLen) ==
    /\ laneState[l] \in {"Idle", "Refused", "Cancelled", "Finished"}
    /\ p \in Prefixes
    /\ reqLen \in RequestLengths
    /\ LET wc       == WorstCaseBlocks(reqLen)
           splitBug == Bug = "restore_split" /\ DiskStartCount(p, reqLen) >= 2
           qCap     == IF Bug = "queue_overflow" THEN MaxQueue + 1 ELSE MaxQueue
           canEnqueue == wc <= MemBudget /\ Cardinality(QueuedLanes) < qCap
           canDirectAdmit ==
               /\ wc <= MemBudget
               /\ Cardinality(QueuedLanes) = MaxQueue
               /\ CanFitRequest(p, reqLen)
               /\ \A q \in QueuedLanes : ~CanFitAdmit(q) /\ waitTicks[q] < MaxBypass
       IN IF canEnqueue
          THEN /\ laneState'   = [laneState EXCEPT ![l] = "Queued"]
               /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = p]
               /\ laneTarget'  = [laneTarget EXCEPT ![l] = reqLen]
               /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
               /\ laneFull'    = [laneFull EXCEPT ![l] = EmptyBlkMap]
               /\ laneWin'     = [laneWin EXCEPT ![l] = EmptyBlkMap]
               /\ reservation' = [reservation EXCEPT ![l] = 0]
               /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
               /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
               /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
               /\ queueSeq'    = Append(queueSeq, l)
          ELSE IF canDirectAdmit
          THEN /\ laneState'   = [laneState EXCEPT ![l] = "Admitted"]
               /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = p]
               /\ laneTarget'  = [laneTarget EXCEPT ![l] = reqLen]
               /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
               /\ laneFull'    = [laneFull EXCEPT ![l] = EmptyBlkMap]
               /\ laneWin'     = [laneWin EXCEPT ![l] = EmptyBlkMap]
               /\ reservation' = [reservation EXCEPT ![l] = 0]
               /\ admitBudget' = [admitBudget EXCEPT ![l] = IF splitBug THEN 1 ELSE wc]
               /\ waitTicks'   = [q \in Lanes |->
                                   IF q = l THEN 0
                                   ELSE IF q \in QueuedLanes THEN waitTicks[q] + 1
                                   ELSE waitTicks[q]]
               /\ ghostBypass' = [q \in Lanes |->
                                   IF q = l THEN 0
                                   ELSE IF q \in QueuedLanes THEN ghostBypass[q] + 1
                                   ELSE ghostBypass[q]]
               /\ UNCHANGED queueSeq
          ELSE /\ laneState'   = [laneState EXCEPT ![l] = "Refused"]
               /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = DefaultPrefix]
               /\ laneTarget'  = [laneTarget EXCEPT ![l] = DefaultTarget]
               /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
               /\ laneFull'    = [laneFull EXCEPT ![l] = EmptyBlkMap]
               /\ laneWin'     = [laneWin EXCEPT ![l] = EmptyBlkMap]
               /\ reservation' = [reservation EXCEPT ![l] = 0]
               /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
               /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
               /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
               /\ UNCHANGED queueSeq
    /\ UNCHANGED <<blocks, startPoints, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

Refuse429(l) ==
    /\ Bug = "deferred_tied_429"
    /\ laneState[l] = "Queued"
    /\ ~CanAdmitNow(l)
    /\ \A q \in QueuedLanes : waitTicks[l] <= waitTicks[q]
    /\ laneState'   = [laneState EXCEPT ![l] = "Refused"]
    /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = DefaultPrefix]
    /\ laneTarget'  = [laneTarget EXCEPT ![l] = DefaultTarget]
    /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
    /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
    /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
    /\ queueSeq'    = RemoveLane(queueSeq, l)
    /\ UNCHANGED <<blocks, startPoints, laneFull, laneWin, reservation, admitBudget,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

Admit(l) ==
    /\ laneState[l] = "Queued"
    /\ CanAdmitNow(l)
    /\ LET wc       == WorstCaseBlocks(laneTarget[l])
           splitBug == Bug = "restore_split" /\ DiskStartCount(lanePrefix[l], laneTarget[l]) >= 2
           bypassed == {q \in (QueuedLanes \ {l}) :
                            OlderQueued(q, l) /\ waitTicks[q] < MaxBypass}
       IN /\ laneState'   = [laneState EXCEPT ![l] = "Admitted"]
          /\ admitBudget' = [admitBudget EXCEPT ![l] = IF splitBug THEN 1 ELSE wc]
          /\ waitTicks'   = [q \in Lanes |->
                                IF q = l THEN waitTicks[l]
                                ELSE IF q \in bypassed THEN waitTicks[q] + 1
                                ELSE waitTicks[q]]
          \* Independent unclamped ghost count of every older queued lane overtaken by `l`:
          /\ ghostBypass' = [q \in Lanes |->
                                IF q = l THEN ghostBypass[l]
                                ELSE IF q \in QueuedLanes /\ OlderQueued(q, l) THEN ghostBypass[q] + 1
                                ELSE ghostBypass[q]]
          /\ queueSeq'    = RemoveLane(queueSeq, l)
    /\ UNCHANGED <<blocks, startPoints, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

\* ---- Prefix Matching & Atomic Working-Set Reservation (Restoring) ---------

BeginRestore(l) ==
    /\ laneState[l] = "Admitted"
    /\ LET p        == lanePrefix[l]
           wc       == WorstCaseBlocks(laneTarget[l])
           k        == BestBoundary(p, MatchCeiling(laneTarget[l]))
           sp       == IF k = 0 THEN EmptyStart ELSE startPoints[<<p, k>>]
           winKeep  == IF Bug = "boundary_window_dropped" THEN {} ELSE sp.win
           diskCnt  == Cardinality({b \in sp.full \cup winKeep : blocks[b].tier = "disk"})
           splitBug == Bug = "restore_split" /\ diskCnt >= 2
           nf       == [i \in BlkIdx |->
                         IF \E b \in sp.full : b[4] = i
                         THEN CHOOSE b \in sp.full : b[4] = i
                         ELSE NoBlock]
           nw       == [i \in BlkIdx |->
                         IF \E b \in winKeep : b[4] = i
                         THEN CHOOSE b \in winKeep : b[4] = i
                         ELSE NoBlock]
           res      == IF splitBug
                       THEN Cardinality({b \in sp.full \cup winKeep : InMemory(b)})
                       ELSE wc
           nst      == [laneState EXCEPT ![l] = "Restoring"]
           nlf      == [laneFull EXCEPT ![l] = nf]
           nlw      == [laneWin EXCEPT ![l] = nw]
       IN /\ (splitBug \/ ReservedTotal + res <= MemBudget)
          /\ laneState'   = nst
          /\ lanePos'     = [lanePos EXCEPT ![l] = k]
          /\ laneFull'    = nlf
          /\ laneWin'     = nlw
          /\ reservation' = [reservation EXCEPT ![l] = res]
          /\ blocks'      = SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
    /\ UNCHANGED <<startPoints, lanePrefix, laneTarget, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

RestoreBlock(l, b) ==
    /\ laneState[l] = "Restoring"
    /\ restoreIO[l] = NoBlock
    /\ b \in LaneBlocks(l)
    /\ blocks[b].tier = "disk"
    /\ MemUsed < MemBudget
    /\ \A r \in ActiveRestoring : waitTicks[l] >= waitTicks[r]
    /\ blocks' = [blocks EXCEPT ![b].tier = "both"]
    /\ reservation' = [reservation EXCEPT ![l] =
                        IF Bug = "restore_split" /\ @ < WorstCaseBlocks(laneTarget[l])
                        THEN @ + 1
                        ELSE @]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, admitBudget, waitTicks, ghostBypass, queueSeq,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

BeginRestoreIO(l, b) ==
    /\ AsyncRestore
    /\ laneState[l] = "Restoring"
    /\ restoreIO[l] = NoBlock
    /\ b \in LaneBlocks(l)
    /\ blocks[b].tier = "disk"
    /\ MemUsed < MemBudget
    /\ \A r \in ActiveRestoring : waitTicks[l] >= waitTicks[r]
    /\ restoreIO' = [restoreIO EXCEPT ![l] = b]
    /\ UNCHANGED <<blocks, startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

CompleteRestoreIO(l) ==
    /\ restoreIO[l] # NoBlock
    /\ LET b == restoreIO[l]
       IN IF laneState[l] = "Restoring"
          THEN /\ (blocks[b].tier = "disk" => MemUsed < MemBudget)
               /\ blocks' = [blocks EXCEPT ![b].tier = IF @ = "disk" THEN "both" ELSE @]
               /\ restoreIO' = [restoreIO EXCEPT ![l] = NoBlock]
               /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                              laneFull, laneWin, reservation, admitBudget, waitTicks,
                              ghostBypass, queueSeq, stepActive, stepLanes, stepSpan,
                              stepPins, stepIO, gpuDone>>
          ELSE /\ laneState[l] = "Cancelling"
               /\ l \notin stepLanes
               /\ LET nst == [laneState EXCEPT ![l] = "Cancelled"]
                      nlf == [laneFull EXCEPT ![l] = EmptyBlkMap]
                      nlw == [laneWin EXCEPT ![l] = EmptyBlkMap]
                      blks1 == SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
                  IN /\ laneState'   = nst
                     /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = DefaultPrefix]
                     /\ laneTarget'  = [laneTarget EXCEPT ![l] = DefaultTarget]
                     /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
                     /\ laneFull'    = nlf
                     /\ laneWin'     = nlw
                     /\ reservation' = [reservation EXCEPT ![l] = 0]
                     /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
                     /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
                     /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
                     /\ restoreIO'   = [restoreIO EXCEPT ![l] = NoBlock]
                     /\ blocks'      = blks1
               /\ UNCHANGED <<startPoints, queueSeq, stepActive, stepLanes, stepSpan,
                              stepPins, stepIO, gpuDone>>

RestoreStartState(l) ==
    /\ laneState[l] = "Restoring"
    /\ restoreIO[l] = NoBlock
    /\ lanePos[l] \in Boundaries
    /\ LET pk == <<lanePrefix[l], lanePos[l]>>
       IN /\ startPoints[pk].kept
          /\ startPoints[pk].stateTier = "disk"
          /\ StateMemUsed < StateMemBudget
          /\ \A r \in ActiveRestoring : waitTicks[l] >= waitTicks[r]
          /\ startPoints' = [startPoints EXCEPT ![pk].stateTier = "both"]
    /\ UNCHANGED <<blocks, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

FinishRestore(l) ==
    /\ laneState[l] = "Restoring"
    /\ restoreIO[l] = NoBlock
    /\ \A b \in LaneBlocks(l) : InMemory(b)
    /\ StartStateReady(l)
    /\ (Bug = "restore_split" =>
            ReservedTotal + (WorstCaseBlocks(laneTarget[l]) - reservation[l]) <= MemBudget)
    /\ laneState'   = [laneState EXCEPT ![l] = "Ready"]
    /\ reservation' = [reservation EXCEPT ![l] =
                        IF Bug = "restore_split" THEN WorstCaseBlocks(laneTarget[l]) ELSE @]
    /\ UNCHANGED <<blocks, startPoints, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

\* ---- Lazy Spill & Tier Eviction Under Memory / Disk Pressure --------------

MemPressure ==
    /\ ColdMemBlocks # {}
    /\ \/ (MemUsed >= MemBudget /\ \E l \in RestoringLanes : \E b \in LaneBlocks(l) : ~InMemory(b))
       \/ (SteppableReady # {} /\ MemUsed + SumBy(SteppableReady, MaxSpanAllocMap) > MemBudget)

NewBoundariesInSpan(l, c) ==
    {<<lanePrefix[l], (bi + 1) * BlockSize>> :
        bi \in {x \in BlkIdx :
            (x + 1) * BlockSize > lanePos[l]
            /\ (x + 1) * BlockSize <= lanePos[l] + c
            /\ (x + 1) * BlockSize <= MatchCeiling(laneTarget[l])}}

NewStateMemNeedMap == [l \in Lanes |->
    IF laneState[l] = "Ready" /\ lanePos[l] < laneTarget[l]
    THEN Cardinality({pk \in NewBoundariesInSpan(l, MaxLegalSpan(l)) : ~StateInMem(pk)})
    ELSE 0]

StateMemPressure ==
    /\ ColdMemStates # {}
    /\ \/ (StateMemUsed >= StateMemBudget
           /\ \E l \in RestoringLanes :
                StateOnDisk(<<lanePrefix[l], lanePos[l]>>)
                /\ ~StateInMem(<<lanePrefix[l], lanePos[l]>>))
       \/ (SteppableReady # {} /\ StateMemUsed + SumBy(SteppableReady, NewStateMemNeedMap) > StateMemBudget)

EvictCleanOrUnrefMem ==
    /\ MemPressure
    /\ \E b \in {d \in ColdMemBlocks : OnDisk(d) \/ blocks[d].refCount = 0} :
        LET nextTier == IF blocks[b].refCount = 0 /\ ~OnDisk(b) THEN "none" ELSE "disk"
        IN blocks' = [blocks EXCEPT ![b] =
                        IF nextTier = "none" THEN EmptyBlock
                        ELSE [@ EXCEPT !.tier = "disk"]]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

StartColdWriteback ==
    /\ Bug # "no_spill"
    /\ stepIO = {}
    /\ MemPressure
    /\ {b \in ColdMemBlocks : OnDisk(b) \/ blocks[b].refCount = 0} = {}
    /\ DiskUsed < DiskBudget
    /\ \E b \in {d \in ColdMemBlocks : blocks[d].tier = "memory" /\ blocks[d].sealed} :
        stepIO' = {b}
    /\ UNCHANGED <<blocks, startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, gpuDone, restoreIO>>

EvictCleanStartState ==
    /\ StateMemPressure
    /\ \E pk \in {x \in ColdMemStates : StateOnDisk(x)} :
        startPoints' = [startPoints EXCEPT ![pk].stateTier = "disk"]
    /\ UNCHANGED <<blocks, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

SpillStartState ==
    /\ StateMemPressure
    /\ {x \in ColdMemStates : StateOnDisk(x)} = {}
    /\ StateDiskUsed < StateDiskBudget
    /\ \E pk \in {x \in ColdMemStates : startPoints[x].stateTier = "memory"} :
        startPoints' = [startPoints EXCEPT ![pk].stateTier =
                            IF Bug = "drop_start_state" THEN "none" ELSE "disk"]
    /\ UNCHANGED <<blocks, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

ReclaimColdDisk ==
    /\ \/ /\ DiskUsed + Cardinality(stepIO) >= DiskBudget
          /\ (MemPressure \/ \E b \in ColdDiskBlocks : blocks[b].refCount = 0)
          /\ \/ /\ \E b \in {d \in ColdDiskBlocks : blocks[d].refCount = 0} :
                      LET nextTier == IF InMemory(b) THEN "memory" ELSE "none"
                      IN blocks' = [blocks EXCEPT ![b] =
                                      IF nextTier = "none" THEN EmptyBlock
                                      ELSE [@ EXCEPT !.tier = "memory"]]
                /\ UNCHANGED startPoints
             \/ /\ ~\E b \in ColdDiskBlocks : blocks[b].refCount = 0
                /\ \E pk \in DOMAIN startPoints :
                      /\ startPoints[pk].kept
                      /\ pk \notin RestoringStartNeed
                      /\ ((startPoints[pk].full \cup startPoints[pk].win) \cap stepIO) = {}
                      /\ \E b \in startPoints[pk].full \cup startPoints[pk].win :
                              b \in ColdDiskBlocks \cup ColdMemBlocks
                      /\ LET nsp == [startPoints EXCEPT ![pk] = EmptyStart]
                         IN /\ startPoints' = nsp
                            /\ blocks' = SyncRefCounts(blocks, laneState, laneFull, laneWin, nsp)
       \/ /\ StateDiskUsed >= StateDiskBudget
          /\ StateMemPressure
          /\ \E pk \in ColdDiskStates \cup ColdMemStates :
                /\ ((startPoints[pk].full \cup startPoints[pk].win) \cap stepIO) = {}
                /\ LET nsp == [startPoints EXCEPT ![pk] = EmptyStart]
                   IN /\ startPoints' = nsp
                      /\ blocks' = SyncRefCounts(blocks, laneState, laneFull, laneWin, nsp)
    /\ UNCHANGED <<laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

DropUnreferenced(b) ==
    /\ Present(b)
    /\ blocks[b].refCount = 0
    /\ b \notin stepPins \cup stepIO
    /\ blocks' = [blocks EXCEPT ![b] = EmptyBlock]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

\* ---- Window Sliding & Lane Completion -------------------------------------

SlideWindow(l) ==
    /\ laneState[l] = "Ready"
    /\ LET keep == History(lanePos[l])
           nw   == [i \in BlkIdx |-> IF i \in keep THEN laneWin[l][i] ELSE NoBlock]
       IN /\ nw # laneWin[l]
          /\ laneWin' = [laneWin EXCEPT ![l] = nw]
          /\ blocks'  = SyncRefCounts(blocks, laneState, laneFull, [laneWin EXCEPT ![l] = nw], startPoints)
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

FinishReady(l) ==
    /\ laneState[l] = "Ready"
    /\ lanePos[l] = laneTarget[l]
    /\ LET nst == [laneState EXCEPT ![l] = "Finished"]
           nlf == [laneFull EXCEPT ![l] = EmptyBlkMap]
           nlw == [laneWin EXCEPT ![l] = EmptyBlkMap]
       IN /\ laneState'   = nst
          /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = DefaultPrefix]
          /\ laneTarget'  = [laneTarget EXCEPT ![l] = DefaultTarget]
          /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
          /\ laneFull'    = nlf
          /\ laneWin'     = nlw
          /\ reservation' = [reservation EXCEPT ![l] = 0]
          /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
          /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
          /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
          /\ blocks'      = SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
    /\ UNCHANGED <<startPoints, queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone, restoreIO>>

\* ---- One Step In Flight (BuildStep -> GpuExecute & CompleteStepIO -> FinishStep) ----

SpanSize(l, c) ==
    IF IsDecoding(l)
    THEN c = 1
    ELSE /\ IsPrefilling(l)
         /\ c \in 1..MaxSpan
         /\ lanePos[l] + c <= PromptEnd(laneTarget[l])
         /\ (Bug = "restore_split" => c = MaxLegalSpan(l))

BuildStep(chosen, spans) ==
    /\ ~stepActive
    /\ (Bug = "writeback_clobber" \/ stepIO = {})
    /\ chosen # {}
    /\ chosen \subseteq SteppableReady
    /\ SumBy(chosen, spans) <= (IF Bug = "over_step_budget" THEN StepTokenBudget + 1 ELSE StepTokenBudget)
    /\ \A l \in chosen : SpanSize(l, spans[l])
    /\ \A l \in chosen : (Bug = "unpruned_window" \/ WindowPruned(l))
    /\ \A l \in (Lanes \ chosen) : spans[l] = 0
    /\ LET maxW       == CHOOSE w \in {waitTicks[r] : r \in SteppableReady} :
                             \A u \in {waitTicks[r] : r \in SteppableReady} : w >= u
           NewFullIds == UNION {{<<"priv", l, "full", i>>   : i \in NeedFullForSpan(l, spans[l])} : l \in chosen}
           NewWinIds  == UNION {{<<"priv", l, "window", i>> : i \in NeedWinForSpan(l, spans[l])}  : l \in chosen}
           NewIds     == NewFullIds \cup NewWinIds
           allocCnt   == Cardinality(NewIds)
           NewBnds    == UNION {NewBoundariesInSpan(l, spans[l]) : l \in chosen}
           newStMem   == Cardinality({pk \in NewBnds : ~StateInMem(pk)})
           FitUnchosenDecoders ==
               {d \in (SteppableReady \ chosen) :
                   IsDecoding(d) /\ MemUsed + allocCnt + NeedAllocMap[d] <= MemBudget}
           HasMaxDecoder ==
               \E d \in SteppableReady :
                   IsDecoding(d) /\ waitTicks[d] = maxW /\ MemUsed + NeedAllocMap[d] <= MemBudget
       IN /\ IF Bug = "no_decode_priority"
             THEN /\ \E l \in chosen : waitTicks[l] = maxW
                  /\ FitUnchosenDecoders # {} =>
                         /\ SumBy(chosen, spans) = StepTokenBudget
                         /\ ~\E p \in chosen : IsPrefilling(p) /\ waitTicks[p] < maxW
                         /\ ~\E p \in chosen : IsPrefilling(p) /\ spans[p] > 1
             ELSE \E leader \in chosen :
                      /\ waitTicks[leader] = maxW
                      /\ HasMaxDecoder => IsDecoding(leader)
                      /\ FitUnchosenDecoders # {} =>
                             /\ SumBy(chosen, spans) >= StepTokenBudget
                             /\ \A q \in (chosen \ {leader}) :
                                    IsDecoding(q) \/ waitTicks[q] >= MaxBypass
                             /\ IsPrefilling(leader) => spans[leader] = 1
          /\ IF Bug = "queued_bypass"
             THEN \A r \in (SteppableReady \ chosen) : waitTicks[r] < MaxBypass
             ELSE \A w \in StepBypassed(chosen) : waitTicks[w] < MaxBypass
          /\ \A b \in NewIds : blocks[b].tier = "none"
          /\ MemUsed + allocCnt <= MemBudget
          /\ StateMemUsed + newStMem <= StateMemBudget
          /\ LET nlf == [l \in Lanes |->
                            IF l \in chosen
                            THEN [i \in BlkIdx |->
                                    IF i \in NeedFullForSpan(l, spans[l])
                                    THEN <<"priv", l, "full", i>>
                                    ELSE laneFull[l][i]]
                            ELSE laneFull[l]]
                 nlw == [l \in Lanes |->
                            IF l \in chosen
                            THEN [i \in BlkIdx |->
                                    IF i \in NeedWinForSpan(l, spans[l])
                                    THEN <<"priv", l, "window", i>>
                                    ELSE laneWin[l][i]]
                            ELSE laneWin[l]]
                 nst == [l \in Lanes |-> IF l \in chosen THEN "InStep" ELSE laneState[l]]
                 initPrivData == IF Bug = "dirty_priv_tail" THEN FullData ELSE EmptyData
                 blks1 == [b \in BlockIds |->
                            IF b \in NewIds
                            THEN [tier |-> "memory", sealed |-> FALSE,
                                  idx |-> b[4],
                                  length |-> 0, data |-> initPrivData, refCount |-> 1]
                            ELSE blocks[b]]
                 blks2 == SyncRefCounts(blks1, nst, nlf, nlw, startPoints)
                 PinsFor(l) ==
                    {nlf[l][i] : i \in Full(lanePos[l] + spans[l])} \cup
                    {nlw[l][i] : i \in SpanWindow(lanePos[l], spans[l])}
                 allPins == UNION {PinsFor(l) : l \in chosen}
                 ioCand == {b \in allPins : blks2[b].sealed /\ blks2[b].tier = "memory"}
                 pendIO == IF Bug = "eager_hot_writeback" /\ ioCand # {} /\ DiskUsed < DiskBudget
                           THEN {CHOOSE b \in ioCand : TRUE}
                           ELSE IF Bug = "writeback_clobber"
                           THEN {}
                           ELSE stepIO
             IN /\ NoBlock \notin allPins
                /\ \A b \in allPins : blks2[b].tier \in {"memory", "both"}
                /\ blocks'      = blks2
                /\ laneFull'    = nlf
                /\ laneWin'     = nlw
                /\ laneState'   = nst
                /\ stepActive'  = TRUE
                /\ stepLanes'   = chosen
                /\ stepSpan'    = spans
                /\ stepPins'    = allPins
                /\ stepIO'      = pendIO
                /\ gpuDone'     = FALSE
                /\ waitTicks'   = [q \in Lanes |->
                                    IF q \in chosen THEN 0
                                    ELSE IF q \in StepBypassed(chosen)
                                    THEN IF waitTicks[q] <= MaxBypass THEN waitTicks[q] + 1 ELSE waitTicks[q]
                                    ELSE waitTicks[q]]
                /\ ghostBypass' = [q \in Lanes |->
                                    IF q \in chosen THEN 0
                                    ELSE IF q \in StepBypassed(chosen) THEN ghostBypass[q] + 1
                                    ELSE ghostBypass[q]]
    /\ UNCHANGED <<startPoints, lanePrefix, laneTarget, lanePos, reservation, admitBudget, queueSeq, restoreIO>>

GpuExecute ==
    /\ stepActive
    /\ ~gpuDone
    /\ blocks' = [b \in BlockIds |->
        IF Bug = "mutate_sealed" /\ b \in stepPins /\ Present(b) /\ blocks[b].sealed
        THEN [blocks[b] EXCEPT !.data = [i \in 0..(BlockSize - 1) |-> 2]]
        ELSE IF \E l \in stepLanes :
            /\ b[4] \in Window(lanePos[l], lanePos[l] + stepSpan[l] - 1)
            /\ b \in {laneFull[l][b[4]], laneWin[l][b[4]]}
            /\ Present(b)
            /\ ~blocks[b].sealed
        THEN LET l       == CHOOSE q \in stepLanes :
                                /\ b[4] \in Window(lanePos[q], lanePos[q] + stepSpan[q] - 1)
                                /\ b \in {laneFull[q][b[4]], laneWin[q][b[4]]}
                 bi      == b[4]
                 endPos  == lanePos[l] + stepSpan[l] - 1
                 loInBlk == IF bi = (lanePos[l] \div BlockSize) THEN lanePos[l] % BlockSize ELSE 0
                 hiExcl  == IF bi = (endPos \div BlockSize) THEN (endPos % BlockSize) + 1 ELSE BlockSize
             IN [blocks[b] EXCEPT
                    !.length = hiExcl,
                    !.data   = [i \in 0..(BlockSize - 1) |->
                                  IF i >= loInBlk /\ i < hiExcl THEN 1 ELSE blocks[b].data[i]]]
        ELSE blocks[b]]
    /\ gpuDone' = TRUE
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, stepIO, restoreIO>>

CompleteStepIO ==
    /\ stepIO # {}
    /\ DiskUsed + Cardinality(stepIO) <= DiskBudget
    /\ blocks' = [b \in BlockIds |->
                    IF b \in stepIO /\ blocks[b].tier = "memory"
                    THEN [blocks[b] EXCEPT !.tier =
                            IF Bug = "eager_hot_writeback" \/ LaneReferrers(b) # {}
                            THEN "both"
                            ELSE "disk"]
                    ELSE blocks[b]]
    /\ stepIO' = {}
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks, ghostBypass,
                   queueSeq, stepActive, stepLanes, stepSpan, stepPins, gpuDone, restoreIO>>

FinishStep ==
    /\ stepActive
    /\ gpuDone
    /\ stepIO = {}
    /\ LET Surviving     == {l \in stepLanes : laneState[l] = "InStep"}
           CancellingSet == {l \in stepLanes : laneState[l] = "Cancelling"}
           npos == [l \in Lanes |->
                    IF l \in CancellingSet THEN 0
                    ELSE IF l \in Surviving THEN lanePos[l] + stepSpan[l]
                    ELSE lanePos[l]]
           SealBlkIndices(l) ==
               {bi \in BlkIdx :
                    /\ (bi + 1) * BlockSize > lanePos[l]
                    /\ (bi + 1) * BlockSize <= npos[l]
                    /\ (bi + 1) * BlockSize <= MatchCeiling(laneTarget[l])}
           PublishedTreeIds ==
               UNION { { <<"tree", lanePrefix[l], "full", bi>>   : bi \in SealBlkIndices(l) }
                       \cup
                       { <<"tree", lanePrefix[l], "window", bi>> : bi \in SealBlkIndices(l) } :
                      l \in Surviving }
           RetiredPrivIds ==
               (UNION { { <<"priv", l, "full", bi>>   : bi \in SealBlkIndices(l) }
                        \cup
                        { <<"priv", l, "window", bi>> : bi \in SealBlkIndices(l) } :
                       l \in Surviving })
               \cup
               (UNION { { <<"priv", l, "full", i>>   : i \in BlkIdx }
                        \cup
                        { <<"priv", l, "window", i>> : i \in BlkIdx } :
                       l \in CancellingSet })
           npfx == [l \in Lanes |-> IF l \in CancellingSet THEN DefaultPrefix ELSE lanePrefix[l]]
           ntgt == [l \in Lanes |-> IF l \in CancellingSet THEN DefaultTarget ELSE laneTarget[l]]
           nlf == [l \in Lanes |->
                    IF l \in CancellingSet THEN EmptyBlkMap
                    ELSE IF l \in Surviving
                    THEN [i \in BlkIdx |->
                            IF i \in SealBlkIndices(l)
                            THEN <<"tree", lanePrefix[l], "full", i>>
                            ELSE laneFull[l][i]]
                    ELSE laneFull[l]]
           \* Leave `laneWin[l]` unpruned in FinishStep so `SlideWindow(l)` executes as an
           \* explicit transition before the next `BuildStep` when blocks leave `History(npos[l])`.
           nlw == [l \in Lanes |->
                    IF l \in CancellingSet THEN EmptyBlkMap
                    ELSE IF l \in Surviving
                    THEN [i \in BlkIdx |->
                            IF i \in SealBlkIndices(l)
                            THEN <<"tree", lanePrefix[l], "window", i>>
                            ELSE laneWin[l][i]]
                    ELSE laneWin[l]]
           blksSealed == [b \in BlockIds |->
                            IF b \in RetiredPrivIds THEN EmptyBlock
                            ELSE IF b \in PublishedTreeIds
                            THEN [tier     |-> IF blocks[b].tier \in {"disk", "both"} THEN "both" ELSE "memory",
                                  sealed   |-> TRUE,
                                  idx      |-> b[4],
                                  length   |-> IF Bug = "unsealed_tree" THEN BlockSize - 1 ELSE BlockSize,
                                  data     |-> FullData,
                                  refCount |-> blocks[b].refCount]
                            ELSE blocks[b]]
           nsp == [pk \in Prefixes \X Boundaries |->
                    IF \E l \in Surviving :
                        lanePrefix[l] = pk[1] /\ \E bi \in SealBlkIndices(l) : (bi + 1) * BlockSize = pk[2]
                    THEN [kept      |-> TRUE,
                          stateTier |-> IF startPoints[pk].stateTier \in {"disk", "both"} THEN "both" ELSE "memory",
                          full      |-> {<<"tree", pk[1], "full", i>>   : i \in Full(pk[2])},
                          win       |-> {<<"tree", pk[1], "window", i>> : i \in History(pk[2])}]
                    ELSE startPoints[pk]]
           nst == [l \in Lanes |->
                    IF l \in CancellingSet THEN "Cancelled"
                    ELSE IF l \in Surviving THEN "Ready"
                    ELSE laneState[l]]
           nres == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE reservation[l]]
           nadm == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE admitBudget[l]]
           ngho == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE ghostBypass[l]]
           nwai == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE waitTicks[l]]
           blksFinal == SyncRefCounts(blksSealed, nst, nlf, nlw, nsp)
       IN /\ blocks'      = blksFinal
          /\ startPoints' = nsp
          /\ lanePos'     = npos
          /\ lanePrefix'  = npfx
          /\ laneTarget'  = ntgt
          /\ laneState'   = nst
          /\ laneFull'    = nlf
          /\ laneWin'     = nlw
          /\ reservation' = nres
          /\ admitBudget' = nadm
          /\ waitTicks'   = nwai
          /\ ghostBypass' = ngho
          /\ stepActive'  = FALSE
          /\ stepLanes'   = {}
          /\ stepSpan'    = [l \in Lanes |-> 0]
          /\ stepPins'    = {}
          /\ gpuDone'     = FALSE
    /\ UNCHANGED <<queueSeq, stepIO, restoreIO>>

\* ---- Client Disconnect / Cancellation -------------------------------------

Cancel(l) ==
    /\ Bug # "restore_split"
    /\ laneState[l] \in {"Queued", "Admitted", "Restoring", "Ready", "InStep"}
    /\ IF (laneState[l] = "InStep" /\ Bug # "cancel_in_step")
          \/ (laneState[l] = "Restoring" /\ restoreIO[l] # NoBlock /\ Bug # "cancel_in_restore")
       THEN /\ laneState' = [laneState EXCEPT ![l] = "Cancelling"]
            /\ UNCHANGED <<blocks, lanePrefix, laneTarget, lanePos, laneFull, laneWin,
                           reservation, admitBudget, waitTicks, ghostBypass, queueSeq, restoreIO>>
       ELSE LET nst == [laneState EXCEPT ![l] = "Cancelled"]
                nlf == [laneFull EXCEPT ![l] = EmptyBlkMap]
                nlw == [laneWin EXCEPT ![l] = EmptyBlkMap]
                blks1 == SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
            IN /\ laneState'   = nst
               /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = DefaultPrefix]
               /\ laneTarget'  = [laneTarget EXCEPT ![l] = DefaultTarget]
               /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
               /\ laneFull'    = nlf
               /\ laneWin'     = nlw
               /\ reservation' = [reservation EXCEPT ![l] = 0]
               /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
               /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
               /\ ghostBypass' = [ghostBypass EXCEPT ![l] = 0]
               /\ queueSeq'    = IF laneState[l] = "Queued" THEN RemoveLane(queueSeq, l) ELSE queueSeq
               /\ blocks'      = blks1
               /\ UNCHANGED restoreIO
    /\ UNCHANGED <<startPoints, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- Next-State Relation & Disjoint Fairness ------------------------------

AdvanceLane(l) ==
    \/ Refuse429(l)
    \/ Admit(l)
    \/ BeginRestore(l)
    \/ (\E b \in BlockIds : RestoreBlock(l, b))
    \/ (\E b \in BlockIds : BeginRestoreIO(l, b))
    \/ CompleteRestoreIO(l)
    \/ RestoreStartState(l)
    \/ FinishRestore(l)
    \/ (Bug # "unpruned_window" /\ SlideWindow(l))
    \/ FinishReady(l)

MemReclaim ==
    \/ EvictCleanOrUnrefMem
    \/ StartColdWriteback
    \/ EvictCleanStartState
    \/ SpillStartState
    \/ ReclaimColdDisk

AnyBuildStep ==
    \E chosen \in ((SUBSET Lanes) \ {{}}), spans \in [Lanes -> 0..MaxSpan] : BuildStep(chosen, spans)

StepProgress ==
    \/ GpuExecute
    \/ CompleteStepIO
    \/ FinishStep

FairSchedulerActions ==
    \/ (\E l \in Lanes : AdvanceLane(l))
    \/ MemReclaim
    \/ AnyBuildStep
    \/ StepProgress

UnfairEnvActions ==
    \/ (\E l \in Lanes, p \in Prefixes, r \in RequestLengths : Arrive(l, p, r))
    \/ (\E l \in Lanes : Cancel(l))
    \/ (\E b \in BlockIds : DropUnreferenced(b))

Next == FairSchedulerActions \/ UnfairEnvActions

Fairness ==
    /\ (\A l \in Lanes : WF_vars(AdvanceLane(l)))
    /\ WF_vars(MemReclaim)
    /\ WF_vars(AnyBuildStep)
    /\ WF_vars(StepProgress)

Spec == Init /\ [][Next]_vars /\ Fairness
SpecSafety == Init /\ [][Next]_vars
SymmetryLanesAndPrefixes == Permutations(Lanes) \cup Permutations(Prefixes)

\* ---- Safety Invariants (docs/DESIGN.md "Models") --------------------------

TypeOK ==
    /\ \A b \in BlockIds :
        /\ blocks[b].tier \in Tiers
        /\ blocks[b].sealed \in BOOLEAN
        /\ blocks[b].idx \in BlkIdx
        /\ blocks[b].length \in 0..BlockSize
        /\ blocks[b].data \in [0..(BlockSize - 1) -> 0..2]
        /\ blocks[b].refCount \in Nat
    /\ \A pk \in Prefixes \X Boundaries :
        /\ startPoints[pk].kept \in BOOLEAN
        /\ startPoints[pk].stateTier \in Tiers
        /\ startPoints[pk].full \subseteq TreeIds
        /\ startPoints[pk].win \subseteq TreeIds
    /\ laneState \in [Lanes -> SlotStates]
    /\ lanePrefix \in [Lanes -> Prefixes]
    /\ laneTarget \in [Lanes -> RequestLengths]
    /\ lanePos \in [Lanes -> 0..MaxPos]
    /\ reservation \in [Lanes -> 0..MemBudget]
    /\ admitBudget \in [Lanes -> 0..MemBudget]
    /\ waitTicks \in [Lanes -> 0..(MaxBypass + 1)]
    /\ ghostBypass \in [Lanes -> Nat]
    /\ queueSeq \in Seq(Lanes)
    /\ Len(queueSeq) = Cardinality(QueuedLanes)
    /\ \A i \in 1..Len(queueSeq) : queueSeq[i] \in QueuedLanes
    /\ \A i, j \in 1..Len(queueSeq) : i # j => queueSeq[i] # queueSeq[j]
    /\ stepActive \in BOOLEAN
    /\ stepLanes \subseteq Lanes
    /\ stepPins \subseteq BlockIds
    /\ stepIO \subseteq BlockIds
    /\ gpuDone \in BOOLEAN
    /\ restoreIO \in [Lanes -> BlockIds \cup {NoBlock}]

StepBlocksInMemoryAndOwned ==
    /\ (stepActive =>
        /\ \A l \in stepLanes :
            /\ laneState[l] \in {"InStep", "Cancelling"}
            /\ \A i \in Full(lanePos[l] + stepSpan[l]) :
                laneFull[l][i] # NoBlock /\ laneFull[l][i] \in stepPins
            /\ \A i \in SpanWindow(lanePos[l], stepSpan[l]) :
                laneWin[l][i] # NoBlock /\ laneWin[l][i] \in stepPins
        /\ \A b \in stepPins :
            /\ InMemory(b)
            /\ LaneReferrers(b) # {}
            /\ blocks[b].refCount >= 1)
    /\ \A b \in stepIO :
        /\ InMemory(b)
        /\ Referrers(b) # {}
        /\ blocks[b].refCount >= 1
    /\ \A l \in Lanes :
        restoreIO[l] # NoBlock =>
            /\ laneState[l] \in {"Restoring", "Cancelling"}
            /\ restoreIO[l] \in LaneBlocks(l)
            /\ Present(restoreIO[l])
            /\ blocks[restoreIO[l]].refCount >= 1

SealedValid ==
    \A b \in BlockIds :
        (Present(b) /\ blocks[b].sealed) =>
            /\ b \in TreeIds
            /\ blocks[b].length = BlockSize
            /\ \A i \in 0..(BlockSize - 1) : blocks[b].data[i] > 0

MaxPrefillBoundary ==
    LET maxR == CHOOSE r \in RequestLengths : \A q \in RequestLengths : r >= q
    IN PromptEnd(maxR)

\* Decision 4 (`docs/DESIGN.md`): decode-written blocks (`>= PromptEnd`) are never
\* sealed into shared `TreeIds` and never kept as start points.
NoDecodeBlocksSealed ==
    /\ \A l \in Lanes :
        laneState[l] \in MemHoldingStates =>
            \A i \in BlkIdx :
                i >= (PromptEnd(laneTarget[l]) \div BlockSize) =>
                    /\ (laneFull[l][i] # NoBlock => laneFull[l][i] \in PrivIds /\ ~blocks[laneFull[l][i]].sealed)
                    /\ (laneWin[l][i] # NoBlock  => laneWin[l][i] \in PrivIds  /\ ~blocks[laneWin[l][i]].sealed)
    /\ \A pk \in Prefixes \X Boundaries :
        startPoints[pk].kept => pk[2] <= MaxPrefillBoundary
    /\ \A b \in TreeIds :
        Present(b) => (b[4] + 1) * BlockSize <= MaxPrefillBoundary

SealedNeverWritten ==
    \A b \in BlockIds :
        (Present(b) /\ blocks[b].sealed /\ blocks'[b].tier # "none") =>
            /\ blocks'[b].sealed = TRUE
            /\ blocks'[b].length = blocks[b].length
            /\ blocks'[b].data = blocks[b].data

UnsealedPrivateInvisibleZero ==
    \A b \in BlockIds :
        (Present(b) /\ ~blocks[b].sealed) =>
            /\ b \in PrivIds
            /\ Cardinality(Referrers(b)) = 1
            /\ Cardinality(LaneReferrers(b)) = 1
            /\ StartReferrers(b) = {}
            /\ blocks[b].tier = "memory"
            /\ blocks[b].length <= BlockSize
            /\ ((~stepActive /\ b[4] < (PromptEnd(laneTarget[b[2]]) \div BlockSize)) =>
                    blocks[b].length < BlockSize)
            /\ \A i \in blocks[b].length..(BlockSize - 1) : blocks[b].data[i] = 0

RefCountMatchesReferrers ==
    \A b \in BlockIds :
        /\ blocks[b].refCount = Cardinality(Referrers(b))
        /\ blocks[b].refCount > 0 => Present(b)
        /\ ~Present(b) => blocks[b].refCount = 0

StartPointsKept ==
    \A pk \in Prefixes \X Boundaries :
        startPoints[pk].kept =>
            /\ startPoints[pk].stateTier \in {"memory", "disk", "both"}
            /\ Cardinality(startPoints[pk].full) = Cardinality(Full(pk[2]))
            /\ Cardinality(startPoints[pk].win) = Cardinality(History(pk[2]))
            /\ \A i \in Full(pk[2]) :
                \E b \in startPoints[pk].full :
                    Present(b) /\ blocks[b].sealed /\ BlockKind(b) = "full" /\ blocks[b].idx = i
            /\ \A i \in History(pk[2]) :
                \E b \in startPoints[pk].win :
                    Present(b) /\ blocks[b].sealed /\ BlockKind(b) = "window" /\ blocks[b].idx = i

LaneHistoryCovered ==
    \A l \in Lanes :
        laneState[l] \in {"Restoring", "Ready", "InStep"} =>
            /\ \A i \in Full(lanePos[l]) :
                laneFull[l][i] # NoBlock /\ Present(laneFull[l][i])
            /\ \A i \in History(lanePos[l]) :
                laneWin[l][i] # NoBlock /\ Present(laneWin[l][i])

BudgetsAndReservations ==
    /\ MemUsed <= MemBudget
    /\ DiskUsed <= DiskBudget
    /\ StateMemUsed <= StateMemBudget
    /\ StateDiskUsed <= StateDiskBudget
    /\ ReservedTotal <= MemBudget
    /\ \A l \in Lanes :
        laneState[l] \in {"Restoring", "Ready", "InStep"} =>
            reservation[l] >= CurrentWorkingSet(l)

\* Non-tautological spillability invariant (`U62`): whenever cold blocks sit in memory
\* under `MemPressure` with no writeback in flight, `MemReclaim` MUST be enabled to
\* evict, spill (`StartColdWriteback`), or reclaim disk space for them.
ColdBlocksSpillable ==
    /\ \A b \in BlockIds :
        (InMemory(b) /\ LaneReferrers(b) = {} /\ b \notin stepIO) =>
            /\ blocks[b].sealed = TRUE
            /\ b \notin stepPins
    /\ (MemPressure /\ stepIO = {}) => ENABLED MemReclaim

\* Independent unclamped ghost pass-over counter (`U59`): verifies both the ceiling
\* `ghostBypass[l] <= MaxBypass` and agreement with the scheduler's `waitTicks[l]`.
BypassBoundHolds ==
    \A l \in Lanes :
        laneState[l] \in WaitingStates =>
            /\ ghostBypass[l] <= MaxBypass
            /\ waitTicks[l] = ghostBypass[l]

StepBudgetHolds ==
    stepActive =>
        /\ SumBy(stepLanes, stepSpan) <= StepTokenBudget
        /\ \A l \in stepLanes :
            lanePos[l] >= PromptEnd(laneTarget[l]) => stepSpan[l] = 1

QueueBounded ==
    /\ Cardinality(QueuedLanes) <= MaxQueue
    /\ \A q \in QueuedLanes : WorstCaseBlocks(laneTarget[q]) <= MemBudget

\* ---- Temporal Properties (Action Safety + Liveness) -----------------------

ActionSealedImmutable == [][SealedNeverWritten]_vars

QueuedNeverRefused ==
    \A l \in Lanes : laneState[l] = "Queued" => laneState'[l] # "Refused"

ActionNoQueuedRefusal == [][QueuedNeverRefused]_vars

LazyColdWritebackStep ==
    /\ \A b \in BlockIds :
        (b \notin stepIO /\ b \in stepIO') =>
            /\ MemPressure
            /\ LaneReferrers(b) = {}
            /\ blocks[b].sealed = TRUE
            /\ blocks[b].tier = "memory"
    /\ \A b \in BlockIds :
        (~OnDisk(b) /\ (OnDisk(b))') =>
            /\ (MemPressure \/ b \in stepIO)
            /\ blocks[b].sealed = TRUE
            /\ blocks[b].tier = "memory"
            /\ (b \notin stepIO => LaneReferrers(b) = {})
            /\ blocks'[b].tier = (IF LaneReferrers(b) = {} THEN "disk" ELSE "both")

ActionLazyColdWriteback == [][LazyColdWritebackStep]_vars

\* An in-flight cold writeback (`b \in stepIO`) is never dropped or clobbered by
\* `BuildStep` or any other transition until `CompleteStepIO` lands it `OnDisk` (`U61`).
WritebackProtectedStep ==
    \A b \in stepIO :
        b \in stepIO' \/ (OnDisk(b))'

ActionWritebackProtected == [][WritebackProtectedStep]_vars

DecodePriorityStep ==
    (~stepActive /\ stepActive') =>
        LET maxW == CHOOSE w \in {waitTicks[r] : r \in SteppableReady} :
                        \A u \in {waitTicks[r] : r \in SteppableReady} : w >= u
            stepAlloc == SumBy(stepLanes', [q \in Lanes |->
                            IF q \in stepLanes' THEN SpanAllocCount(q, stepSpan'[q]) ELSE 0])
            FitUnchosenDecoders ==
                {d \in (SteppableReady \ stepLanes') :
                    IsDecoding(d) /\ MemUsed + stepAlloc + NeedAllocMap[d] <= MemBudget}
            HasMaxDecoder ==
                \E d \in SteppableReady :
                    IsDecoding(d) /\ waitTicks[d] = maxW /\ MemUsed + NeedAllocMap[d] <= MemBudget
        IN \E leader \in stepLanes' :
            /\ waitTicks[leader] = maxW
            /\ HasMaxDecoder => IsDecoding(leader)
            /\ FitUnchosenDecoders # {} =>
                   /\ SumBy(stepLanes', stepSpan') = StepTokenBudget
                   /\ \A q \in (stepLanes' \ {leader}) :
                          IsDecoding(q) \/ waitTicks[q] >= MaxBypass
                   /\ IsPrefilling(leader) => stepSpan'[leader] = 1

ActionDecodePriority == [][DecodePriorityStep]_vars

LiveRequestResolves ==
    (laneState[ProbeLane] \in ({"Queued"} \cup AdmittedStates))
        ~> (laneState[ProbeLane] \in {"Refused", "Cancelled", "Finished"})

=============================================================================
