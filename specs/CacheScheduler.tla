---------------------------- MODULE CacheScheduler ----------------------------
\* Coupled model of the block cache and lane scheduler (docs/DESIGN.md,
\* sections "Blocks", "Scheduler", and "Models").
\*
\* Vocabulary matches specs/BlockLayout.tla:
\*   - Kinds == {"full", "window"}: the two cache block kinds;
\*   - BlockSize (B in BlockLayout), WindowSize (W in BlockLayout): positions
\*     per block and sliding-window width in positions;
\*   - Window, History, SpanWindow, Full, SpanBound: exact logical block sets;
\*   - start point, span, pin, sealed block, unsealed tail, memory/disk tiers;
\*   - MaxBypass: the scheduler's bypass bound W covering Ready, Restoring,
\*     and Queued lanes.

EXTENDS Integers, FiniteSets, TLC

CONSTANTS
    Lanes,          \* Set of lane slots, e.g. {a, b}
    Prefixes,       \* Set of token prefixes, e.g. {p1, p2}
    RequestLengths, \* Allowed request target lengths, e.g. {2, 4}
    BlockSize,      \* Positions per block (64 in production, 2 here)
    WindowSize,     \* Sliding-window positions (1024 in production, 2 here)
    MaxPos,          \* Maximum context positions (4 or 8 here)
    MaxSpan,         \* Maximum tokens per prefill span (2 here)
    StepTokenBudget, \* Aggregate token budget across all spans in one step
    MemBudget,       \* Memory budget in blocks
    DiskBudget,      \* Disk budget in blocks
    MaxQueue,        \* Queue limit before 429 refusal
    MaxBypass,       \* Bypass bound W for Ready, Restoring, and Queued lanes
    Bug              \* "none", "cancel_in_step", "restore_split", "boundary_window_dropped", "unpruned_window"

ASSUME
    /\ BlockSize > 0
    /\ WindowSize >= BlockSize
    /\ MaxPos >= BlockSize
    /\ MaxPos % BlockSize = 0
    /\ RequestLengths # {}
    /\ \A r \in RequestLengths : r \in 1..MaxPos /\ r % BlockSize = 0
    /\ MaxSpan \in 1..BlockSize
    /\ StepTokenBudget >= MaxSpan
    /\ MemBudget >= 2
    /\ DiskBudget >= 2
    /\ MaxQueue >= 0
    /\ MaxBypass >= 1
    /\ Bug \in {"none", "cancel_in_step", "restore_split", "boundary_window_dropped", "unpruned_window"}

Kinds == {"full", "window"}
Tiers == {"none", "memory", "disk", "both"}
BlkIdx == 0..((MaxPos - 1) \div BlockSize)
Boundaries == {k \in 1..MaxPos : k % BlockSize = 0}
DefaultPrefix == CHOOSE p \in Prefixes : TRUE
DefaultTarget == CHOOSE r \in RequestLengths : TRUE
ProbeLane == CHOOSE l \in Lanes : TRUE

\* Canonical block identifiers:
\*   - TreeIds: sealed blocks in the shared prefix tree, keyed by (prefix, kind, idx);
\*   - PrivIds: private unsealed tail blocks being written by one lane (lane, kind).
TreeIds == {<<"tree", p, k, i>> : p \in Prefixes, k \in Kinds, i \in BlkIdx}
PrivIds == {<<"priv", l, k>> : l \in Lanes, k \in Kinds}
BlockIds == TreeIds \cup PrivIds
NoBlock == <<"none">>

BlockKind(b) == b[3]

\* Logical block-range helpers (identical to specs/BlockLayout.tla).
Ceil(n, d) == (n + d - 1) \div d
SpanBound(c) == Ceil(WindowSize - 1 + c, BlockSize) + 1
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div BlockSize : p \in lo..hi}
History(n) == Window(IF n > WindowSize THEN n - WindowSize ELSE 0, n - 1)
SpanWindow(n, c) == Window(IF n >= WindowSize THEN n - WindowSize + 1 ELSE 0, n + c - 1)
Full(n) == Window(0, n - 1)

TotalFullBlocks(m) == Cardinality(Full(m))
MaxWinBlocks(m) ==
    LET bound == SpanBound(MaxSpan)
        allW  == Cardinality(Full(m))
    IN IF bound < allW THEN bound ELSE allW
WorstCaseBlocks(m) == TotalFullBlocks(m) + MaxWinBlocks(m)

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
    stateKept |-> FALSE,
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
WaitingStates == {"Queued", "Restoring", "Ready"}

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
    waitTicks,      \* [Lanes -> 0..(MaxBypass + 1)]
    stepActive,     \* BOOLEAN: one step in flight
    stepLanes,      \* Subset of Lanes in the current step
    stepSpan,       \* [Lanes -> 0..MaxSpan]
    stepPins,       \* Subset of BlockIds pinned in memory by the step's spans
    stepIO,         \* Subset of BlockIds with pending disk I/O in this step
    gpuDone         \* BOOLEAN: GPU work of the in-flight step has completed

vars == <<
    blocks, startPoints,
    laneState, lanePrefix, laneTarget, lanePos, laneFull, laneWin,
    reservation, admitBudget, waitTicks,
    stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone
>>

\* ---- Derived Queries & Referrer Accounting --------------------------------

InMemory(b) == blocks[b].tier \in {"memory", "both"}
OnDisk(b)   == blocks[b].tier \in {"disk", "both"}
Present(b)  == blocks[b].tier # "none"

MemUsed  == Cardinality({b \in BlockIds : InMemory(b)})
DiskUsed == Cardinality({b \in BlockIds : OnDisk(b)})

LaneBlockSet(f, w) ==
    ({f[i] : i \in BlkIdx} \cup {w[i] : i \in BlkIdx}) \ {NoBlock}

LaneBlocks(l) == LaneBlockSet(laneFull[l], laneWin[l])

OwningLaneStates == AdmittedStates
LaneReferrersOf(b, st, lf, lw) ==
    {<<"lane", l>> : l \in {q \in Lanes : st[q] \in OwningLaneStates /\ b \in LaneBlockSet(lf[q], lw[q])}}

StartReferrersOf(b, sp) ==
    {<<"start", pk[1], pk[2]>> : pk \in {x \in DOMAIN sp :
        sp[x].kept /\ (b \in sp[x].full \/ (Bug # "boundary_window_dropped" /\ b \in sp[x].win))}}

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
                ELSE [blks[b] EXCEPT !.refCount = refs]]

ReservedTotal       == SumBy({l \in Lanes : laneState[l] \in MemHoldingStates}, reservation)
AdmittedTotal       == SumBy({l \in Lanes : laneState[l] \in AdmittedStates}, admitBudget)
QueuedLanes         == {l \in Lanes : laneState[l] = "Queued"}
ReadyLanes          == {l \in Lanes : laneState[l] = "Ready"}
RestoringLanes      == {l \in Lanes : laneState[l] = "Restoring"}
ActiveRestoring     == {r \in RestoringLanes : \E rb \in LaneBlocks(r) : blocks[rb].tier = "disk"}
SteppableReady      == {l \in ReadyLanes : lanePos[l] < laneTarget[l]}

\* Positions below PromptEnd(t) are prompt prefill (spans up to MaxSpan);
\* positions at or above PromptEnd(t) are token-by-token decode (1-token spans).
PromptEnd(t)    == IF t > BlockSize THEN t - BlockSize ELSE 0
IsPrefilling(l) == laneState[l] = "Ready" /\ lanePos[l] < PromptEnd(laneTarget[l])
IsDecoding(l)   == laneState[l] = "Ready" /\ lanePos[l] >= PromptEnd(laneTarget[l]) /\ lanePos[l] < laneTarget[l]

NeedAllocMap == [l \in Lanes |->
    IF laneState[l] = "Ready" /\ lanePos[l] < laneTarget[l]
    THEN (IF laneFull[l][lanePos[l] \div BlockSize] = NoBlock THEN 1 ELSE 0) +
         (IF laneWin[l][lanePos[l] \div BlockSize] = NoBlock THEN 1 ELSE 0)
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
    /\ stepActive  = FALSE
    /\ stepLanes   = {}
    /\ stepSpan    = [l \in Lanes |-> 0]
    /\ stepPins    = {}
    /\ stepIO      = {}
    /\ gpuDone     = FALSE

\* ---- Request Arrival, Queueing, Admission & 429 Refusal -------------------

Arrive(l, p, reqLen) ==
    /\ laneState[l] \in {"Idle", "Refused", "Cancelled", "Finished"}
    /\ p \in Prefixes
    /\ reqLen \in RequestLengths
    /\ laneState'   = [laneState EXCEPT ![l] = "Queued"]
    /\ lanePrefix'  = [lanePrefix EXCEPT ![l] = p]
    /\ laneTarget'  = [laneTarget EXCEPT ![l] = reqLen]
    /\ lanePos'     = [lanePos EXCEPT ![l] = 0]
    /\ laneFull'    = [laneFull EXCEPT ![l] = EmptyBlkMap]
    /\ laneWin'     = [laneWin EXCEPT ![l] = EmptyBlkMap]
    /\ reservation' = [reservation EXCEPT ![l] = 0]
    /\ admitBudget' = [admitBudget EXCEPT ![l] = 0]
    /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
    /\ UNCHANGED <<blocks, startPoints, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

DiskStartCount(p, maxP) ==
    LET k == BestBoundary(p, maxP)
    IN IF k = 0 THEN 0
       ELSE Cardinality({b \in startPoints[<<p, k>>].full \cup startPoints[<<p, k>>].win : blocks[b].tier = "disk"})

CanFitAdmit(l) ==
    LET wc       == WorstCaseBlocks(laneTarget[l])
        splitBug == Bug = "restore_split" /\ DiskStartCount(lanePrefix[l], laneTarget[l]) >= 2
    IN IF splitBug
       THEN AdmittedTotal + 1 <= MemBudget
       ELSE AdmittedTotal + wc <= MemBudget

CanAdmitNow(l) ==
    /\ CanFitAdmit(l)
    \* Longest-waiting fitting Queued lane first; once any Queued lane hits
    \* MaxBypass, younger arrivals are locked out until the head cohort drains.
    /\ \A q \in (QueuedLanes \ {l}) : CanFitAdmit(q) => waitTicks[l] >= waitTicks[q]
    /\ \A q \in (QueuedLanes \ {l}) : waitTicks[q] >= MaxBypass => waitTicks[l] >= waitTicks[q]

Refuse429(l) ==
    /\ laneState[l] = "Queued"
    /\ ~CanAdmitNow(l)
    /\ \/ WorstCaseBlocks(laneTarget[l]) > MemBudget
       \/ /\ Cardinality(QueuedLanes) > MaxQueue
          /\ \A q \in QueuedLanes : waitTicks[l] <= waitTicks[q]
    /\ laneState'  = [laneState EXCEPT ![l] = "Refused"]
    /\ lanePrefix' = [lanePrefix EXCEPT ![l] = DefaultPrefix]
    /\ laneTarget' = [laneTarget EXCEPT ![l] = DefaultTarget]
    /\ lanePos'    = [lanePos EXCEPT ![l] = 0]
    /\ waitTicks'  = [waitTicks EXCEPT ![l] = 0]
    /\ UNCHANGED <<blocks, startPoints, laneFull, laneWin, reservation, admitBudget,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

Admit(l) ==
    /\ laneState[l] = "Queued"
    /\ CanAdmitNow(l)
    /\ LET wc       == WorstCaseBlocks(laneTarget[l])
           splitBug == Bug = "restore_split" /\ DiskStartCount(lanePrefix[l], laneTarget[l]) >= 2
           bypassed == {q \in (QueuedLanes \ {l}) :
                            waitTicks[l] < waitTicks[q] \/ waitTicks[q] < MaxBypass}
       IN /\ laneState'   = [laneState EXCEPT ![l] = "Admitted"]
          /\ admitBudget' = [admitBudget EXCEPT ![l] = IF splitBug THEN 1 ELSE wc]
          /\ waitTicks'   = [q \in Lanes |->
                                IF q = l THEN 0
                                ELSE IF q \in bypassed THEN waitTicks[q] + 1
                                ELSE waitTicks[q]]
    /\ UNCHANGED <<blocks, startPoints, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- Prefix Matching & Atomic Working-Set Reservation (Restoring) ---------

BeginRestore(l) ==
    /\ laneState[l] = "Admitted"
    /\ LET p        == lanePrefix[l]
           wc       == WorstCaseBlocks(laneTarget[l])
           k        == BestBoundary(p, laneTarget[l])
           sp       == IF k = 0 THEN EmptyStart ELSE startPoints[<<p, k>>]
           diskCnt  == Cardinality({b \in sp.full \cup sp.win : blocks[b].tier = "disk"})
           splitBug == Bug = "restore_split" /\ diskCnt >= 2
           nf       == [i \in BlkIdx |->
                         IF \E b \in sp.full : b[4] = i
                         THEN CHOOSE b \in sp.full : b[4] = i
                         ELSE NoBlock]
           nw       == [i \in BlkIdx |->
                         IF \E b \in sp.win : b[4] = i
                         THEN CHOOSE b \in sp.win : b[4] = i
                         ELSE NoBlock]
           res      == IF splitBug
                       THEN Cardinality({b \in sp.full \cup sp.win : InMemory(b)})
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
          /\ waitTicks'   = [waitTicks EXCEPT ![l] = 0]
          /\ blocks'      = SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
    /\ UNCHANGED <<startPoints, lanePrefix, laneTarget, admitBudget,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

RestoreBlock(l, b) ==
    /\ laneState[l] = "Restoring"
    /\ b \in LaneBlocks(l)
    /\ blocks[b].tier = "disk"
    /\ MemUsed < MemBudget
    \* Longest-waiting restoring lane first; honest unclamped bypass count.
    /\ \A r \in ActiveRestoring : waitTicks[l] >= waitTicks[r]
    /\ \A r \in (ActiveRestoring \ {l}) : waitTicks[r] < MaxBypass
    /\ blocks' = [blocks EXCEPT ![b].tier = "both"]
    /\ reservation' = [reservation EXCEPT ![l] =
                        IF Bug = "restore_split" /\ @ < WorstCaseBlocks(laneTarget[l])
                        THEN @ + 1
                        ELSE @]
    /\ waitTicks' = [q \in Lanes |->
                        IF q = l THEN 0
                        ELSE IF q \in (ActiveRestoring \ {l})
                        THEN waitTicks[q] + 1
                        ELSE waitTicks[q]]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, admitBudget,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

FinishRestore(l) ==
    /\ laneState[l] = "Restoring"
    /\ \A b \in LaneBlocks(l) : InMemory(b)
    /\ laneState' = [laneState EXCEPT ![l] = "Ready"]
    /\ waitTicks' = [waitTicks EXCEPT ![l] = 0]
    /\ UNCHANGED <<blocks, startPoints, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- Lazy Spill & Tier Eviction Under Memory / Disk Pressure --------------

MemPressure ==
    /\ ColdMemBlocks # {}
    /\ \/ (MemUsed >= MemBudget /\ \E l \in RestoringLanes : \E b \in LaneBlocks(l) : ~InMemory(b))
       \/ (SteppableReady # {} /\ MemUsed + SumBy(SteppableReady, NeedAllocMap) > MemBudget)

EvictCleanOrUnrefMem ==
    /\ MemPressure
    /\ \E b \in {d \in ColdMemBlocks : OnDisk(d) \/ blocks[d].refCount = 0} :
        LET nextTier == IF blocks[b].refCount = 0 /\ ~OnDisk(b) THEN "none" ELSE "disk"
        IN blocks' = [blocks EXCEPT ![b] =
                        IF nextTier = "none" THEN EmptyBlock
                        ELSE [@ EXCEPT !.tier = "disk"]]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

SpillColdMemToDisk ==
    /\ MemPressure
    /\ {b \in ColdMemBlocks : OnDisk(b) \/ blocks[b].refCount = 0} = {}
    /\ DiskUsed + Cardinality(stepIO) < DiskBudget
    /\ \E b \in {d \in ColdMemBlocks : blocks[d].tier = "memory" /\ blocks[d].sealed} :
        blocks' = [blocks EXCEPT ![b].tier = "disk"]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

ReclaimColdDisk ==
    /\ DiskUsed + Cardinality(stepIO) >= DiskBudget
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
                /\ \E b \in startPoints[pk].full \cup startPoints[pk].win :
                        b \in ColdDiskBlocks \cup ColdMemBlocks
                /\ LET nsp == [startPoints EXCEPT ![pk] = EmptyStart]
                   IN /\ startPoints' = nsp
                      /\ blocks' = SyncRefCounts(blocks, laneState, laneFull, laneWin, nsp)
    /\ UNCHANGED <<laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

DropUnreferenced(b) ==
    /\ Present(b)
    /\ blocks[b].refCount = 0
    /\ b \notin stepPins \cup stepIO
    /\ blocks' = [blocks EXCEPT ![b] = EmptyBlock]
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- Window Sliding & Lane Completion -------------------------------------

SlideWindow(l) ==
    /\ laneState[l] = "Ready"
    /\ LET keep == History(lanePos[l])
           nw   == [i \in BlkIdx |-> IF i \in keep THEN laneWin[l][i] ELSE NoBlock]
       IN /\ nw # laneWin[l]
          /\ laneWin' = [laneWin EXCEPT ![l] = nw]
          /\ blocks'  = SyncRefCounts(blocks, laneState, laneFull, [laneWin EXCEPT ![l] = nw], startPoints)
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

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
          /\ blocks'      = SyncRefCounts(blocks, nst, nlf, nlw, startPoints)
    /\ UNCHANGED <<startPoints, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- One Step In Flight (BuildStep -> GpuExecute & CompleteStepIO -> FinishStep) ----

SpanSize(l, c) ==
    IF IsDecoding(l)
    THEN c = 1
    ELSE /\ IsPrefilling(l)
         /\ c \in 1..MaxSpan
         /\ lanePos[l] + c <= PromptEnd(laneTarget[l])
         /\ c <= BlockSize - (lanePos[l] % BlockSize)

BuildStep(chosen, spans) ==
    /\ ~stepActive
    /\ chosen # {}
    /\ chosen \subseteq SteppableReady
    /\ SumBy(chosen, spans) <= StepTokenBudget
    /\ \A l \in chosen : SpanSize(l, spans[l])
    /\ \A l \in (Lanes \ chosen) : spans[l] = 0
    /\ LET maxW       == CHOOSE w \in {waitTicks[r] : r \in SteppableReady} :
                             \A u \in {waitTicks[r] : r \in SteppableReady} : w >= u
           NeedFullAlloc(l) == laneFull[l][lanePos[l] \div BlockSize] = NoBlock
           NeedWinAlloc(l)  == laneWin[l][lanePos[l] \div BlockSize] = NoBlock
           NewFullIds == {<<"priv", l, "full">>   : l \in {q \in chosen : NeedFullAlloc(q)}}
           NewWinIds  == {<<"priv", l, "window">> : l \in {q \in chosen : NeedWinAlloc(q)}}
           NewIds     == NewFullIds \cup NewWinIds
           allocCnt   == Cardinality(NewIds)
           \* Ready decoders excluded from chosen whose private tail blocks fit in memory.
           FitUnchosenDecoders ==
               {d \in (SteppableReady \ chosen) :
                   IsDecoding(d) /\ MemUsed + allocCnt + NeedAllocMap[d] <= MemBudget}
       IN /\ \E l \in chosen : waitTicks[l] = maxW
          \* Longest-waiting Ready first, then other Ready decoders, then prefill spans.
          /\ FitUnchosenDecoders # {} =>
                 /\ SumBy(chosen, spans) = StepTokenBudget
                 /\ ~\E p \in chosen : IsPrefilling(p) /\ waitTicks[p] < maxW
                 /\ ~\E p \in chosen : IsPrefilling(p) /\ spans[p] > 1
          /\ \A r \in (SteppableReady \ chosen) : waitTicks[r] < MaxBypass
          /\ \A b \in NewIds : blocks[b].tier = "none"
          /\ MemUsed + allocCnt <= MemBudget
          /\ LET nlf == [l \in Lanes |->
                            IF l \in chosen /\ NeedFullAlloc(l)
                            THEN [laneFull[l] EXCEPT ![lanePos[l] \div BlockSize] = <<"priv", l, "full">>]
                            ELSE laneFull[l]]
                 nlw == [l \in Lanes |->
                            IF l \in chosen /\ NeedWinAlloc(l)
                            THEN [laneWin[l] EXCEPT ![lanePos[l] \div BlockSize] = <<"priv", l, "window">>]
                            ELSE laneWin[l]]
                 nst == [l \in Lanes |-> IF l \in chosen THEN "InStep" ELSE laneState[l]]
                 blks1 == [b \in BlockIds |->
                            IF b \in NewIds
                            THEN LET owner == b[2]
                                 IN [tier |-> "memory", sealed |-> FALSE,
                                     idx |-> lanePos[owner] \div BlockSize,
                                     length |-> 0, data |-> EmptyData, refCount |-> 1]
                            ELSE blocks[b]]
                 blks2 == SyncRefCounts(blks1, nst, nlf, nlw, startPoints)
                 PinsFor(l) ==
                    {nlf[l][i] : i \in Full(lanePos[l] + spans[l])} \cup
                    {nlw[l][i] : i \in SpanWindow(lanePos[l], spans[l])}
                 allPins == UNION {PinsFor(l) : l \in chosen}
                 ioCand == {b \in allPins : blks2[b].sealed /\ blks2[b].tier = "memory"}
                 pendIO == IF ioCand # {} /\ DiskUsed < DiskBudget
                           THEN {CHOOSE b \in ioCand : TRUE}
                           ELSE {}
             IN /\ NoBlock \notin allPins
                /\ \A b \in allPins : blks2[b].tier \in {"memory", "both"}
                /\ blocks'     = blks2
                /\ laneFull'   = nlf
                /\ laneWin'    = nlw
                /\ laneState'  = nst
                /\ stepActive' = TRUE
                /\ stepLanes'  = chosen
                /\ stepSpan'   = spans
                /\ stepPins'   = allPins
                /\ stepIO'     = pendIO
                /\ gpuDone'    = FALSE
                /\ waitTicks'  = [q \in Lanes |->
                                    IF q \in chosen THEN 0
                                    ELSE IF q \in (SteppableReady \ chosen) THEN waitTicks[q] + 1
                                    ELSE waitTicks[q]]
    /\ UNCHANGED <<startPoints, lanePrefix, laneTarget, lanePos, reservation, admitBudget>>

GpuExecute ==
    /\ stepActive
    /\ ~gpuDone
    /\ blocks' = [b \in BlockIds |->
        IF \E l \in stepLanes :
            /\ b \in {laneFull[l][lanePos[l] \div BlockSize], laneWin[l][lanePos[l] \div BlockSize]}
            /\ Present(b)
            /\ ~blocks[b].sealed
        THEN LET l   == CHOOSE q \in stepLanes :
                            b \in {laneFull[q][lanePos[q] \div BlockSize],
                                   laneWin[q][lanePos[q] \div BlockSize]}
                 off == lanePos[l] % BlockSize
                 cnt == stepSpan[l]
             IN [blocks[b] EXCEPT
                    !.length = off + cnt,
                    !.data   = [i \in 0..(BlockSize - 1) |->
                                  IF i >= off /\ i < off + cnt THEN 1 ELSE blocks[b].data[i]]]
        ELSE blocks[b]]
    /\ gpuDone' = TRUE
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, stepIO>>

CompleteStepIO ==
    /\ stepActive
    /\ stepIO # {}
    /\ DiskUsed + Cardinality(stepIO) <= DiskBudget
    /\ blocks' = [b \in BlockIds |->
                    IF b \in stepIO /\ blocks[b].tier = "memory"
                    THEN [blocks[b] EXCEPT !.tier = "both"]
                    ELSE blocks[b]]
    /\ stepIO' = {}
    /\ UNCHANGED <<startPoints, laneState, lanePrefix, laneTarget, lanePos,
                   laneFull, laneWin, reservation, admitBudget, waitTicks,
                   stepActive, stepLanes, stepSpan, stepPins, gpuDone>>

FinishStep ==
    /\ stepActive
    /\ gpuDone
    /\ stepIO = {}
    /\ LET Surviving     == {l \in stepLanes : laneState[l] = "InStep"}
           CancellingSet == {l \in stepLanes : laneState[l] = "Cancelling"}
           BoundaryLanes == {l \in Surviving : (lanePos[l] + stepSpan[l]) % BlockSize = 0}
           PublishedTreeIds ==
               UNION {{<<"tree", lanePrefix[l], "full",   ((lanePos[l] + stepSpan[l]) \div BlockSize) - 1>>,
                       <<"tree", lanePrefix[l], "window", ((lanePos[l] + stepSpan[l]) \div BlockSize) - 1>>} :
                      l \in BoundaryLanes}
           RetiredPrivIds ==
               UNION {{<<"priv", l, "full">>, <<"priv", l, "window">>} : l \in BoundaryLanes \cup CancellingSet}
           npos == [l \in Lanes |->
                    IF l \in CancellingSet THEN 0
                    ELSE IF l \in Surviving THEN lanePos[l] + stepSpan[l]
                    ELSE lanePos[l]]
           npfx == [l \in Lanes |-> IF l \in CancellingSet THEN DefaultPrefix ELSE lanePrefix[l]]
           ntgt == [l \in Lanes |-> IF l \in CancellingSet THEN DefaultTarget ELSE laneTarget[l]]
           nlf == [l \in Lanes |->
                    IF l \in CancellingSet THEN EmptyBlkMap
                    ELSE IF l \in BoundaryLanes
                    THEN LET bi == (npos[l] \div BlockSize) - 1
                         IN [laneFull[l] EXCEPT ![bi] = <<"tree", lanePrefix[l], "full", bi>>]
                    ELSE laneFull[l]]
           nlw == [l \in Lanes |->
                    IF l \in CancellingSet THEN EmptyBlkMap
                    ELSE IF l \in Surviving
                    THEN LET rawW == IF l \in BoundaryLanes
                                     THEN LET bi == (npos[l] \div BlockSize) - 1
                                          IN [laneWin[l] EXCEPT ![bi] = <<"tree", lanePrefix[l], "window", bi>>]
                                     ELSE laneWin[l]
                         IN IF Bug = "unpruned_window"
                            THEN rawW
                            ELSE [i \in BlkIdx |-> IF i \in History(npos[l]) THEN rawW[i] ELSE NoBlock]
                    ELSE laneWin[l]]
           blksSealed == [b \in BlockIds |->
                            IF b \in RetiredPrivIds THEN EmptyBlock
                            ELSE IF b \in PublishedTreeIds
                            THEN [tier     |-> IF blocks[b].tier \in {"disk", "both"} THEN "both" ELSE "memory",
                                  sealed   |-> TRUE,
                                  idx      |-> b[4],
                                  length   |-> BlockSize,
                                  data     |-> FullData,
                                  refCount |-> blocks[b].refCount]
                            ELSE blocks[b]]
           nsp == [pk \in Prefixes \X Boundaries |->
                    IF \E l \in BoundaryLanes : lanePrefix[l] = pk[1] /\ npos[l] = pk[2]
                    THEN [kept      |-> TRUE,
                          stateKept |-> TRUE,
                          full      |-> {<<"tree", pk[1], "full", i>>   : i \in Full(pk[2])},
                          win       |-> {<<"tree", pk[1], "window", i>> : i \in History(pk[2])}]
                    ELSE startPoints[pk]]
           nst == [l \in Lanes |->
                    IF l \in CancellingSet THEN "Cancelled"
                    ELSE IF l \in Surviving THEN "Ready"
                    ELSE laneState[l]]
           nres == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE reservation[l]]
           nadm == [l \in Lanes |-> IF l \in CancellingSet THEN 0 ELSE admitBudget[l]]
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
          /\ stepActive'  = FALSE
          /\ stepLanes'   = {}
          /\ stepSpan'    = [l \in Lanes |-> 0]
          /\ stepPins'    = {}
          /\ gpuDone'     = FALSE
    /\ UNCHANGED <<waitTicks, stepIO>>

\* ---- Client Disconnect / Cancellation -------------------------------------

Cancel(l) ==
    /\ laneState[l] \in {"Queued", "Admitted", "Restoring", "Ready", "InStep"}
    /\ IF laneState[l] = "InStep" /\ Bug # "cancel_in_step"
       THEN /\ laneState' = [laneState EXCEPT ![l] = "Cancelling"]
            /\ UNCHANGED <<blocks, lanePrefix, laneTarget, lanePos, laneFull, laneWin,
                           reservation, admitBudget, waitTicks>>
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
               /\ blocks'      = blks1
    /\ UNCHANGED <<startPoints, stepActive, stepLanes, stepSpan, stepPins, stepIO, gpuDone>>

\* ---- Next-State Relation & Disjoint Fairness ------------------------------

AdvanceLane(l) ==
    \/ Refuse429(l)
    \/ Admit(l)
    \/ BeginRestore(l)
    \/ (\E b \in BlockIds : RestoreBlock(l, b))
    \/ FinishRestore(l)
    \/ FinishReady(l)

MemReclaim ==
    \/ EvictCleanOrUnrefMem
    \/ SpillColdMemToDisk
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
    \/ (\E l \in Lanes : SlideWindow(l) \/ Cancel(l))
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
        /\ blocks[b].data \in [0..(BlockSize - 1) -> {0, 1}]
        /\ blocks[b].refCount \in Nat
    /\ \A pk \in Prefixes \X Boundaries :
        /\ startPoints[pk].kept \in BOOLEAN
        /\ startPoints[pk].stateKept \in BOOLEAN
        /\ startPoints[pk].full \subseteq TreeIds
        /\ startPoints[pk].win \subseteq TreeIds
    /\ laneState \in [Lanes -> SlotStates]
    /\ lanePrefix \in [Lanes -> Prefixes]
    /\ laneTarget \in [Lanes -> RequestLengths]
    /\ lanePos \in [Lanes -> 0..MaxPos]
    /\ reservation \in [Lanes -> 0..MemBudget]
    /\ admitBudget \in [Lanes -> 0..MemBudget]
    /\ waitTicks \in [Lanes -> 0..(MaxBypass + 1)]
    /\ stepActive \in BOOLEAN
    /\ stepLanes \subseteq Lanes
    /\ stepPins \subseteq BlockIds
    /\ stepIO \subseteq BlockIds
    /\ gpuDone \in BOOLEAN

StepBlocksInMemoryAndOwned ==
    stepActive =>
        /\ \A l \in stepLanes :
            /\ laneState[l] \in {"InStep", "Cancelling"}
            /\ \A i \in Full(lanePos[l] + stepSpan[l]) :
                laneFull[l][i] # NoBlock /\ laneFull[l][i] \in stepPins
            /\ \A i \in SpanWindow(lanePos[l], stepSpan[l]) :
                laneWin[l][i] # NoBlock /\ laneWin[l][i] \in stepPins
        /\ \A b \in stepPins \cup stepIO :
            /\ InMemory(b)
            /\ LaneReferrers(b) # {}
            /\ blocks[b].refCount >= 1

SealedValid ==
    \A b \in BlockIds :
        (Present(b) /\ blocks[b].sealed) =>
            /\ b \in TreeIds
            /\ blocks[b].length = BlockSize
            /\ blocks[b].data = FullData

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
            /\ (~stepActive => blocks[b].length < BlockSize)
            /\ \A i \in blocks[b].length..(BlockSize - 1) : blocks[b].data[i] = 0

RefCountMatchesReferrers ==
    \A b \in BlockIds :
        /\ blocks[b].refCount = Cardinality(Referrers(b))
        /\ blocks[b].refCount > 0 => Present(b)
        /\ ~Present(b) => blocks[b].refCount = 0

StartPointsKept ==
    \A pk \in Prefixes \X Boundaries :
        startPoints[pk].kept =>
            /\ startPoints[pk].stateKept = TRUE
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
    /\ ReservedTotal <= MemBudget
    /\ \A l \in Lanes :
        laneState[l] \in {"Restoring", "Ready", "InStep"} =>
            reservation[l] >= CurrentWorkingSet(l)

ColdBlocksSpillable ==
    \A b \in BlockIds :
        (InMemory(b) /\ LaneReferrers(b) = {} /\ b \notin stepIO) =>
            /\ blocks[b].sealed = TRUE
            /\ b \notin stepPins
            /\ \/ OnDisk(b)
               \/ blocks[b].refCount = 0
               \/ DiskUsed + Cardinality(stepIO) < DiskBudget
               \/ ColdDiskBlocks # {}
               \/ StartReferrers(b) # {}

BypassBoundHolds ==
    \A l \in Lanes :
        laneState[l] \in WaitingStates => waitTicks[l] <= MaxBypass

StepBudgetHolds ==
    stepActive =>
        /\ SumBy(stepLanes, stepSpan) <= StepTokenBudget
        /\ \A l \in stepLanes :
            lanePos[l] >= PromptEnd(laneTarget[l]) => stepSpan[l] = 1

\* ---- Temporal Properties (Action Safety + Liveness) -----------------------
\* Every `l \in Lanes` is symmetric in `Init`, `Next`, and `Fairness`. Moreover,
\* the only exits from `"Queued"` are `{"Admitted", "Refused", "Cancelled"}`, and
\* no transition from `AdmittedStates` can reach `"Refused"` without first hitting
\* `{"Finished", "Cancelled"}`. Hence `LiveRequestResolves` on `ProbeLane` holds
\* iff BOTH (a) every queued request is admitted, refused, or cancelled AND
\* (b) every admitted/restoring lane finishes or is cancelled, while keeping a
\* minimal 2-state Buchi tableau in TLC.

ActionSealedImmutable == [][SealedNeverWritten]_vars

LiveRequestResolves ==
    (laneState[ProbeLane] \in ({"Queued"} \cup AdmittedStates))
        ~> (laneState[ProbeLane] \in {"Refused", "Cancelled", "Finished"})

=============================================================================
