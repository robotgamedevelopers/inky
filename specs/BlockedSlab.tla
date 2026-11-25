---------------------------- MODULE BlockedSlab ----------------------------
\* Layout and state model for `BlockedSlab` slot allocation, LIFO `Drop` reuse,
\* `(slot * 2 + swa) * bt_stride_u64` strided block-table growth/re-layout, and
\* `BlockTable` `[dirty_lo..=dirty_hi]` incremental GPU upload
\* (`src/engine.rs:560-683`, `src/engine.rs:1877-1934`, `src/blocks.rs:844-933`).

EXTENDS Integers, FiniteSets, Sequences

CONSTANTS
    Lanes,          \* Set of concurrent lane handles (cardinality 1..4)
    InitBlocks,     \* Initial `default_blocks` passed to `BlockedSlab::new`
    MaxBlocks,      \* Maximum `need_blocks` a lane may request (tests stride growth)
    Align,          \* `BT_STRIDE_ALIGN_U64` (32 in production, 1 or 2 here)
    MaxRecycles,    \* Bounded `Drop` -> `free_slots.pop()` reuse cycles per run
    Bug             \* "none" | "flat_grow" | "no_zero_reuse" | "narrow_dirty"

ASSUME
    /\ Lanes # {}
    /\ Cardinality(Lanes) <= 4
    /\ InitBlocks >= 1
    /\ MaxBlocks >= InitBlocks
    /\ Align >= 1
    /\ MaxRecycles >= 1
    /\ Bug \in {"none", "flat_grow", "no_zero_reuse", "narrow_dirty"}

\* Exact `BlockedSlab::next_cap` (`src/engine.rs:582-590`).
NextCap(need) ==
    IF need <= 1 THEN 1
    ELSE IF need <= 2 THEN 2
    ELSE IF need <= 4 THEN 4
    ELSE 8

AlignUp(n) ==
    LET m == IF n < 1 THEN 1 ELSE n
    IN ((m + Align - 1) \div Align) * Align

MaxCap == NextCap(Cardinality(Lanes))
MaxStride == AlignUp(MaxBlocks)
MaxBtCells == MaxCap * 2 * MaxStride
NoSlot == -1
CleanRange == <<-1, -1>>
ProbeLane == CHOOSE l \in Lanes : TRUE
PeerLanes == Lanes \ {ProbeLane}

PeerOrder ==
    LET F[S \in SUBSET PeerLanes] ==
        IF S = {} THEN << >>
        ELSE LET x == CHOOSE v \in S : TRUE
             IN <<x>> \o F[S \ {x}]
    IN F[PeerLanes]

PeerRank(l) == CHOOSE i \in 1..Len(PeerOrder) : PeerOrder[i] = l

BtBase(slotIdx, swa, stride) == (slotIdx * 2 + swa) * stride

\* Unique non-zero packed GPU word per `(slotIdx, swa, block)` so any cross-slot,
\* cross-plane (`swa = 0` vs `swa = 1`), cross-block, or flat-re-layout aliasing is
\* immediately observable in `btBuf`.
EntryWord(slotIdx, swa, b) == (slotIdx + 1) * 32 + swa * 8 + b + 1

VARIABLES
    cap,            \* Current `BlockedSlab.cap` (0, 1, 2, 4)
    used,           \* High-water `BlockedSlab.used`
    btStride,       \* Current `BlockedSlab.bt_stride_u64`
    freeSlots,      \* LIFO stack `Seq(0..(MaxCap - 1))` pushed on `Drop`
    recycles,       \* Number of `Drop` events executed so far (`0..MaxRecycles`)
    csBuf,          \* [0..(MaxCap - 1) -> {0, 1}] (0 = zero-filled, 1 = dirty conv state)
    btBuf,          \* [0..(MaxBtCells - 1) -> Nat]
    laneActive,     \* [Lanes -> BOOLEAN]
    laneSlot,       \* [Lanes -> {NoSlot} \cup (0..(MaxCap - 1))]
    laneBlocks,     \* [Lanes -> 0..MaxBlocks]
    cpuTable,       \* [Lanes -> [0..1 -> [0..(MaxBlocks - 1) -> Nat]]]
    dirty,          \* [Lanes -> [0..1 -> {CleanRange} \cup ((0..MaxBlocks) \X (0..MaxBlocks))]]
    fullSync,       \* [Lanes -> [0..1 -> BOOLEAN]]
    freshAlloc      \* [Lanes -> BOOLEAN]: TRUE immediately after `alloc_slot` before any step/sync

vars == <<
    cap, used, btStride, freeSlots, recycles, csBuf, btBuf,
    laneActive, laneSlot, laneBlocks, cpuTable, dirty, fullSync, freshAlloc
>>

Init ==
    /\ cap = 0
    /\ used = 0
    /\ btStride = AlignUp(InitBlocks)
    /\ freeSlots = << >>
    /\ recycles = 0
    /\ csBuf = [s \in 0..(MaxCap - 1) |-> 0]
    /\ btBuf = [i \in 0..(MaxBtCells - 1) |-> 0]
    /\ laneActive = [l \in Lanes |-> FALSE]
    /\ laneSlot = [l \in Lanes |-> NoSlot]
    /\ laneBlocks = [l \in Lanes |-> 0]
    /\ cpuTable = [l \in Lanes |-> [swa \in 0..1 |-> [b \in 0..(MaxBlocks - 1) |-> 0]]]
    /\ dirty = [l \in Lanes |-> [swa \in 0..1 |-> CleanRange]]
    /\ fullSync = [l \in Lanes |-> [swa \in 0..1 |-> FALSE]]
    /\ freshAlloc = [l \in Lanes |-> FALSE]

AllowedNeedBlocks(l) ==
    IF l = ProbeLane \/ PeerRank(l) = Len(PeerOrder)
    THEN {InitBlocks, MaxBlocks}
    ELSE {InitBlocks}

\* `Lane::new_blocked` -> `BlockedSlab::alloc_slot(gpu, need_blocks)`
\* Idle peer handles are anonymous, so canonicalize idle peer allocation order while
\* leaving `ProbeLane` free to allocate at any slot index or from `freeSlots`.
AllocLane(l, needBlocks) ==
    /\ ~laneActive[l]
    /\ \A p2 \in PeerLanes : laneActive[p2] => ~freshAlloc[p2]
    /\ (l \in PeerLanes =>
            /\ dirty[ProbeLane][1] = CleanRange
            /\ ~\E p2 \in PeerLanes : ~laneActive[p2] /\ PeerRank(p2) < PeerRank(l))
    /\ needBlocks \in AllowedNeedBlocks(l)
    /\ LET hasFree    == Len(freeSlots) > 0
           slotIdx    == IF hasFree THEN freeSlots[Len(freeSlots)] ELSE used
           nextUsed   == IF hasFree THEN used ELSE used + 1
           nextFree   == IF hasFree THEN SubSeq(freeSlots, 1, Len(freeSlots) - 1) ELSE freeSlots
           needStride == AlignUp(needBlocks)
           newCap     == IF nextUsed > cap THEN NextCap(nextUsed) ELSE cap
           strideGrew == needStride > btStride
           newStride  == IF strideGrew THEN needStride ELSE btStride
           grownCs    == [s \in 0..(MaxCap - 1) |-> IF s < cap THEN csBuf[s] ELSE 0]
           grownBt    == IF newCap > cap \/ strideGrew
                         THEN IF strideGrew /\ Bug # "flat_grow"
                              THEN [i \in 0..(MaxBtCells - 1) |->
                                      LET s   == i \div (2 * newStride)
                                          rem == i % (2 * newStride)
                                          swa == rem \div newStride
                                          off == rem % newStride
                                      IN IF s < cap /\ off < btStride
                                         THEN btBuf[BtBase(s, swa, btStride) + off]
                                         ELSE 0]
                              ELSE [i \in 0..(MaxBtCells - 1) |->
                                      IF i < cap * 2 * btStride THEN btBuf[i] ELSE 0]
                         ELSE btBuf
           finalCs    == IF Bug = "no_zero_reuse"
                         THEN grownCs
                         ELSE [grownCs EXCEPT ![slotIdx] = 0]
           finalBt    == IF Bug = "no_zero_reuse"
                         THEN grownBt
                         ELSE [i \in 0..(MaxBtCells - 1) |->
                                 IF i >= slotIdx * 2 * newStride /\ i < (slotIdx + 1) * 2 * newStride
                                 THEN 0
                                 ELSE grownBt[i]]
       IN /\ newCap <= MaxCap
          /\ cap'        = newCap
          /\ used'       = nextUsed
          /\ btStride'   = newStride
          /\ freeSlots'  = nextFree
          /\ csBuf'      = finalCs
          /\ btBuf'      = finalBt
          /\ laneActive' = [laneActive EXCEPT ![l] = TRUE]
          /\ laneSlot'   = [laneSlot EXCEPT ![l] = slotIdx]
          /\ laneBlocks' = [laneBlocks EXCEPT ![l] = needBlocks]
          /\ cpuTable'   = [cpuTable EXCEPT ![l] = [swa \in 0..1 |-> [b \in 0..(MaxBlocks - 1) |-> 0]]]
          /\ dirty'      = [dirty EXCEPT ![l] = [swa \in 0..1 |-> CleanRange]]
          /\ fullSync'   = [fullSync EXCEPT ![l] = [swa \in 0..1 |-> TRUE]]
          /\ freshAlloc' = [freshAlloc EXCEPT ![l] = TRUE]
    /\ UNCHANGED recycles

MarkDirty(curDirty, isFull, blk) ==
    IF isFull THEN curDirty
    ELSE IF curDirty = CleanRange \/ Bug = "narrow_dirty"
         THEN <<blk, blk>>
         ELSE <<IF curDirty[1] < blk THEN curDirty[1] ELSE blk,
                IF curDirty[2] > blk THEN curDirty[2] ELSE blk>>

\* Fine-grained per-block mutation (`BlockTable::fill` / `clear_kind_entry`) on `ProbeLane`
\* `Window` table (`swa = 1`), exercising non-adjacent `[dirty_lo..=dirty_hi]` coalescing
\* and single-cell `clear_kind_entry` zeroing.
MutateProbeWindow(blk, clear) ==
    /\ laneActive[ProbeLane]
    /\ \A p \in PeerLanes : laneActive[p] => ~freshAlloc[p]
    /\ blk \in 0..(laneBlocks[ProbeLane] - 1)
    /\ blk \in {0, laneBlocks[ProbeLane] - 1}
    /\ (clear => (dirty[ProbeLane][1] = CleanRange /\ ~fullSync[ProbeLane][1]))
    /\ LET s   == laneSlot[ProbeLane]
           val == IF clear THEN 0 ELSE EntryWord(s, 1, blk)
       IN /\ cpuTable[ProbeLane][1][blk] # val
          /\ cpuTable' = [cpuTable EXCEPT ![ProbeLane][1][blk] = val]
          /\ dirty'    = [dirty EXCEPT ![ProbeLane][1] = MarkDirty(@, fullSync[ProbeLane][1], blk)]
    /\ UNCHANGED <<cap, used, btStride, freeSlots, recycles, csBuf, btBuf,
                   laneActive, laneSlot, laneBlocks, fullSync, freshAlloc>>

DirtyRangeOf(l, swa) ==
    IF laneBlocks[l] = 0 THEN CleanRange
    ELSE IF fullSync[l][swa] THEN <<0, laneBlocks[l] - 1>>
    ELSE dirty[l][swa]

\* Populate & upload `[0..laneBlocks[ProbeLane])` on `ProbeLane`'s `Full` table (`swa = 0`)
\* and dirty `cs_buf`, while `Window` (`swa = 1`) exercises async `fullSync` and `[dirty_lo..=dirty_hi]`.
PopulateProbeFull ==
    /\ laneActive[ProbeLane]
    /\ \A p \in PeerLanes : laneActive[p] => ~freshAlloc[p]
    /\ dirty[ProbeLane][1] = CleanRange
    /\ LET s     == laneSlot[ProbeLane]
           nb    == laneBlocks[ProbeLane]
           b0    == BtBase(s, 0, btStride)
           nextF == [b \in 0..(MaxBlocks - 1) |->
                       IF b < nb THEN EntryWord(s, 0, b) ELSE 0]
       IN /\ (nextF # cpuTable[ProbeLane][0] \/ DirtyRangeOf(ProbeLane, 0) # CleanRange \/ csBuf[s] = 0)
          /\ csBuf'      = [csBuf EXCEPT ![s] = 1]
          /\ cpuTable'   = [cpuTable EXCEPT ![ProbeLane][0] = nextF]
          /\ btBuf'      = [i \in 0..(MaxBtCells - 1) |->
                              IF i >= b0 /\ i < b0 + nb
                              THEN nextF[i - b0]
                              ELSE btBuf[i]]
          /\ dirty'      = [dirty EXCEPT ![ProbeLane][0] = CleanRange]
          /\ fullSync'   = [fullSync EXCEPT ![ProbeLane][0] = FALSE]
          /\ freshAlloc' = [freshAlloc EXCEPT ![ProbeLane] = FALSE]
    /\ UNCHANGED <<cap, used, btStride, freeSlots, recycles,
                   laneActive, laneSlot, laneBlocks>>

\* `BlockTable::clear_entries` (`src/blocks.rs:1036-1047`): clears CPU table and marks full sync.
ResetProbeTable ==
    /\ laneActive[ProbeLane]
    /\ \A p \in PeerLanes : laneActive[p] => ~freshAlloc[p]
    /\ csBuf[laneSlot[ProbeLane]] = 0
    /\ dirty[ProbeLane][1] = CleanRange
    /\ ~fullSync[ProbeLane][1]
    /\ \E swa \in 0..1, b \in 0..(laneBlocks[ProbeLane] - 1) : cpuTable[ProbeLane][swa][b] # 0
    /\ cpuTable'   = [cpuTable EXCEPT ![ProbeLane] = [swa \in 0..1 |-> [b \in 0..(MaxBlocks - 1) |-> 0]]]
    /\ dirty'      = [dirty EXCEPT ![ProbeLane] = [swa \in 0..1 |-> CleanRange]]
    /\ fullSync'   = [fullSync EXCEPT ![ProbeLane] = [swa \in 0..1 |-> TRUE]]
    /\ freshAlloc' = [freshAlloc EXCEPT ![ProbeLane] = FALSE]
    /\ UNCHANGED <<cap, used, btStride, freeSlots, recycles, csBuf, btBuf,
                   laneActive, laneSlot, laneBlocks>>

\* `BlockTable::sync_dirty_to_slice` (`src/blocks.rs:907-933`) on `ProbeLane`.
SyncDirtyProbe(swa) ==
    /\ laneActive[ProbeLane]
    /\ \A p \in PeerLanes : laneActive[p] => ~freshAlloc[p]
    /\ swa \in 0..1
    /\ LET rng  == DirtyRangeOf(ProbeLane, swa)
           base == BtBase(laneSlot[ProbeLane], swa, btStride)
       IN /\ rng # CleanRange
          /\ btBuf'      = [i \in 0..(MaxBtCells - 1) |->
                              IF i >= base + rng[1] /\ i <= base + rng[2]
                              THEN cpuTable[ProbeLane][swa][i - base]
                              ELSE btBuf[i]]
          /\ dirty'      = [dirty EXCEPT ![ProbeLane][swa] = CleanRange]
          /\ fullSync'   = [fullSync EXCEPT ![ProbeLane][swa] = FALSE]
          /\ freshAlloc' = [freshAlloc EXCEPT ![ProbeLane] = FALSE]
    /\ UNCHANGED <<cap, used, btStride, freeSlots, recycles, csBuf,
                   laneActive, laneSlot, laneBlocks, cpuTable>>

\* Peer lane step: populates & uploads its `[full | win]` block tables and marks `cs` dirty.
StepPeer(l) ==
    /\ l \in PeerLanes
    /\ laneActive[l]
    /\ freshAlloc[l]
    /\ LET s     == laneSlot[l]
           nb    == laneBlocks[l]
           b0    == BtBase(s, 0, btStride)
           b1    == BtBase(s, 1, btStride)
           nextT == [swa \in 0..1 |->
                       [b \in 0..(MaxBlocks - 1) |->
                           IF b < nb THEN EntryWord(s, swa, b) ELSE 0]]
       IN /\ csBuf'      = [csBuf EXCEPT ![s] = 1]
          /\ cpuTable'   = [cpuTable EXCEPT ![l] = nextT]
          /\ btBuf'      = [i \in 0..(MaxBtCells - 1) |->
                              IF i >= b0 /\ i < b0 + nb
                              THEN nextT[0][i - b0]
                              ELSE IF i >= b1 /\ i < b1 + nb
                              THEN nextT[1][i - b1]
                              ELSE btBuf[i]]
          /\ dirty'      = [dirty EXCEPT ![l] = [swa \in 0..1 |-> CleanRange]]
          /\ fullSync'   = [fullSync EXCEPT ![l] = [swa \in 0..1 |-> FALSE]]
          /\ freshAlloc' = [freshAlloc EXCEPT ![l] = FALSE]
    /\ UNCHANGED <<cap, used, btStride, freeSlots, recycles, laneActive, laneSlot, laneBlocks>>

\* `Drop for BlockedSlot` (`src/engine.rs:679-683`) on a dirty peer slot, exercising LIFO recycling.
DropLane(l) ==
    /\ l \in PeerLanes
    /\ laneActive[l]
    /\ ~freshAlloc[l]
    /\ \A p2 \in PeerLanes : laneActive[p2] => ~freshAlloc[p2]
    /\ ~\E p2 \in PeerLanes :
            laneActive[p2] /\ ~freshAlloc[p2]
            /\ laneBlocks[p2] = laneBlocks[l]
            /\ PeerRank(p2) < PeerRank(l)
    /\ dirty[ProbeLane][1] = CleanRange
    /\ recycles < MaxRecycles
    /\ freeSlots'  = Append(freeSlots, laneSlot[l])
    /\ recycles'   = recycles + 1
    /\ laneActive' = [laneActive EXCEPT ![l] = FALSE]
    /\ laneSlot'   = [laneSlot EXCEPT ![l] = NoSlot]
    /\ laneBlocks' = [laneBlocks EXCEPT ![l] = 0]
    /\ cpuTable'   = [cpuTable EXCEPT ![l] = [swa \in 0..1 |-> [b \in 0..(MaxBlocks - 1) |-> 0]]]
    /\ dirty'      = [dirty EXCEPT ![l] = [swa \in 0..1 |-> CleanRange]]
    /\ fullSync'   = [fullSync EXCEPT ![l] = [swa \in 0..1 |-> FALSE]]
    /\ freshAlloc' = [freshAlloc EXCEPT ![l] = FALSE]
    /\ UNCHANGED <<cap, used, btStride, csBuf, btBuf>>

DoneStutter ==
    /\ UNCHANGED vars

Next ==
    \/ (\E l \in Lanes, nb \in {InitBlocks, MaxBlocks} : AllocLane(l, nb))
    \/ PopulateProbeFull
    \/ ResetProbeTable
    \/ (\E swa \in 0..1 : SyncDirtyProbe(swa))
    \/ (\E b \in 0..(MaxBlocks - 1), clr \in BOOLEAN : MutateProbeWindow(b, clr))
    \/ (\E l \in PeerLanes : StepPeer(l))
    \/ (\E l \in PeerLanes : DropLane(l))
    \/ DoneStutter

Spec == Init /\ [][Next]_vars

\* ---- Invariants -----------------------------------------------------------

TypeOK ==
    /\ cap \in 0..MaxCap
    /\ used \in 0..cap
    /\ btStride \in 1..MaxStride
    /\ btStride % Align = 0
    /\ recycles \in 0..MaxRecycles
    /\ Len(freeSlots) <= used
    /\ \A i \in 1..Len(freeSlots) : freeSlots[i] \in 0..(used - 1)
    /\ \A i, j \in 1..Len(freeSlots) : i # j => freeSlots[i] # freeSlots[j]

\* Active lanes never alias each other's slot, free-list entries, or `[full | win]` regions.
SlotIsolation ==
    /\ \A l \in Lanes :
        laneActive[l] =>
            /\ laneSlot[l] \in 0..(used - 1)
            /\ laneBlocks[l] <= btStride
            /\ ~\E i \in 1..Len(freeSlots) : freeSlots[i] = laneSlot[l]
    /\ \A l1, l2 \in Lanes :
        (laneActive[l1] /\ laneActive[l2] /\ l1 # l2) =>
            /\ laneSlot[l1] # laneSlot[l2]
            /\ \A swa1, swa2 \in 0..1 :
                BtBase(laneSlot[l1], swa1, btStride) + btStride
                    <= BtBase(laneSlot[l2], swa2, btStride)
                \/ BtBase(laneSlot[l2], swa2, btStride) + btStride
                    <= BtBase(laneSlot[l1], swa1, btStride)
    /\ \A l \in Lanes :
        laneActive[l] =>
            BtBase(laneSlot[l], 0, btStride) + btStride = BtBase(laneSlot[l], 1, btStride)

\* Freshly allocated or recycled slots start with zeroed `cs` and zeroed GPU block tables,
\* and no active lane ever sees leftover block-table entries past `laneBlocks[l]`.
CleanSlotZeroed ==
    \A l \in Lanes :
        laneActive[l] =>
            /\ (freshAlloc[l] => csBuf[laneSlot[l]] = 0)
            /\ \A swa \in 0..1 :
                /\ (\A b \in laneBlocks[l]..(btStride - 1) :
                        btBuf[BtBase(laneSlot[l], swa, btStride) + b] = 0)
                /\ (freshAlloc[l] =>
                        (\A b \in 0..(btStride - 1) :
                            btBuf[BtBase(laneSlot[l], swa, btStride) + b] = 0))

\* Every difference between `cpuTable[l][swa]` and `btBuf` is covered by `DirtyRangeOf(l, swa)`;
\* once synced (`DirtyRangeOf == CleanRange`), GPU and CPU block tables match bit-for-bit,
\* including across `BlockedSlab` capacity and stride re-layouts!
SyncedMatchesCpu ==
    \A l \in Lanes, swa \in 0..1 :
        laneActive[l] =>
            LET rng  == DirtyRangeOf(l, swa)
                base == BtBase(laneSlot[l], swa, btStride)
            IN \A b \in 0..(laneBlocks[l] - 1) :
                btBuf[base + b] # cpuTable[l][swa][b] =>
                    (rng # CleanRange /\ b >= rng[1] /\ b <= rng[2])

=============================================================================
