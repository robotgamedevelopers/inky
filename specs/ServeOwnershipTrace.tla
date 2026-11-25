------------------------- MODULE ServeOwnershipTrace -------------------------
\* Trace conformance specification for production `ServeBlockCache`
\* (`src/serve/block_cache.rs`), `BlockPools` / `BlockTable` (`src/blocks.rs`),
\* `EntryTree` (`src/prefix.rs`), and `SpillStore` (`src/spill.rs`).
\*
\* Replay checks multi-lane warm binds, singleflight rebinds, prefill/decode
\* span steps, sliding-window pruning, LRU start eviction + SSD spill, SSD
\* restore, corrupt-spill checksum rejection, namespace isolation, and
\* transactional `ensure_span_blocks` rollback on `PoolExhausted`.

EXTENDS Integers, Sequences, FiniteSets, ServeOwnershipTraceData

VARIABLES traceIdx, step, prevTableFullRefs, prevTableWinRefs, prevFullGen, prevWinGen, prevLaneFull, prevLaneWin

vars == <<traceIdx, step, prevTableFullRefs, prevTableWinRefs, prevFullGen, prevWinGen, prevLaneFull, prevLaneWin>>

CurTrace == Traces[traceIdx]
Row == CurTrace[step]

SumSeq(seq) ==
    LET RECURSIVE Helper(_)
        Helper(i) == IF i > Len(seq) THEN 0 ELSE seq[i] + Helper(i + 1)
    IN Helper(1)

ValidOps == {
    "init",
    "bind",
    "rebind",
    "ensure_ok",
    "ensure_rollback",
    "prefill_finish",
    "decode_finish",
    "release_lane",
    "evict_lru",
    "maintain_spill",
    "corrupt_restore_rejected",
    "ns_isolated",
    "shutdown"
}

Init ==
    /\ traceIdx \in 1..Len(Traces)
    /\ step = 1
    /\ Row.op = "init"
    /\ prevTableFullRefs = Row.tableFullRefs
    /\ prevTableWinRefs = Row.tableWinRefs
    /\ prevFullGen = Row.fullGen
    /\ prevWinGen = Row.winGen
    /\ prevLaneFull = Row.laneFullBlocks
    /\ prevLaneWin = Row.laneWinBlocks

Next ==
    \/ /\ step < Len(CurTrace)
       /\ prevTableFullRefs' = Row.tableFullRefs
       /\ prevTableWinRefs' = Row.tableWinRefs
       /\ prevFullGen' = Row.fullGen
       /\ prevWinGen' = Row.winGen
       /\ prevLaneFull' = Row.laneFullBlocks
       /\ prevLaneWin' = Row.laneWinBlocks
       /\ step' = step + 1
       /\ UNCHANGED traceIdx
    \/ /\ step = Len(CurTrace)
       /\ UNCHANGED vars

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ traceIdx \in 1..Len(Traces)
    /\ step \in 1..Len(CurTrace)
    /\ Row.op \in ValidOps
    /\ Len(Row.fullRefs) = Row.numFull
    /\ Len(Row.winRefs) = Row.numWin
    /\ Len(Row.fullGen) = Len(prevFullGen)
    /\ Len(Row.winGen) = Len(prevWinGen)
    /\ \A s \in 1..Row.numFull : Row.fullGen[s] >= prevFullGen[s]
    /\ \A s \in 1..Row.numWin : Row.winGen[s] >= prevWinGen[s]
    /\ Len(Row.startFullRefs) = Row.numFull
    /\ Len(Row.startWinRefs) = Row.numWin
    /\ Len(Row.tableFullRefs) = Row.numFull
    /\ Len(Row.tableWinRefs) = Row.numWin

\* Exact conservation of every BlockPool slot reference:
\* pool.record(slot).refs() == active_lane_table_refs[slot] + actual_starts_refs[slot].
PoolRefConservation ==
    /\ \A s \in 1..Row.numFull :
        /\ Row.fullRefs[s] = Row.tableFullRefs[s] + Row.actualStartFullRefs[s]
        /\ (Row.fullRefs[s] > 0) => (Row.fullGen[s] > 0)
    /\ \A s \in 1..Row.numWin :
        /\ Row.winRefs[s] = Row.tableWinRefs[s] + Row.actualStartWinRefs[s]
        /\ (Row.winRefs[s] > 0) => (Row.winGen[s] > 0)

\* ServeBlockCache's incremental `start_slot_refs` array matches actual RetainedStart holdings,
\* and `free_plus_evictable_slots` equals the slots with zero active-lane references.
StartRefsConsistent ==
    /\ Row.startFullRefs = Row.actualStartFullRefs
    /\ Row.startWinRefs = Row.actualStartWinRefs
    /\ Row.freePlusEvictFull =
        Cardinality({s \in 1..Row.numFull : Row.tableFullRefs[s] = 0})
    /\ Row.freePlusEvictWin =
        Cardinality({s \in 1..Row.numWin : Row.tableWinRefs[s] = 0})

\* Every active lane table entry names a live slot at its current generation,
\* every RetainedStart entry is live and sealed with start_holders > 0 in EntryTree,
\* and EntryTree pins match active lanes' sealed prefix paths.
GenerationAndTreeConsistent ==
    /\ Row.allLaneEntriesLive = TRUE
    /\ Row.allStartEntriesLiveSealed = TRUE
    /\ Row.treeStartHoldersMatch = TRUE
    /\ Row.treePinsMatchLanes = TRUE

\* Sliding-window pruning removes expired window blocks so no lane keeps expired
\* window entries past EXT (at most MaxWinPerLane blocks per lane).
WindowSpanBounded ==
    /\ Row.expiredWinPresent = FALSE
    /\ \A l \in 1..Len(Row.laneWinCount) : Row.laneWinCount[l] <= Row.maxWinPerLane

\* Prefix bind and rebind always select the deepest live checkpoint strictly shorter than prompt.
DeepestStartBound ==
    (Row.op \in {"bind", "rebind"}) => (Row.matchedAt = Row.expectedDeepestAt)

\* Disk block ownership in `spill_key_owners` matches `SpillStore` contents and `spilled_starts`.
SpillOwnershipConsistent ==
    /\ Row.spillOwnersConsistent = TRUE
    /\ Row.spilledTipsConsistent = TRUE

\* Corrupt `.blk` payloads on SSD are rejected by SHA-256 verification and unlinked, never restored.
CorruptSpillNeverRestored ==
    /\ Row.corruptRestored = FALSE
    /\ (Row.op = "corrupt_restore_rejected") => (Row.corruptCleaned = TRUE)

\* Distinct cache namespaces never share RAM prefix hits or SSD restores.
NamespaceIsolated ==
    (Row.op = "ns_isolated") => (Row.matchedAt = 0)

\* Explicit state invariant for rollback steps: active lane table refs and block bindings
\* remain strictly unchanged even if LRU eviction freed cold start entries first.
RollbackAtomic ==
    (Row.op = "ensure_rollback") =>
        /\ Row.tableFullRefs = prevTableFullRefs
        /\ Row.tableWinRefs = prevTableWinRefs
        /\ Row.laneFullBlocks = prevLaneFull
        /\ Row.laneWinBlocks = prevLaneWin

\* Clean shutdown drains all retained start references and active lane references.
CleanAtShutdown ==
    (Row.op = "shutdown") =>
        /\ SumSeq(Row.fullRefs) = 0
        /\ SumSeq(Row.winRefs) = 0
        /\ Row.retainedStarts = 0

=============================================================================
