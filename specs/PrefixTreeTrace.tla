--------------------------- MODULE PrefixTreeTrace ---------------------------
\* The trace binding for PrefixTree: the code's multi-seed walks
\* (src/prefix.rs, prefix_trace_dump) emit one row per operation with
\* the state actually reached across multiple seeds (Traces), and this
\* spec requires every row in every seed trace to be a valid transition
\* of PrefixTree.tla with separate Full and Window pool generations and
\* per-kind pool reference books.

EXTENDS PrefixTree, PrefixTreeTraceData

VARIABLES traceIdx, step
tvars == <<vars, traceIdx, step>>

CurTrace == Traces[traceIdx]

Matches(sv, tv, ov, gv, wv, wg, pv, wpv, rv, qv, hv, rqv, row) ==
    /\ sv = row.sealed
    /\ tv = row.starts
    /\ ov = row.slotOf
    /\ gv = row.genOf
    /\ wv = row.winSlotOf
    /\ wg = row.winGenOf
    /\ pv = row.poolGen
    /\ wpv = row.winPoolGen
    /\ rv = row.refs
    /\ qv = row.result
    /\ hv = row.sought
    /\ rqv = row.req
    /\ row.bad = FALSE

Now(row) == Matches(sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen, refs, result, sought, req, row)
Post(row) == Matches(sealed', starts', slotOf', genOf', winSlotOf', winGenOf', poolGen', winPoolGen', refs', result', sought', req', row)

StepOp(row) ==
    CASE row.op = "seal" -> Seal(row.s, row.slot, row.wslot)
      [] row.op = "seal-stale" ->
            /\ UNCHANGED <<sealed, starts, slotOf, genOf, winSlotOf, winGenOf, poolGen, winPoolGen, refs, badInsert>>
            /\ sought' = FALSE
            /\ req' = ReqFull
            /\ result' = <<>>
      [] row.op = "seal-dup" -> Dedup(row.s)
      [] row.op = "seal-twin" -> ReSeal(row.s, row.slot, row.wslot)
      [] row.op = "pin" -> Pin(row.s)
      [] row.op = "unpin" -> Unpin(row.s)
      [] row.op = "retain" -> Retain(row.s)
      [] row.op = "release" -> Release(row.s)
      [] row.op = "reuse-full" -> ReuseFull(row.slot)
      [] row.op = "reuse-window" -> ReuseWindow(row.wslot)
      [] row.op = "remove" -> Evict(row.s)
      [] row.op = "seek" -> Seek(row.req)

\* Per-kind pool reference books: Full and Window reference counts on each
\* slot are checked independently rather than summed.
PoolBooks ==
    LET row == CurTrace[step] IN
        \A slot \in Slots :
            /\ row.poolRefs[slot] =
                Cardinality({s \in sealed \ {<<>>} : refs[s] = 1 /\ slotOf[s] = slot})
            /\ row.winPoolRefs[slot] =
                Cardinality({s \in sealed \ {<<>>} : refs[s] = 1 /\ winSlotOf[s] = slot})

TInit ==
    /\ traceIdx \in 1..Len(Traces)
    /\ Init
    /\ Now(CurTrace[1])
    /\ step = 1

TNext ==
    \/ /\ step < Len(CurTrace)
       /\ LET row == CurTrace[step + 1] IN StepOp(row) /\ Post(row)
       /\ step' = step + 1
       /\ UNCHANGED traceIdx
    \/ /\ step = Len(CurTrace)
       /\ UNCHANGED tvars

TSpec == TInit /\ [][TNext]_tvars
=============================================================================
