--------------------------- MODULE PrefixTreeTrace ---------------------------
\* The trace binding for PrefixTree: the code's seeded walk
\* (src/prefix.rs, prefix_trace_dump) emits one row per operation with
\* the state it actually reached, PrefixTreeTraceData holds the rows,
\* and this spec admits exactly the walk's behaviour — each step must
\* be the named action of PrefixTree.tla landing in the emitted state.
\* If the code and the model ever disagree (as they did on eviction),
\* some row has no model step and TLC says which.
\*
\* Abstract generations are the records' generations minus one, so
\* the model starts at 0; slots are the integers 1..3 here.

EXTENDS PrefixTree, PrefixTreeTraceData

VARIABLE step
tvars == <<vars, step>>

Matches(sv, tv, ov, gv, pv, rv, qv, hv, row) ==
    /\ sv = row.sealed
    /\ tv = row.starts
    /\ ov = row.slotOf
    /\ gv = row.genOf
    /\ pv = row.poolGen
    /\ rv = row.refs
    /\ qv = row.result
    /\ hv = row.sought

Now(row) == Matches(sealed, starts, slotOf, genOf, poolGen, refs, result, sought, row)
Post(row) == Matches(sealed', starts', slotOf', genOf', poolGen', refs', result', sought', row)

\* A refused stale insert is no model step at all: the code left the
\* state untouched, and the correct model (Bug = "evict", so
\* SealClaimed is disabled) has no action that could take it.
StepOp(row) ==
    CASE row.op = "seal" -> Seal(row.s, row.slot)
      [] row.op = "seal-stale" -> UNCHANGED vars
      [] row.op = "seal-dup" -> Dedup(row.s)
      [] row.op = "seal-twin" -> ReSeal(row.s, row.slot)
      [] row.op = "retain" -> Retain(row.s)
      [] row.op = "release" -> Release(row.s)
      [] row.op = "reuse" -> Reuse(row.slot)
      [] row.op = "remove" -> Evict(row.s)
      [] row.op = "seek" -> Seek

\* The pool's books, read from the records at every row: the refs the
\* pool holds on a slot are exactly the live blocks sealed onto it.
\* The model's own refs are per block; this binds the pool's counts,
\* so a dropped ref (or a leaked one) has nowhere to hide.
PoolBooks ==
    LET row == Trace[step] IN
        \A slot \in Slots :
            row.poolRefs[slot] =
                Cardinality({s \in sealed \ {<<>>} : slotOf[s] = slot})

TInit ==
    /\ Init
    /\ Now(Trace[1])
    /\ step = 1

TNext ==
    \/ /\ step < Len(Trace)
       /\ LET row == Trace[step + 1] IN StepOp(row) /\ Post(row)
       /\ step' = step + 1
    \/ /\ step = Len(Trace)
       /\ UNCHANGED tvars

TSpec == TInit /\ [][TNext]_tvars
=============================================================================
