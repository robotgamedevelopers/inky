--------------------------- MODULE TraceCheck ---------------------------
EXTENDS MemMode, TraceData, TraceProtocol, Sequences
VARIABLES trace, pos
TraceState == Traces[trace][pos]
Matches == /\ limit = TraceState.limit /\ saved = TraceState.saved
           /\ disk = TraceState.disk /\ conns = TraceState.conns
           /\ alive = TraceState.alive /\ owed = TraceState.owed
           /\ owner = TraceState.owner
\* Lift the concrete call log into the model's rollback observation, rather
\* than inferring an attempt from Mode.saved or choosing a model branch.
ObservedRollback(p,s) ==
    LET cs == s.calls
        n == IF p.saved = NONE THEN 3 ELSE 1
        found == IF p.saved = NONE THEN p.limit ELSE p.saved
        attempted == Call(cs,n+2,"set",found)
    IN IF s.op = "idle" THEN rollback
       ELSE IF s.op = "connect" /\ p.conns = {} /\
               Call(cs,n,"set",Target) /\ At(cs,n).result # 0
            THEN [needed |-> At(cs,n+1).result # found, attempted |-> attempted,
                  restored |-> IF attempted THEN cs[n+2].effect = "assigned" ELSE FALSE,
                  found |-> found]
            ELSE NoRollback
ReplayInit == /\ trace \in 1..Len(Traces) /\ pos = 1 /\ Matches /\ rollback = NoRollback
ReplayNext == /\ UNCHANGED trace
              /\ pos' = IF pos < Len(Traces[trace]) THEN pos + 1 ELSE pos
              /\ rollback' = IF pos' = pos THEN rollback ELSE ObservedRollback(Traces[trace][pos],Traces[trace][pos'])
              /\ limit' = Traces[trace][pos'].limit
              /\ saved' = Traces[trace][pos'].saved
              /\ disk' = Traces[trace][pos'].disk
              /\ conns' = Traces[trace][pos'].conns
              /\ alive' = Traces[trace][pos'].alive
              /\ owed' = Traces[trace][pos'].owed
              /\ owner' = Traces[trace][pos'].owner
Replay == ReplayInit /\ [][ReplayNext]_<<vars, trace, pos>>
\* Finite observed prefixes check initial and next-state conformance, not fairness.
Conforms == Init /\ [][[Next]_vars /\ Protocol(Traces[trace][pos], Traces[trace][pos'], Target)]_<<vars, trace, pos>>
=============================================================================
