--------------------------- MODULE TraceCheck ---------------------------
EXTENDS MemMode, TraceData, TraceProtocol, Sequences
VARIABLES trace, pos
TraceState == Traces[trace][pos]
Matches == /\ limit = TraceState.limit /\ saved = TraceState.saved
           /\ disk = TraceState.disk /\ conns = TraceState.conns
           /\ alive = TraceState.alive /\ owed = TraceState.owed
           /\ owner = TraceState.owner
ReplayInit == /\ trace \in 1..Len(Traces) /\ pos = 1 /\ Matches
ReplayNext == /\ UNCHANGED trace
              /\ pos' = IF pos < Len(Traces[trace]) THEN pos + 1 ELSE pos
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
