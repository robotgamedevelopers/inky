--------------------------- MODULE StartTailsTrace ---------------------------
\* Replays the walks scripts/blockbook-traces.rs records from src/blockbook.rs.
\* Conforms requires the first state to be Init and every recorded call to be a
\* step of StartTails (or change nothing). RefsMatch, ReadsMatch and StatesMatch
\* check what the model does not hold as state: the slot records' reference
\* counts, the reads attention made through each lane's table, and the GPU side's
\* live recurrent states.
\*
\* A recorded state is <<full pool, window pool, tables, pos, prompt, running,
\* head, start points, kept states, reads>>, with kinds as 1 (full) and 2
\* (window); scripts/blockbook-traces.rs writes it.
EXTENDS StartTails, StartTailsTraceData, Sequences
VARIABLES trace, at

KindName == <<"full", "window">>
KindNo(kd) == IF kd = "full" THEN 1 ELSE 2
Tab(S) == [k \in Keys |->
    IF \E e \in S : e[1] = KindNo(k[1]) /\ e[2] = k[2]
    THEN LET e == CHOOSE e \in S : e[1] = KindNo(k[1]) /\ e[2] = k[2]
         IN [slot |-> e[3], generation |-> e[4]]
    ELSE NoEntry]
Rec(r) == [generation |-> r[1], length |-> r[2], sealed |-> r[3] = 1,
           data |-> [i \in 0..(B - 1) |-> r[5 + i]]]
PoolOf(s) == [kd \in Kinds |-> [x \in Slots |-> Rec(s[KindNo(kd)][x])]]
ReasonsOf(n) == (IF n % 2 = 1 THEN {"prompt"} ELSE {})
                \cup (IF (n \div 2) % 2 = 1 THEN {"every"} ELSE {})
                \cup (IF (n \div 4) % 2 = 1 THEN {"head"} ELSE {})
StartOf(p) == IF p[1] = 1
              THEN [live |-> TRUE, pos |-> p[2], state |-> p[3], table |-> Tab(p[6]),
                    reasons |-> ReasonsOf(p[5]), lane |-> p[4]]
              ELSE Dead
PerLane(t) == [l \in Lanes |-> t[l + 1]]

TS == Traces[trace][at]

ReplayInit ==
 /\ trace \in 1..Len(Traces) /\ at = 1
 /\ pool = PoolOf(TS)
 /\ table = [l \in Lanes |-> Tab(TS[3][l + 1])]
 /\ pos = PerLane(TS[4]) /\ prompt = PerLane(TS[5])
 /\ running = [l \in Lanes |-> TS[6][l + 1] = 1] /\ head = PerLane(TS[7])
 /\ starts = [i \in StartIds |-> StartOf(TS[8][i])]
 /\ freed = {}

ReplayNext ==
 /\ UNCHANGED trace
 /\ at' = IF at < Len(Traces[trace]) THEN at + 1 ELSE at
 /\ LET s == Traces[trace][at'] IN
    /\ pool' = PoolOf(s)
    /\ table' = [l \in Lanes |-> Tab(s[3][l + 1])]
    /\ pos' = PerLane(s[4]) /\ prompt' = PerLane(s[5])
    /\ running' = [l \in Lanes |-> s[6][l + 1] = 1] /\ head' = PerLane(s[7])
    /\ starts' = [i \in StartIds |-> StartOf(s[8][i])]
    /\ freed' = {}

Replay == ReplayInit /\ [][ReplayNext]_<<vars, trace, at>>

\* Finite recorded prefixes check initial and next-state conformance, not liveness.
InitConforms == (at = 1) => Init
StepConforms == [][Next]_vars

\* Every kept recurrent state belongs to one live start point.
StatesMatch == TS[9] = Cardinality({i \in StartIds : starts[i].live})

RefsMatch == \A kd \in Kinds, x \in Slots :
    TS[KindNo(kd)][x][4] = Cardinality(LaneRefs(kd, x)) + Cardinality(StartRefs(kd, x))

ReadsMatch == \A r \in TS[10] :
    /\ running[r[1]] /\ table[r[1]][<<KindName[r[2]], r[3]>>] # NoEntry
    /\ r[4] = Visible(pos[r[1]], r[3])
=============================================================================
