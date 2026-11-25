--------------------------- MODULE ServeStepTrace ---------------------------
\* One frame per invocation of the step planner used by serve. The workload
\* supplies runnable and pending flags and both positions. The plan and wait counters on return
\* are observations checked against the policy below.
EXTENDS Naturals, Sequences, FiniteSets, ServeStepTraceData

VARIABLE frame
Init == frame = 1
Next == IF frame < Len(Frames) THEN frame' = frame + 1 ELSE UNCHANGED frame
Spec == Init /\ [][Next]_frame

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

\* Deterministic priority inside a phase: an older wait wins, then lane ID.
Before(f, i, j) ==
    \/ f.before[i] > f.before[j]
    \/ /\ f.before[i] = f.before[j]
       /\ i < j

RECURSIVE Tokens(_, _)
Tokens(f, lanes) ==
    IF lanes = {} THEN 0 ELSE
        LET i == CHOOSE lane \in lanes : TRUE
        IN f.spans[i] + Tokens(f, lanes \ {i})

Phase(f, decode, prefill) ==
    IF decode = {} THEN IF prefill = {} THEN "idle" ELSE "prefill"
    ELSE IF /\ \E i \in prefill : f.before[i] = f.ceiling
            /\ \A i \in decode : f.before[i] < f.ceiling
         THEN "prefill" ELSE "decode"

\* Each earlier lane has a uniquely determined allocation. Use its observed
\* span to compute the residual budget; its own equation is checked as well.
Span(f, candidates, i) ==
    IF i \notin candidates THEN 0 ELSE
        LET earlier == {j \in candidates : Before(f, j, i)}
            left == Max(1, f.budget) - Tokens(f, earlier)
            owed == {j \in candidates : Before(f, i, j) /\ f.before[j] = f.ceiling}
            remaining == f.endPos[i] - f.pos[i]
            grant == Min(remaining, Max(1, left - Cardinality(owed)))
        IN  IF left <= 0 THEN 0
            ELSE IF f.mode = "decode" THEN 1
            ELSE IF /\ f.align > 0 /\ grant > f.align /\ grant < remaining
                 THEN grant - ((f.pos[i] + grant) % f.align)
                 ELSE grant

ValidFrame(f) ==
    LET lanes == DOMAIN f.runnable
        decode == {i \in lanes : f.runnable[i] /\ f.pending[i] /\ f.pos[i] >= f.endPos[i]}
        prefill == {i \in lanes : f.runnable[i] /\ f.pos[i] < f.endPos[i]}
        candidates == CASE f.mode = "decode" -> decode
                        [] f.mode = "prefill" -> prefill
                        [] OTHER -> {}
    IN  /\ f.version = 2 /\ lanes # {}
        /\ DOMAIN f.pending = lanes
        /\ \A s \in {f.before, f.after, f.pos, f.endPos, f.starts, f.spans} : DOMAIN s = lanes
        /\ f.ceiling = 16 /\ f.budget \in Nat /\ f.align \in Nat
        /\ f.mode = Phase(f, decode, prefill)
        /\ \A i \in lanes :
               /\ f.before[i] \in 0..f.ceiling
               /\ f.runnable[i] \in BOOLEAN /\ f.pending[i] \in BOOLEAN
               /\ f.pos[i] \in Nat /\ f.endPos[i] \in Nat
               /\ f.spans[i] = Span(f, candidates, i)
               /\ f.starts[i] = IF f.spans[i] > 0 THEN f.pos[i] ELSE 0
               /\ f.after[i] = IF f.spans[i] > 0 THEN 0
                               ELSE IF i \in decode \cup prefill
                                    THEN Min(f.ceiling, f.before[i] + 1)
                                    ELSE f.before[i]

AgreesWithPlannerPolicy == IF Len(Frames) = 0 THEN FALSE ELSE ValidFrame(Frames[frame])
=============================================================================
