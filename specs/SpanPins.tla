------------------------------ MODULE SpanPins ------------------------------
EXTENDS Integers, FiniteSets
CONSTANTS B, W, C, Bug
VARIABLES offset, selected, pin
vars == <<offset, selected, pin>>
\* First queried position is W-1+offset. Its oldest key is offset;
\* the last query is W-2+offset+C. Include both endpoints.
Required(o) == {p \div B : p \in o..(W - 2 + o + C)}
Bound == (W - 1 + C + B - 1) \div B + 1
Init == /\ offset = 0 /\ selected = FALSE /\ pin = {}
Select(o) ==
 /\ ~selected /\ o \in 0..(B-1)
 /\ offset' = o /\ selected' = TRUE
 /\ pin' = IF Bug THEN {b \in Required(o) : b < 17} ELSE Required(o)
Next == \E o \in 0..(B-1) : (~Bug \/ o = B-1) /\ Select(o)
Spec == Init /\ [][Next]_vars
Covered == selected => Required(offset) \subseteq pin
BoundHolds == selected => Cardinality(Required(offset)) <= Bound
\* This is a witness to a real maximum, not merely a loose upper bound.
MaximumIs25 == Cardinality(Required(B-1)) = 25 /\ Bound = 25
=============================================================================
