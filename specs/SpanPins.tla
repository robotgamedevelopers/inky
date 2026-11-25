------------------------------ MODULE SpanPins ------------------------------
EXTENDS Integers, FiniteSets
CONSTANTS B, W, C, Bug
ASSUME /\ B > 0 /\ W >= B /\ C > 0
       /\ Bug \in {"none", "cap17", "overbound"}
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
 /\ pin' = IF Bug = "cap17" THEN {b \in Required(o) : b < 17}
           ELSE IF Bug = "overbound" THEN Required(o) \cup {Bound}
           ELSE Required(o)
Next == \E o \in 0..(B-1) : (Bug = "none" \/ o = B-1) /\ Select(o)
Spec == Init /\ [][Next]_vars
Covered == selected => Required(offset) \subseteq pin
BoundHolds == selected => Cardinality(pin) <= Bound
\* State-dependent witness (`U169`): at worst alignment `offset = B - 1`, `pin`
\* reaches 25 window blocks (`Bound = 25`), failing on a 448-token chunk (`SpanPinsMax24`).
MaximumIs25 == (selected /\ offset = B - 1 /\ Bug = "none") => Cardinality(pin) = 25 /\ Bound = 25
=============================================================================
