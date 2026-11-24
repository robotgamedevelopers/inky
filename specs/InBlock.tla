------------------------------ MODULE InBlock -----------------------------
\* The bytes inside one block slot. Order is [pass][K|V][head][position][HD],
\* each element Kvb bytes. Bounds: 2 passes, 3 heads, 4 positions, HD = 2.
\* Next walks those coordinates in order. It stores addr by inlining the
\* arithmetic, and appends that stored address to seen. It does not call
\* Formula. Ordered compares addr to Formula. NoCollision says the addresses
\* in seen are unique, so it reads the state the action wrote.
\* Bug = "major" stores index * (stride / B). Bug = "addrcol" drops the head
\* term from the stored address. Both collide on a reached state.
EXTENDS Integers, Sequences

CONSTANTS Kvb, Bug
ASSUME Kvb \in {1, 2} /\ Bug \in {"none", "major", "addrcol"}

Passes == 2
Heads == 3
B == 4
HD == 2
NCoords == Passes * 2 * Heads * B

stride == Passes * 2 * Heads * B * HD * Kvb

\* Not called by Next.
Formula(p, k, h, i) ==
  ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

At(p, k, h, i) == ((p * 2 + k) * Heads + h) * B + i

VARIABLES p, k, h, i, addr, seen
vars == <<p, k, h, i, addr, seen>>

Init ==
  /\ p = 0 /\ k = 0 /\ h = 0 /\ i = 0
  /\ addr = 0
  /\ seen = <<0>>

Next ==
  \/ /\ At(p, k, h, i) < NCoords - 1
     /\ LET n == At(p, k, h, i) + 1
            pp == n \div (2 * Heads * B)
            kk == (n \div (Heads * B)) - pp * 2
            hh == (n \div B) - (n \div (Heads * B)) * Heads
            ii == n - (n \div B) * B
        IN /\ p' = pp /\ k' = kk /\ h' = hh /\ i' = ii
           /\ addr' = IF Bug = "major" THEN ii * (stride \div B)
                      ELSE IF Bug = "addrcol"
                           THEN ((pp * 2 + kk) * Heads) * B * HD * Kvb + ii * HD * Kvb
                           ELSE ((pp * 2 + kk) * Heads + hh) * B * HD * Kvb + ii * HD * Kvb
           /\ seen' = Append(seen, addr')
  \/ /\ At(p, k, h, i) = NCoords - 1
     /\ UNCHANGED vars

Ordered == addr = Formula(p, k, h, i)

Inside == addr + HD * Kvb <= stride

NoCollision == \A x, y \in 1..Len(seen) :
  x # y => seen[x] # seen[y]

Spec == Init /\ [][Next]_vars
=============================================================================
