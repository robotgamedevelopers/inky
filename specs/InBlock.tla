------------------------------ MODULE InBlock -----------------------------
\* The bytes inside one block slot. Order is [pass][K|V][head][position][HD],
\* each element Kvb bytes. Bounds: 2 passes, 3 heads, 4 positions, HD = 2.
\* The action stores addr by inlining the arithmetic. It does not call Formula
\* or AddrOf. Bug = "major" stores index * (stride / B), the position-major step.
\* Ordered compares addr to Formula. NoCollision quantifies over AddrOf, so the
\* major step collides. Inside says one row stays in the slot.
EXTENDS Integers

CONSTANTS Kvb, Bug
ASSUME Kvb \in {1, 2} /\ Bug \in {"none", "major"}

Passes == 2
Heads == 3
B == 4
HD == 2

stride == Passes * 2 * Heads * B * HD * Kvb

\* Not called by Next.
Formula(p, k, h, i) ==
  ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

\* What this Bug stores. Not called by Next. The major arm ignores pass, plane and head.
AddrOf(p, k, h, i) ==
  IF Bug = "major"
  THEN i * (stride \div B)
  ELSE ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

Coords == (0..(Passes - 1)) \X (0..1) \X (0..(Heads - 1)) \X (0..(B - 1))

VARIABLES p, k, h, i, addr
vars == <<p, k, h, i, addr>>

Init ==
  /\ p = 0 /\ k = 0 /\ h = 0 /\ i = 0
  /\ addr = 0

Next == \E pp \in 0..(Passes - 1), kk \in 0..1, hh \in 0..(Heads - 1), ii \in 0..(B - 1) :
  /\ <<pp, kk, hh, ii>> # <<p, k, h, i>>
  /\ p' = pp /\ k' = kk /\ h' = hh /\ i' = ii
  /\ addr' = IF Bug = "major"
             THEN ii * (stride \div B)
             ELSE ((pp * 2 + kk) * Heads + hh) * B * HD * Kvb + ii * HD * Kvb

Ordered == addr = Formula(p, k, h, i)

Inside == addr + HD * Kvb <= stride

NoCollision == \A c, d \in Coords :
  c # d => AddrOf(c[1], c[2], c[3], c[4]) # AddrOf(d[1], d[2], d[3], d[4])

Spec == Init /\ [][Next]_vars
=============================================================================
