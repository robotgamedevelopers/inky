------------------------------ MODULE InBlock -----------------------------
\* The bytes inside one block slot. Order is [pass][K|V][head][position][HD],
\* each element Kvb bytes, so one head's positions are contiguous.
\* Bounds: 2 passes, 3 heads, 4 positions. Bug = "nohead" drops the head term,
\* so head 0 and head 1 share an address. Head 0 still matches, which is why the
\* witness is the step that first selects a later head.
EXTENDS Integers

CONSTANTS Kvb, Bug
ASSUME Kvb \in {1, 2} /\ Bug \in {"none", "nohead"}

Passes == 2
Heads == 3
B == 4
HD == 2
\* K = 0, V = 1.

Row(p, k, h, i) ==
  IF Bug = "nohead"
  THEN ((p * 2 + k) * Heads) * B * HD * Kvb + i * HD * Kvb
  ELSE ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

Coords == (0..(Passes - 1)) \X (0..1) \X (0..(Heads - 1)) \X (0..(B - 1))

VARIABLES p, k, h, i
vars == <<p, k, h, i>>

Init == p = 0 /\ k = 0 /\ h = 0 /\ i = 0

\* Any other coordinate. Every coordinate is reachable in one step, and none is dead.
Next == \E pp \in 0..(Passes - 1), kk \in 0..1, hh \in 0..(Heads - 1), ii \in 0..(B - 1) :
  /\ <<pp, kk, hh, ii>> # <<p, k, h, i>>
  /\ p' = pp /\ k' = kk /\ h' = hh /\ i' = ii

\* The address carries the head. The no-head step fails this as soon as h > 0.
HeadTerm ==
  Row(p, k, h, i) = ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

\* Two different coordinates do not share an address.
NoCollision == \A c, d \in Coords :
  c # d => Row(c[1], c[2], c[3], c[4]) # Row(d[1], d[2], d[3], d[4])

Spec == Init /\ [][Next]_vars
=============================================================================
