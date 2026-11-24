------------------------------ MODULE InBlock -----------------------------
\* The bytes inside one block slot. Order is [pass][K|V][head][position][HD],
\* each element Kvb bytes, so one head's positions are contiguous.
\* Bounds: 2 passes, 3 heads, 4 positions, HD = 2.
\* Bug = "major" stores index * (stride / B), the position-major step
\* `offset + index * (stride / 64)` at these bounds. The stored address is `addr`.
\* Ordered compares it to Formula, which this action does not call.
EXTENDS Integers

CONSTANTS Kvb, Bug
ASSUME Kvb \in {1, 2} /\ Bug \in {"none", "major"}

Passes == 2
Heads == 3
B == 4
HD == 2
\* K = 0, V = 1.

stride == Passes * 2 * Heads * B * HD * Kvb

\* Independent of the action. Pass, then K or V, then head, then position.
Formula(p, k, h, i) ==
  ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

\* The position-major step. It ignores pass, K or V, and head.
Major(i) == i * (stride \div B)

\* Engine-sized addresses. HD 128, 64 positions, 8 heads, kvb 2.
\* The same four numbers are asserted by key_row in src/blocks.rs.
RealRow(p, k, h, i) ==
  LET Rhd == 128 IN LET Rb == 64 IN LET Rheads == 8 IN LET Rkvb == 2 IN
    ((p * 2 + k) * Rheads + h) * Rb * Rhd * Rkvb + i * Rhd * Rkvb

Trace == {
  <<0, 0, 0, 0, 0>>,
  <<0, 0, 0, 1, 256>>,
  <<0, 0, 1, 0, 16384>>,
  <<0, 1, 2, 0, 163840>>
}

Coords == (0..(Passes - 1)) \X (0..1) \X (0..(Heads - 1)) \X (0..(B - 1))

VARIABLES p, k, h, i, addr
vars == <<p, k, h, i, addr>>

Init ==
  /\ p = 0 /\ k = 0 /\ h = 0 /\ i = 0
  /\ addr = IF Bug = "major" THEN Major(0) ELSE Formula(0, 0, 0, 0)

\* Any other coordinate. Every coordinate is reachable in one step, and none is dead.
Next == \E pp \in 0..(Passes - 1), kk \in 0..1, hh \in 0..(Heads - 1), ii \in 0..(B - 1) :
  /\ <<pp, kk, hh, ii>> # <<p, k, h, i>>
  /\ p' = pp /\ k' = kk /\ h' = hh /\ i' = ii
  /\ addr' = IF Bug = "major" THEN Major(ii) ELSE Formula(pp, kk, hh, ii)

\* The stored address is the independent formula, not a restatement the action calls.
Ordered == addr = Formula(p, k, h, i)

\* One row stays inside the slot.
Inside == addr + HD * Kvb <= stride

\* Two different coordinates do not share a formula address.
NoCollision == \A c, d \in Coords :
  c # d => Formula(c[1], c[2], c[3], c[4]) # Formula(d[1], d[2], d[3], d[4])

\* The engine trace. Constant, so every state checks it.
TraceOk == \A t \in Trace : RealRow(t[1], t[2], t[3], t[4]) = t[5]

Spec == Init /\ [][Next]_vars
=============================================================================
