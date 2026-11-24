------------------------------ MODULE InBlockTrace -----------------------
\* The behaviour key_row_at emits at these bounds. InBlockTraceData is
\* written by `inky inblock-trace`. Next follows those addresses. Ordered
\* compares each one to Formula, which Next does not call.
EXTENDS Integers, Sequences, InBlockTraceData

CONSTANTS Kvb
ASSUME Kvb \in {1, 2}

Passes == 2
Heads == 3
B == 4
HD == 2

Formula(p, k, h, i) ==
  ((p * 2 + k) * Heads + h) * B * HD * Kvb + i * HD * Kvb

Trace == IF Kvb = 2 THEN Trace2 ELSE Trace1

VARIABLES n, p, k, h, i, addr
vars == <<n, p, k, h, i, addr>>

Init ==
  /\ n = 1
  /\ p = Trace[1][1] /\ k = Trace[1][2] /\ h = Trace[1][3] /\ i = Trace[1][4]
  /\ addr = Trace[1][5]

Next ==
  \/ /\ n < Len(Trace)
     /\ n' = n + 1
     /\ p' = Trace[n'][1] /\ k' = Trace[n'][2] /\ h' = Trace[n'][3] /\ i' = Trace[n'][4]
     /\ addr' = Trace[n'][5]
     /\ <<p', k', h', i'>> # <<p, k, h, i>>
  \/ /\ n = Len(Trace)
     /\ UNCHANGED vars

Ordered == addr = Formula(p, k, h, i)

Spec == Init /\ [][Next]_vars
=============================================================================
