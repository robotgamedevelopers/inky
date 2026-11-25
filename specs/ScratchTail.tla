---------------------------- MODULE ScratchTail ----------------------------
\* The sliding-window K/V scratch of one multi-span prefill step: its row
\* layout and who last wrote each row (src/engine.rs layer_b; k_zero, k_kvcopy,
\* k_kvprep_b and k_attn_h in src/prefill.metal). Every span of the step, in
\* every sliding-window layer, uses the same scratch. Rows are relative to the
\* row of the span's first key (EXT + KGUARD in the code): the span copies its
\* window history (min(p0, W) keys) into the rows just below 0 and writes its
\* own keys into rows 0 .. t - 1. Attention reads keys in whole tiles of AK
\* positions, from the window's first key rounded down to a tile to its last
\* key rounded up, so it reads rows on both sides that are not the span's.
\* Those keys are masked with a zero weight, and a zero weight times a NaN or
\* an infinity is a NaN: every row the span reads must be its own or zero.
\* k_zero clears the scratch once, at the start of the step.
\*
\* Below row 0 the KGUARD rows stay zero (no span writes them) and the span's
\* own history covers the rest. Above, rows t .. Reach - 1 are written by any
\* earlier span (or the previous layer's later span) with a longer t. The fix
\* zeroes them before the span reads them; a step of one span never dirties
\* them, so the code zeroes only when the step has more than one span.
\*
\* Bug = "notail" is the reversed witness (the code before the fix):
\* NoForeignRead fails.
EXTENDS Integers, Sequences

CONSTANTS AK, W, MaxT, MaxSpans, Layers, Bug
ASSUME /\ AK \in 1..8 /\ W \in 1..8 /\ MaxT \in 1..8 /\ MaxSpans \in 1..4
       /\ Layers \in 1..3 /\ Bug \in {"none", "notail"}

G == AK                                  \* KGUARD: one tile (both 64 in the code)
Min(a, b) == IF a < b THEN a ELSE b
Zero == 0                                \* a cleared row; spans are 1..n
RoundDown(x) == (x \div AK) * AK
RoundUp(x) == RoundDown(x + AK - 1)
\* the first key of the window of the span's first query, rounded down to a tile
KStart(p0) == IF p0 + 1 > W THEN RoundDown(p0 + 1 - W) ELSE 0
Lo(sp) == KStart(sp.p0) - sp.p0          \* lowest row read
Reach(sp) == RoundUp(sp.p0 + sp.t) - sp.p0   \* one past the highest row read
Rows == -(W + G) .. (MaxT + G - 1)       \* the scratch: scap = EXT + cap + 2 KGUARD
Span == [p0 : 0 .. (W + 2 * AK), t : 1 .. MaxT]

VARIABLES spans, scratch, layer, s, pc
vars == <<spans, scratch, layer, s, pc>>

Init == /\ \E n \in 1 .. MaxSpans : spans \in [1 .. n -> Span]
        /\ scratch = [r \in Rows |-> Zero]      \* k_zero at the start of the step
        /\ layer = 1 /\ s = 1 /\ pc = "write"

\* span s, this layer: history copy (k_kvcopy), its keys (k_kvprep_b), and the fix
\* (k_kvcopy with to_ring = 2 over rows t .. Reach - 1, when the step has several spans)
Write == /\ pc = "write"
         /\ LET sp == spans[s] IN
            scratch' = [r \in Rows |->
                IF r >= -Min(sp.p0, W) /\ r < sp.t THEN s
                ELSE IF Bug = "none" /\ Len(spans) > 1 /\ r >= sp.t /\ r < Reach(sp) THEN Zero
                ELSE scratch[r]]
         /\ pc' = "read"
         /\ UNCHANGED <<spans, layer, s>>

\* span s's attention (k_attn_h); then the next span, or the next layer's first
Read == /\ pc = "read"
        /\ IF s < Len(spans) THEN s' = s + 1 /\ layer' = layer
                             ELSE s' = 1 /\ layer' = layer + 1
        /\ pc' = IF s = Len(spans) /\ layer = Layers THEN "done" ELSE "write"
        /\ UNCHANGED <<spans, scratch>>

Done == pc = "done" /\ UNCHANGED vars

Next == Write \/ Read \/ Done
Spec == Init /\ [][Next]_vars

TypeOK == /\ scratch \in [Rows -> 0 .. MaxSpans]
          /\ pc \in {"write", "read", "done"}
\* every row the span reads lies inside the scratch (the KGUARD sizing)
InBounds == pc = "read" => (Lo(spans[s]) >= -(W + G) /\ Reach(spans[s]) - 1 <= MaxT + G - 1)
\* every row the span reads is its own or zero: no other span's key reaches it
NoForeignRead == pc = "read" =>
    \A r \in Lo(spans[s]) .. (Reach(spans[s]) - 1) : scratch[r] \in {Zero, s}
=============================================================================
