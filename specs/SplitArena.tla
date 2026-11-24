----------------------------- MODULE SplitArena -----------------------------
EXTENDS Integers, FiniteSets

\* Layout of the tensor path's attention partials (src/engine.rs:
\* fixed_split_count, split_scratch_slots, fixed_attn_batch and the batch loop
\* in layer_b; src/kernels.metal: k_attn_fixed_b and k_attn_fixed_combine),
\* with the same arithmetic at a small scale. One span (p0, t) of one layer
\* kind runs its query rows [ts, t) in batches. Each batch dispatches every
\* (row, head, split) with split < nsplit whose first key j0 lies before the
\* row's end, writing slot (row * NH + head) * nsplit + split; the combine then
\* reads the row's Count visible splits from the same slots. The arena holds
\* NH * Slots partials; anything past that is the next arena region.
CONSTANT Capacity, Bug
NH == 2            \* query heads
KpsSwa == 2        \* window split width (64 in production)
KpsFull == 4       \* full split width (256)
W == 4             \* window, EXT (1024); a multiple of KpsSwa as in production
MaxCtx == 8        \* the engine's context limit
ASSUME /\ Capacity \in {3, 4, 6}
       /\ Bug \in {"none", "edgeDropped", "rowsUnbounded"}

Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b
CeilDiv(a, b) == (a + b - 1) \div b
\* fixed_split_count(first, end, width)
SplitCount(first, end, w) == IF first = end THEN 0 ELSE (end - 1) \div w + 1 - first \div w
\* split_scratch_slots(max_ctx): Capacity stands for the 256-slot floor
Slots == Max(Capacity, CeilDiv(MaxCtx, KpsFull) + 1)

VARIABLES swa, p0, t, start, phase, arena
vars == <<swa, p0, t, start, phase, arena>>

Kps == IF swa THEN KpsSwa ELSE KpsFull
SpanEnd == p0 + t
\* fixed_attn_batch: the reserved split stride, with the edge split once the window slides
NSplit == IF swa /\ SpanEnd > W
          THEN CeilDiv(W - 1, Kps) + (IF Bug = "edgeDropped" THEN 0 ELSE 1)
          ELSE SplitCount(0, SpanEnd, Kps)
Rows == IF Bug = "rowsUnbounded" THEN Slots ELSE Slots \div NSplit
N == Min(t - start, Rows)

\* row r of the span (0-based) as the kernels see it
End(r) == p0 + r + 1
First(r) == IF swa /\ End(r) > W THEN End(r) - W ELSE 0
Count(r) == SplitCount(First(r), End(r), Kps)
Owner(i, h, sp) == <<p0 + start + i, h, First(start + i) \div Kps + sp>>
Slot(i, h, sp) == (i * NH + h) * NSplit + sp

\* k_attn_fixed_b: grid (nsplit, heads, rows); returns when j0 >= end
Dispatched == {<<i, h, sp>> \in (0..(N - 1)) \X (0..(NH - 1)) \X (0..(NSplit - 1)) :
                  (First(start + i) \div Kps + sp) * Kps < End(start + i)}
WSlot(c) == Slot(c[1], c[2], c[3])
\* k_attn_fixed_combine: the row's visible splits in increasing order
Reads == UNION {{<<i, h, s>> : s \in 0..(Count(start + i) - 1)} : <<i, h>> \in (0..(N - 1)) \X (0..(NH - 1))}

Init ==
    /\ swa \in BOOLEAN
    /\ p0 \in 0..(MaxCtx - 1)
    /\ t \in 1..(MaxCtx - p0)
    /\ start \in 0..(t - 1)             \* ts: tail trimming starts the query rows late
    /\ phase = "attn"
    /\ arena = [s \in {} |-> <<>>]

Attend ==
    /\ phase = "attn"
    /\ arena' = [s \in DOMAIN arena \cup {WSlot(c) : c \in Dispatched} |->
                    IF \E c \in Dispatched : WSlot(c) = s
                    THEN LET c == CHOOSE c \in Dispatched : WSlot(c) = s IN Owner(c[1], c[2], c[3])
                    ELSE arena[s]]
    /\ phase' = "comb"
    /\ UNCHANGED <<swa, p0, t, start>>
Combine ==
    /\ phase = "comb"
    /\ start' = start + N
    /\ phase' = IF start + N >= t THEN "done" ELSE "attn"
    /\ UNCHANGED <<swa, p0, t, arena>>
Done == phase = "done" /\ UNCHANGED vars

Next == Attend \/ Combine \/ Done
Spec == Init /\ [][Next]_vars /\ WF_vars(Attend) /\ WF_vars(Combine)

\* the loop advances: step_by(0) would panic, and an empty batch never finishes
RowsPositive == Rows >= 1
\* no partial is written past the arena (into the logits that follow it)
WritesInArena == \A s \in DOMAIN arena : s < NH * Slots
\* one dispatch never writes one slot twice
WritesDisjoint == \A c1, c2 \in Dispatched : c1 # c2 => WSlot(c1) # WSlot(c2)
\* every partial the combine reads is this batch's own, for its row, head and absolute split
CombineReadsOwn ==
    phase = "comb" =>
        \A r \in Reads : LET s == Slot(r[1], r[2], r[3]) IN
            /\ s < NH * Slots
            /\ s \in DOMAIN arena
            /\ arena[s] = Owner(r[1], r[2], r[3])
Finishes == <>(phase = "done")
=============================================================================
