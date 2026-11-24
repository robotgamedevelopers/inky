----------------------------- MODULE StartTails -----------------------------
\* Start points and unsealed tails (docs/DESIGN.md, Blocks), at capacity 3 or 4
\* slots per kind. Bound to src/blockbook.rs: scripts/check-blockbook-traces.sh
\* replays deterministic random walks of that file through StartTailsTrace.tla,
\* so every recorded call must be a step of this module.
\*
\* Vocabulary follows BlockLayout.tla (Kinds, Window, History, Full) and the
\* start-point record of CacheScheduler.tla (state, full and window blocks of a
\* boundary). As in src/blocks.rs, each kind has its own pool of 1-based slots.
\* The step in flight, its pins, tiers and budgets belong to BlockLayout and
\* CacheScheduler; this module owns which blocks a start point holds, when a
\* start point is kept and dropped, what a lane may read of its tail, and
\* zeroing on reuse.
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Lanes, Capacity, B, W, Every, MaxPos, MaxSpan, MaxStarts,
          MaxGeneration, Bug
ASSUME /\ Lanes \subseteq Nat /\ Capacity >= 2 /\ B > 0 /\ W > 0 /\ Every > 0
       /\ MaxPos > B /\ MaxSpan > 0 /\ MaxStarts > 0 /\ MaxGeneration > 1
       /\ Bug \in {"none", "unheld", "prune", "overread", "nozero"}

Kinds == {"full", "window"}
Slots == 1..Capacity
Blocks == 0..((MaxPos - 1) \div B)
Keys == Kinds \X Blocks
StartIds == 1..MaxStarts
Reasons == {"prompt", "every", "head"}
NoLane == -1
NoEntry == [slot |-> 0, generation |-> 0]
EmptyTable == [k \in Keys |-> NoEntry]
Zeros == [i \in 0..(B - 1) |-> 0]
Fresh == [generation |-> 0, length |-> 0, sealed |-> FALSE, data |-> Zeros]
Dead == [live |-> FALSE, pos |-> 0, state |-> 0, table |-> EmptyTable,
         reasons |-> {}, lane |-> NoLane]

Lesser(x, y) == IF x < y THEN x ELSE y
Greater(x, y) == IF x > y THEN x ELSE y
Smallest(S) == CHOOSE m \in S : \A y \in S : m <= y
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div B : p \in lo..hi}
History(n) == Window(IF n > W THEN n - W ELSE 0, n - 1)
Full(n) == Window(0, n - 1)
Floor(n) == (n \div B) * B
\* What a start point at boundary n keeps: every full block before n and the
\* window blocks covering the W positions before n.
Retained(n) == ({"full"} \X Full(n)) \cup ({"window"} \X History(n))
\* What a lane at n keeps after its step: all full blocks, its window history,
\* and window blocks it allocated ahead.
LaneKeeps(n) == ({"full"} \X Blocks)
                \cup ({"window"} \X (History(n) \cup {b \in Blocks : b * B >= n}))
Restrict(t, ks) == [k \in Keys |-> IF k \in ks THEN t[k] ELSE NoEntry]
Entries(t) == {k \in Keys : t[k] # NoEntry}
Holds(t, kd, x) == \E k \in Entries(t) : k[1] = kd /\ t[k].slot = x
\* Positions of block b a lane at position n has written.
Seen(n, b) == Lesser(B, Greater(0, n - b * B))
\* What attention reads of block b, masked to the lane's position. The
\* "overread" defect reads one position past it.
Visible(n, b) == IF Bug = "overread" THEN Lesser(B, Seen(n, b) + 1) ELSE Seen(n, b)

\* freed is a ghost for the "prune" defect only: slots pruning released
\* outright although a start point holds them. It stays empty otherwise.
VARIABLES pool, table, pos, prompt, running, starts, head, freed
vars == <<pool, table, pos, prompt, running, starts, head, freed>>

Init ==
 /\ pool = [kd \in Kinds |-> [x \in Slots |-> Fresh]]
 /\ table = [l \in Lanes |-> EmptyTable]
 /\ pos = [l \in Lanes |-> 0]
 /\ prompt = [l \in Lanes |-> 0]
 /\ running = [l \in Lanes |-> FALSE]
 /\ starts = [i \in StartIds |-> Dead]
 /\ head = [l \in Lanes |-> 0]
 /\ freed = {}

LaneRefs(kd, x) == {l \in Lanes : Holds(table[l], kd, x)}
StartRefs(kd, x) == {i \in StartIds : starts[i].live /\ Holds(starts[i].table, kd, x)}
Referenced(kd, x) == LaneRefs(kd, x) # {} \/ StartRefs(kd, x) # {}
\* The allocator's view. The "unheld" defect is a start point that does not hold
\* its window blocks; the "prune" defect is pruning that releases a window slot
\* outright, under a start point that holds it.
StartHolds(i, kd, x) == /\ starts[i].live /\ Holds(starts[i].table, kd, x)
                        /\ (Bug = "unheld" => kd = "full")
Usable(kd, x) == /\ LaneRefs(kd, x) = {}
                 /\ (<<kd, x>> \in freed \/ \A i \in StartIds : ~StartHolds(i, kd, x))
                 /\ pool[kd][x].generation < MaxGeneration
\* Drop the head reason from start point h; a point with no reason left dies.
DropHead(st, h) ==
    IF h = 0 THEN st
    ELSE LET left == st[h].reasons \ {"head"}
         IN [st EXCEPT ![h] = IF left = {} THEN Dead
                              ELSE [st[h] EXCEPT !.reasons = left, !.lane = NoLane]]

\* A request takes lane l with a prompt of p positions, either cold or resumed
\* from a live start point before the prompt's end.
Begin(l, p, i) ==
 /\ ~running[l] /\ p \in 1..MaxPos
 /\ IF i = 0 THEN UNCHANGED <<table, pos>>
    ELSE /\ starts[i].live /\ starts[i].pos < p
         /\ table' = [table EXCEPT ![l] = starts[i].table]
         /\ pos' = [pos EXCEPT ![l] = starts[i].pos]
 /\ running' = [running EXCEPT ![l] = TRUE]
 /\ prompt' = [prompt EXCEPT ![l] = p]
 /\ UNCHANGED <<pool, starts, head, freed>>

\* Tail blocks for the lane's next span: block b of both kinds at once, each
\* in the lowest usable slot of its kind's pool, zeroed, at the next generation.
\* Neither is taken unless both can be.
Taken(l, b, kd) ==
    LET x == Smallest({y \in Slots : Usable(kd, y)})
    IN [slot |-> x, generation |-> pool[kd][x].generation + 1]
Alloc(l, b) ==
 /\ running[l] /\ \A kd \in Kinds : table[l][<<kd, b>>] = NoEntry
 /\ b \in Window(pos[l], Lesser(MaxPos, pos[l] + MaxSpan) - 1)
 /\ \A kd \in Kinds : \E x \in Slots : Usable(kd, x)
 /\ pool' = [kd \in Kinds |-> [x \in Slots |->
       IF x = Taken(l, b, kd).slot
       THEN [generation |-> Taken(l, b, kd).generation, length |-> 0, sealed |-> FALSE,
             data |-> IF Bug = "nozero" THEN pool[kd][x].data ELSE Zeros]
       ELSE pool[kd][x]]]
 /\ table' = [table EXCEPT ![l] = [k \in Keys |->
       IF k[2] = b THEN Taken(l, b, k[1]) ELSE table[l][k]]]
 /\ freed' = freed \ {<<kd, Taken(l, b, kd).slot>> : kd \in Kinds}
 /\ UNCHANGED <<pos, prompt, running, starts, head>>

\* Write c positions of lane l's tails, both kinds, as one step.
Written(l, c, kd, x) ==
    {p \in pos[l]..(pos[l] + c - 1) : table[l][<<kd, p \div B>>].slot = x}
Write(l, c) ==
 [kd \in Kinds |-> [x \in Slots |->
    LET mine == Written(l, c, kd, x)
        rec == pool[kd][x]
    IN IF mine = {} THEN rec
       ELSE LET len == ((CHOOSE m \in mine : \A y \in mine : y <= m) % B) + 1
            IN [rec EXCEPT !.length = len, !.sealed = (len = B),
                           !.data = [i \in 0..(B - 1) |->
                                       IF \E p \in mine : p % B = i THEN 1 ELSE rec.data[i]]]]]

\* One step of lane l: c positions written through its table, its window pruned
\* to History, and, when the step ends a prefill on a block boundary, a start
\* point kept. The point's reasons are the lane head, the prompt's last
\* boundary and every Every-th block. The lane's previous head loses its head
\* reason first, so its id can be the new point's. With no free id the step
\* keeps no point.
Step(l, c) ==
 /\ running[l] /\ c \in 1..MaxSpan /\ pos[l] + c <= MaxPos
 /\ LET n == pos[l] + c
        bs == Window(pos[l], n - 1)
        t == Restrict(table[l], LaneKeeps(n))
        keep == /\ n % B = 0 /\ n <= Floor(prompt[l])
                /\ IF head[l] = 0 THEN TRUE ELSE starts[head[l]].pos < n
        dropped == DropHead(starts, head[l])
        free == {j \in StartIds : ~dropped[j].live}
        rs == {"head"} \cup (IF n = Floor(prompt[l]) THEN {"prompt"} ELSE {})
                       \cup (IF (n \div B) % Every = 0 THEN {"every"} ELSE {})
    IN /\ \A k \in Kinds \X bs : table[l][k] # NoEntry
       /\ \A k \in Kinds \X bs :
             pool[k[1]][table[l][k].slot].length = Greater(pos[l], k[2] * B) - k[2] * B
       /\ pool' = Write(l, c)
       /\ pos' = [pos EXCEPT ![l] = n]
       /\ table' = [table EXCEPT ![l] = t]
       /\ freed' = IF Bug = "prune"
                   THEN freed \cup {<<"window", table[l][k].slot>> : k \in Entries(table[l]) \ Entries(t)}
                   ELSE freed
       /\ IF keep /\ free # {}
          THEN LET i == Smallest(free)
               IN /\ starts' = [dropped EXCEPT ![i] =
                        [live |-> TRUE, pos |-> n, state |-> n,
                         table |-> Restrict(t, Retained(n)), reasons |-> rs, lane |-> l]]
                  /\ head' = [head EXCEPT ![l] = i]
          ELSE UNCHANGED <<starts, head>>
 /\ UNCHANGED <<prompt, running>>

\* The request ends: its tails are dropped with the lane's table.
Finish(l) ==
 /\ running[l]
 /\ running' = [running EXCEPT ![l] = FALSE]
 /\ table' = [table EXCEPT ![l] = EmptyTable]
 /\ pos' = [pos EXCEPT ![l] = 0]
 /\ prompt' = [prompt EXCEPT ![l] = 0]
 /\ starts' = DropHead(starts, head[l])
 /\ head' = [head EXCEPT ![l] = 0]
 /\ UNCHANGED <<pool, freed>>

\* Memory pressure retires a start point (the scheduler chooses which).
Retire(i) ==
 /\ starts[i].live
 /\ starts' = [starts EXCEPT ![i] = Dead]
 /\ head' = [l \in Lanes |-> IF head[l] = i THEN 0 ELSE head[l]]
 /\ UNCHANGED <<pool, table, pos, prompt, running, freed>>

Next ==
 \/ \E l \in Lanes :
      \/ \E p \in 1..MaxPos, i \in 0..MaxStarts : Begin(l, p, i)
      \/ \E b \in Blocks : Alloc(l, b)
      \/ \E c \in 1..MaxSpan : Step(l, c)
      \/ Finish(l)
 \/ \E i \in StartIds : Retire(i)

TypeOK ==
 /\ \A kd \in Kinds, x \in Slots :
      /\ pool[kd][x].generation \in 0..MaxGeneration
      /\ pool[kd][x].length \in 0..B
      /\ pool[kd][x].sealed = (pool[kd][x].length = B)
      /\ pool[kd][x].data \in [0..(B - 1) -> {0, 1}]
 /\ pos \in [Lanes -> 0..MaxPos]
 /\ prompt \in [Lanes -> 0..MaxPos] /\ running \in [Lanes -> BOOLEAN]
 /\ head \in [Lanes -> 0..MaxStarts]
 /\ \A i \in StartIds : starts[i].reasons \subseteq Reasons
 /\ Bug # "prune" => freed = {}

\* Every lane entry resolves to its slot's current generation, and each
\* block's length is exactly what the lane has written of it.
LaneCovered ==
 /\ \A l \in Lanes : ~running[l] => table[l] = EmptyTable /\ head[l] = 0
 /\ \A l \in Lanes : running[l] =>
      /\ \A k \in Retained(pos[l]) : table[l][k] # NoEntry
      /\ \A k \in Entries(table[l]) :
           LET rec == pool[k[1]][table[l][k].slot] IN
           /\ table[l][k].generation = rec.generation
           /\ rec.length = Seen(pos[l], k[2])

\* (a) A live start point still holds its state, its full blocks and the window
\* blocks before it, all sealed, at the generation it kept.
StartRetained == \A i \in StartIds : starts[i].live =>
 /\ starts[i].state = starts[i].pos
 /\ starts[i].pos % B = 0 /\ starts[i].pos > 0
 /\ Entries(starts[i].table) = Retained(starts[i].pos)
 /\ \A k \in Retained(starts[i].pos) :
      LET e == starts[i].table[k] IN
      /\ e.generation = pool[k[1]][e.slot].generation
      /\ pool[k[1]][e.slot].sealed

\* (b) Every read attention can make through a running lane's table sees only
\* positions the block's writer wrote. The trace check compares each read the
\* engine reports with Visible.
ReadsWritten == \A l \in Lanes : running[l] => \A k \in Entries(table[l]) :
 LET rec == pool[k[1]][table[l][k].slot]
     n == Visible(pos[l], k[2])
 IN /\ n <= rec.length
    /\ \A i \in 0..(n - 1) : rec.data[i] = 1

\* (c) An unsealed tail has one lane, no start point, and zeros past its length.
TailPrivateZero == \A kd \in Kinds, x \in Slots :
 Referenced(kd, x) /\ ~pool[kd][x].sealed =>
 /\ Cardinality(LaneRefs(kd, x)) = 1 /\ StartRefs(kd, x) = {}
 /\ \A i \in pool[kd][x].length..(B - 1) : pool[kd][x].data[i] = 0

\* Heads: one per running lane at most, and a live point always has a reason.
HeadsConsistent ==
 /\ \A l \in Lanes : head[l] # 0 =>
      /\ running[l] /\ starts[head[l]].live
      /\ "head" \in starts[head[l]].reasons /\ starts[head[l]].lane = l
 /\ \A i \in StartIds : starts[i].live =>
      /\ starts[i].reasons # {}
      /\ ("head" \in starts[i].reasons) =
           (starts[i].lane # NoLane /\ head[starts[i].lane] = i)

Spec == Init /\ [][Next]_vars
=============================================================================
