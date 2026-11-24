------------------------- MODULE PerKindBlockLayout -------------------------
\* Per-kind block pool and per-lane block table model matching src/blocks.rs:
\*   - Separate pools for "full" (2 MiB stride) and "window" (10 MiB stride).
\*   - 1-based Slot indices per kind (1..FullCapacity, 1..WindowCapacity);
\*     generation counters never wrap.
\*   - Plain u64 byte offsets from base 0: (SlotBase + (slot - 1)) * stride(kind),
\*     represented losslessly in 32-bit-safe limbs
\*       [hi |-> u64 >> 32, mib |-> (u64 >> 20) % 4096, sub |-> u64 % 2^20]
\*     so TLC checks both base 0 (SlotBase = 0) and >= 4 GiB high slots
\*     (SlotBase = 2048) without signed 32-bit integer overflow.

EXTENDS Integers, FiniteSets, TLC

CONSTANTS Lanes, FullCapacity, WindowCapacity, B, W, MaxPos, MaxSpan, MaxGeneration, SlotBase, Bug

ASSUME /\ FullCapacity >= 1 /\ WindowCapacity >= 1 /\ FullCapacity + WindowCapacity >= 3
       /\ B > 0 /\ W > 0
       /\ MaxPos > B /\ MaxSpan > 0 /\ MaxGeneration > 1
       /\ SlotBase >= 0
       /\ B * 2 < 1048576
       /\ Bug \in {"none", "stale", "wrong_stride", "trunc32", "window", "span", "start"}

Kinds == {"full", "window"}
Cap(kind) == IF kind = "full" THEN FullCapacity ELSE WindowCapacity
Slots(kind) == 1..Cap(kind)
Blocks == 0..((MaxPos - 1) \div B)
Keys == Kinds \X Blocks

MiBBytes == 1048576
GiB4MiB == 4096
StrideMiB(kind) == IF kind = "full" THEN 2 ELSE 10
ZeroOffset == [hi |-> 0, mib |-> 0, sub |-> 0]
PoolIndex(s) == SlotBase + (s - 1)

Offset(kind, s) ==
    LET totalMiB == PoolIndex(s) * StrideMiB(kind)
    IN [hi |-> totalMiB \div GiB4MiB, mib |-> totalMiB % GiB4MiB, sub |-> 0]

Truncate32(o) == [hi |-> 0, mib |-> o.mib, sub |-> o.sub]

EntryOffset(kind, s) ==
    IF Bug = "wrong_stride" THEN Offset("full", s)
    ELSE IF Bug = "trunc32" THEN Truncate32(Offset(kind, s))
    ELSE Offset(kind, s)

AddBytes(o, n) ==
    LET totalSub == o.sub + n
        carryMiB == o.mib + (totalSub \div MiBBytes)
    IN [hi  |-> o.hi + (carryMiB \div GiB4MiB),
        mib |-> carryMiB % GiB4MiB,
        sub |-> totalSub % MiBBytes]

NoEntry == [slot |-> 0, generation |-> 0, offset |-> ZeroOffset, kind |-> "none"]
EmptyTable == [k \in Keys |-> NoEntry]

Ceil(n, d) == (n + d - 1) \div d
SpanBound(c) == Ceil(W - 1 + c, B) + 1
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div B : p \in lo..hi}
History(n) == Window(IF n > W THEN n - W ELSE 0, n - 1)
SpanWindow(n, c) == Window(IF n >= W THEN n - W + 1 ELSE 0, n + c - 1)
Full(n) == Window(0, n - 1)
Needed(n, c) == ({"full"} \X Full(n + c)) \cup ({"window"} \X SpanWindow(n, c))

Handle(e) == <<e.kind, e.slot, e.generation>>
Handles(t) == {Handle(t[k]) : k \in {j \in Keys : t[j] # NoEntry}}
Restrict(t, ks) == [k \in Keys |-> IF k \in ks THEN t[k] ELSE NoEntry]

VARIABLES pool, table, pos, phase, target, spanStart, budget, pin,
          startKept, startPos, startTable, startState
vars == <<pool, table, pos, phase, target, spanStart, budget, pin,
          startKept, startPos, startTable, startState>>

Init ==
    /\ pool = [kind \in Kinds |-> [s \in Slots(kind) |->
                  [generation |-> 0, kind |-> kind, block |-> 0, length |-> 0,
                   data |-> [i \in 0..(B - 1) |-> 0]]]]
    /\ table = [l \in Lanes |-> EmptyTable]
    /\ pos = [l \in Lanes |-> 0]
    /\ phase = [l \in Lanes |-> "idle"]
    /\ target = [l \in Lanes |-> 0]
    /\ spanStart = [l \in Lanes |-> 0]
    /\ budget = [l \in Lanes |-> 0]
    /\ pin = [l \in Lanes |-> {}]
    /\ startKept = FALSE /\ startPos = 0
    /\ startTable = EmptyTable /\ startState = 0

LaneHandles == UNION {Handles(table[l]) : l \in {q \in Lanes : phase[q] # "done"}}
StepHandles == UNION {pin[l] : l \in Lanes}
StartHandles == IF startKept THEN Handles(startTable) ELSE {}
Owned == LaneHandles \cup StepHandles \cup
         (IF Bug = "start" THEN {h \in StartHandles : h[1] = "full"} ELSE StartHandles)

Free(kind, s) == ~\E h \in Owned : h[1] = kind /\ h[2] = s

Resolves(kind, e) ==
    /\ e # NoEntry
    /\ e.kind = kind
    /\ e.slot \in Slots(kind)
    /\ (e.slot \in Slots(kind) =>
            /\ e.generation = pool[kind][e.slot].generation
            /\ pool[kind][e.slot].kind = kind)

Prepare(l, c) ==
    /\ phase[l] = "idle" /\ c \in 1..MaxSpan /\ pos[l] + c <= MaxPos
    /\ phase' = [phase EXCEPT ![l] = "prepare"]
    /\ target' = [target EXCEPT ![l] = pos[l] + c]
    /\ spanStart' = [spanStart EXCEPT ![l] = pos[l]]
    /\ budget' = [budget EXCEPT ![l] = IF Bug = "span" THEN Ceil(W, B) + 1 ELSE SpanBound(c)]
    /\ UNCHANGED <<pool, table, pos, pin, startKept, startPos, startTable, startState>>

Alloc(l, k, s) ==
    /\ phase[l] = "prepare"
    /\ k \in Needed(spanStart[l], target[l] - spanStart[l])
    /\ table[l][k] = NoEntry
    /\ LET kind == k[1] IN
       /\ s \in Slots(kind) /\ Free(kind, s)
       /\ pool[kind][s].generation < MaxGeneration
       /\ \A r \in 1..(s - 1) : ~Free(kind, r) \/ pool[kind][r].generation = MaxGeneration
       /\ LET g == pool[kind][s].generation + 1 IN
          /\ pool' = [pool EXCEPT ![kind][s] =
                [generation |-> g, kind |-> kind, block |-> k[2], length |-> 0,
                 data |-> [i \in 0..(B - 1) |-> 0]]]
          /\ table' = [table EXCEPT ![l][k] =
                [slot |-> s, generation |-> g, offset |-> EntryOffset(kind, s), kind |-> kind]]
    /\ UNCHANGED <<pos, phase, target, spanStart, budget, pin,
                   startKept, startPos, startTable, startState>>

BeginStep(l) ==
    /\ phase[l] = "prepare"
    /\ \A k \in Needed(spanStart[l], target[l] - spanStart[l]) : table[l][k] # NoEntry
    /\ phase' = [phase EXCEPT ![l] = "step"]
    /\ pin' = [pin EXCEPT ![l] = Handles(Restrict(table[l], Needed(spanStart[l], target[l] - spanStart[l])))]
    /\ UNCHANGED <<pool, table, pos, target, spanStart, budget,
                   startKept, startPos, startTable, startState>>

Write(l) ==
    /\ phase[l] = "step" /\ pos[l] < target[l]
    /\ LET b == pos[l] \div B
           off == pos[l] % B
       IN pool' = [kind \in Kinds |-> [s \in Slots(kind) |->
              IF table[l][<<kind, b>>].slot = s
              THEN [pool[kind][s] EXCEPT !.length = @ + 1, !.data[off] = 1]
              ELSE pool[kind][s]]]
    /\ pos' = [pos EXCEPT ![l] = @ + 1]
    /\ UNCHANGED <<table, phase, target, spanStart, budget, pin,
                   startKept, startPos, startTable, startState>>

EndStep(l) ==
    /\ phase[l] = "step" /\ pos[l] = target[l]
    /\ phase' = [phase EXCEPT ![l] = "idle"]
    /\ pin' = [pin EXCEPT ![l] = {}]
    /\ UNCHANGED <<pool, table, pos, target, spanStart, budget,
                   startKept, startPos, startTable, startState>>

SlideWindow(l) ==
    /\ phase[l] = "idle"
    /\ LET keep == IF Bug = "window" /\ pos[l] > W
                   THEN History(pos[l]) \ {(pos[l] - W) \div B}
                   ELSE History(pos[l])
           t == Restrict(table[l], ({"full"} \X Blocks) \cup ({"window"} \X keep))
       IN /\ table[l] # t /\ table' = [table EXCEPT ![l] = t]
    /\ UNCHANGED <<pool, pos, phase, target, spanStart, budget, pin,
                   startKept, startPos, startTable, startState>>

KeepStart(l) ==
    /\ phase[l] = "idle" /\ pos[l] > 0 /\ pos[l] % B = 0 /\ ~startKept
    /\ startKept' = TRUE /\ startPos' = pos[l] /\ startState' = pos[l]
    /\ startTable' = Restrict(table[l], ({"full"} \X Full(pos[l])) \cup ({"window"} \X History(pos[l])))
    /\ UNCHANGED <<pool, table, pos, phase, target, spanStart, budget, pin>>

DropStart ==
    /\ startKept /\ startKept' = FALSE /\ startTable' = EmptyTable
    /\ UNCHANGED <<pool, table, pos, phase, target, spanStart, budget, pin, startPos, startState>>

Match(l) ==
    /\ startKept /\ phase[l] = "idle" /\ pos[l] = 0
    /\ table' = [table EXCEPT ![l] = startTable]
    /\ pos' = [pos EXCEPT ![l] = startPos]
    /\ UNCHANGED <<pool, phase, target, spanStart, budget, pin,
                   startKept, startPos, startTable, startState>>

Finish(l) ==
    /\ phase[l] = "idle"
    /\ phase' = [phase EXCEPT ![l] = "done"]
    /\ table' = [table EXCEPT ![l] = IF Bug = "stale" THEN @ ELSE EmptyTable]
    /\ UNCHANGED <<pool, pos, target, spanStart, budget, pin,
                   startKept, startPos, startTable, startState>>

Next ==
    \/ \E l \in Lanes :
        \/ \E c \in 1..MaxSpan : Prepare(l, c)
        \/ \E k \in Keys : \E s \in Slots(k[1]) : Alloc(l, k, s)
        \/ BeginStep(l) \/ Write(l) \/ EndStep(l) \/ SlideWindow(l)
        \/ KeepStart(l) \/ Match(l) \/ Finish(l)
    \/ DropStart

TypeOK ==
    /\ pos \in [Lanes -> 0..MaxPos]
    /\ phase \in [Lanes -> {"idle", "prepare", "step", "done"}]
    /\ \A kind \in Kinds : \A s \in Slots(kind) :
        /\ pool[kind][s].generation \in 0..MaxGeneration
        /\ pool[kind][s].kind = kind
        /\ pool[kind][s].block \in Blocks
        /\ pool[kind][s].length \in 0..B
        /\ pool[kind][s].data \in [0..(B - 1) -> {0, 1}]

GenerationMatches ==
    \A l \in Lanes, k \in Keys :
        table[l][k] # NoEntry => Resolves(k[1], table[l][k])

WindowCovered ==
    \A l \in Lanes : phase[l] # "done" =>
        \A b \in History(pos[l]) : table[l][<<"window", b>>] # NoEntry

FullCovered ==
    \A l \in Lanes : phase[l] # "done" =>
        \A b \in Full(pos[l]) : table[l][<<"full", b>>] # NoEntry

SpanPinned ==
    \A l \in Lanes : phase[l] = "step" =>
        /\ budget[l] >= SpanBound(target[l] - spanStart[l])
        /\ Handles(Restrict(table[l], Needed(spanStart[l], target[l] - spanStart[l]))) \subseteq pin[l]
        /\ \A h \in pin[l] :
            /\ h[1] \in Kinds /\ h[2] \in Slots(h[1])
            /\ ((h[1] \in Kinds /\ h[2] \in Slots(h[1])) => h[3] = pool[h[1]][h[2]].generation)

StartRetained ==
    startKept =>
        /\ startState = startPos
        /\ \A k \in ({"full"} \X Full(startPos)) \cup ({"window"} \X History(startPos)) :
            /\ Resolves(k[1], startTable[k])
            /\ (startTable[k].slot \in Slots(k[1]) => pool[k[1]][startTable[k].slot].length = B)

TailPrivateZero ==
    \A kind \in Kinds : \A s \in Slots(kind) : pool[kind][s].length < B =>
        /\ Cardinality({l \in Lanes : \E b \in Blocks :
                table[l][<<kind, b>>] # NoEntry /\ table[l][<<kind, b>>].slot = s
                /\ phase[l] # "done"}) <= 1
        /\ \A i \in pool[kind][s].length..(B - 1) : pool[kind][s].data[i] = 0

TableIndex ==
    \A l \in Lanes, k \in Keys : table[l][k] # NoEntry =>
        /\ table[l][k].slot \in Slots(k[1])
        /\ (table[l][k].slot \in Slots(k[1]) => pool[k[1]][table[l][k].slot].block = k[2])

ValidOffset(kind, e) ==
    /\ e.slot \in Slots(kind)
    /\ e.offset.hi >= 0
    /\ e.offset.mib \in 0..(GiB4MiB - 1)
    /\ e.offset.sub \in 0..(MiBBytes - 1)
    /\ (e.slot \in Slots(kind) =>
            /\ e.offset = Offset(kind, e.slot)
            /\ \A i \in 0..(B - 1) :
                AddBytes(e.offset, i * 2) = AddBytes(Offset(kind, e.slot), i * 2))

Address64 ==
    /\ \A l \in Lanes, k \in Keys :
        table[l][k] # NoEntry => ValidOffset(k[1], table[l][k])
    /\ startKept => \A k \in Keys :
        startTable[k] # NoEntry => ValidOffset(k[1], startTable[k])

Spec == Init /\ [][Next]_vars

=============================================================================
