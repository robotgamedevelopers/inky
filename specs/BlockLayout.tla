---------------------------- MODULE BlockLayout ----------------------------
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Lanes, FullCapacity, WindowCapacity, B, W, MaxPos, MaxSpan, MaxGeneration, SlotBase, Kvb, Bug
ASSUME /\ FullCapacity >= 1 /\ WindowCapacity >= 1 /\ FullCapacity + WindowCapacity >= 3
       /\ B >= 2 /\ W > 0 /\ Kvb \in {1, 2}
       /\ MaxPos > B /\ MaxSpan > 0 /\ MaxGeneration > 1 /\ SlotBase >= 0
       /\ Bug \in {"none", "stale", "window", "span", "start", "trunc32"}

Kinds == {"full", "window"}
Cap(kind) == IF kind = "full" THEN FullCapacity ELSE WindowCapacity
Slots(kind) == 1..Cap(kind)
Blocks == 0..((MaxPos - 1) \div B)
Keys == Kinds \X Blocks

\* Each kind owns a distinct pool starting at byte offset 0:
\*   byte_offset(kind, slot, kvb) = (SlotBase + (slot - 1)) * stride_bytes(kind, kvb)
\* where at kvb = 2 (fp16) full is 2 MiB and window is 10 MiB, and at kvb = 1 (--kv8)
\* full is 1 MiB and window is 5 MiB (src/blocks.rs).
MiBBytes == 1048576
GiB4MiB == 4096
StrideMiB(kind) == (IF kind = "full" THEN 1 ELSE 5) * Kvb
ZeroOffset == [hi |-> 0, mib |-> 0, sub |-> 0]
PoolIndex(s) == SlotBase + (s - 1)
Offset(kind, s) ==
  LET totalMiB == PoolIndex(s) * StrideMiB(kind)
  IN [hi |-> totalMiB \div GiB4MiB, mib |-> totalMiB % GiB4MiB, sub |-> 0]
Truncate32(o) == [hi |-> 0, mib |-> o.mib, sub |-> o.sub]
EntryOffset(kind, s) == IF Bug = "trunc32" THEN Truncate32(Offset(kind, s)) ELSE Offset(kind, s)
AddBytes(o, n) ==
  LET totalSub == o.sub + n
      carryMiB == o.mib + (totalSub \div MiBBytes)
  IN [hi  |-> o.hi + (carryMiB \div GiB4MiB),
      mib |-> carryMiB % GiB4MiB,
      sub |-> totalSub % MiBBytes]
OffsetLt(a, b) ==
  \/ a.hi < b.hi
  \/ (a.hi = b.hi /\ a.mib < b.mib)
  \/ (a.hi = b.hi /\ a.mib = b.mib /\ a.sub < b.sub)
OffsetLe(a, b) == a = b \/ OffsetLt(a, b)
AddMiB(o, m) ==
  LET totalMiB == o.mib + m
  IN [hi |-> o.hi + (totalMiB \div GiB4MiB), mib |-> totalMiB % GiB4MiB, sub |-> o.sub]
PoolEnd(kind) ==
  LET endMiB == (SlotBase + Cap(kind)) * StrideMiB(kind)
  IN [hi |-> endMiB \div GiB4MiB, mib |-> endMiB % GiB4MiB, sub |-> 0]

NoEntry == [slot |-> 0, generation |-> 0, offset |-> ZeroOffset, kind |-> "none", kvb |-> 0]
EmptyTable == [k \in Keys |-> NoEntry]

Ceil(n, d) == (n + d - 1) \div d
SpanBound(c) == Ceil(W - 1 + c, B) + 1
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div B : p \in lo..hi}
History(n) == Window(IF n > W THEN n - W ELSE 0, n - 1)
SpanWindow(n, c) == Window(IF n >= W THEN n - W + 1 ELSE 0, n + c - 1)
Full(n) == Window(0, n - 1)
Needed(n, c) == ({"full"} \X Full(n + c))
                 \cup ({"window"} \X SpanWindow(n, c))
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
              data |-> [i \in 0..(B-1) |-> 0]]]]
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
Owned == LaneHandles \cup StepHandles \cup (IF Bug = "start" THEN {h \in StartHandles : h[1] = "full"} ELSE StartHandles)
Free(kind, s) == ~\E h \in Owned : h[1] = kind /\ h[2] = s
Resolves(kind, e) == /\ e # NoEntry
                     /\ e.kind = kind
                     /\ e.kvb = Kvb
                     /\ e.slot \in Slots(kind)
                     /\ e.generation > 0
                     /\ e.generation = pool[kind][e.slot].generation
                     /\ pool[kind][e.slot].kind = kind

Prepare(l, c) ==
 /\ phase[l] = "idle" /\ c \in 1..MaxSpan /\ pos[l] + c <= MaxPos
 /\ phase' = [phase EXCEPT ![l] = "prepare"]
 /\ target' = [target EXCEPT ![l] = pos[l] + c]
 /\ spanStart' = [spanStart EXCEPT ![l] = pos[l]]
 /\ budget' = [budget EXCEPT ![l] = IF Bug = "span" THEN Ceil(W, B)+1 ELSE SpanBound(c)]
 /\ UNCHANGED <<pool, table, pos, pin, startKept, startPos, startTable, startState>>

Alloc(l, k, s) ==
 /\ phase[l] = "prepare"
 /\ k \in Needed(spanStart[l], target[l]-spanStart[l])
 /\ table[l][k] = NoEntry /\ s \in Slots(k[1]) /\ Free(k[1], s)
 /\ pool[k[1]][s].generation < MaxGeneration
 /\ \A r \in 1..(s-1) : ~Free(k[1], r) \/ pool[k[1]][r].generation = MaxGeneration
 /\ LET kind == k[1]
        g == pool[kind][s].generation + 1
    IN /\ pool' = [pool EXCEPT ![kind][s] =
            [generation |-> g, kind |-> kind, block |-> k[2], length |-> 0, data |-> [i \in 0..(B-1) |-> 0]]]
       /\ table' = [table EXCEPT ![l][k] =
            [slot |-> s, generation |-> g, offset |-> EntryOffset(kind, s), kind |-> kind, kvb |-> Kvb]]
 /\ UNCHANGED <<pos, phase, target, spanStart, budget, pin,
                 startKept, startPos, startTable, startState>>

BeginStep(l) ==
 /\ phase[l] = "prepare"
 /\ \A k \in Needed(spanStart[l], target[l]-spanStart[l]) : table[l][k] # NoEntry
 /\ phase' = [phase EXCEPT ![l] = "step"]
 /\ pin' = [pin EXCEPT ![l] = Handles(Restrict(table[l], Needed(spanStart[l], target[l]-spanStart[l])))]
 /\ UNCHANGED <<pool, table, pos, target, spanStart, budget,
                 startKept, startPos, startTable, startState>>

Write(l) ==
 /\ phase[l] = "step" /\ pos[l] < target[l]
 /\ LET b == pos[l] \div B
    IN pool' = [kind \in Kinds |->
         LET s == table[l][<<kind, b>>].slot
         IN [pool[kind] EXCEPT ![s].length = @ + 1, ![s].data[pos[l] % B] = 1]]
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
                THEN History(pos[l]) \ { (pos[l]-W) \div B }
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

Next == (\E l \in Lanes :
          (\E c \in 1..MaxSpan : Prepare(l,c))
          \/ (\E k \in Keys : \E s \in Slots(k[1]) : Alloc(l,k,s))
          \/ BeginStep(l) \/ Write(l) \/ EndStep(l) \/ SlideWindow(l)
          \/ KeepStart(l) \/ Match(l) \/ Finish(l)) \/ DropStart

TypeOK ==
 /\ pos \in [Lanes -> 0..MaxPos]
 /\ phase \in [Lanes -> {"idle", "prepare", "step", "done"}]
 /\ \A kind \in Kinds : \A s \in Slots(kind) :
      /\ pool[kind][s].generation \in 0..MaxGeneration
      /\ pool[kind][s].kind = kind
      /\ pool[kind][s].length \in 0..B
      /\ pool[kind][s].data \in [0..(B-1) -> {0,1}]
GenerationMatches == \A l \in Lanes, k \in Keys :
 table[l][k] # NoEntry => Resolves(k[1], table[l][k])
WindowCovered == \A l \in Lanes : phase[l] # "done" =>
 \A b \in History(pos[l]) : table[l][<<"window",b>>] # NoEntry
FullCovered == \A l \in Lanes : phase[l] # "done" =>
 \A b \in Full(pos[l]) : table[l][<<"full",b>>] # NoEntry
SpanPinned == \A l \in Lanes : phase[l] = "step" =>
 /\ budget[l] >= SpanBound(target[l]-spanStart[l])
 /\ Handles(Restrict(table[l], Needed(spanStart[l],target[l]-spanStart[l]))) \subseteq pin[l]
 /\ \A h \in pin[l] : h[3] = pool[h[1]][h[2]].generation
StartRetained == startKept =>
 /\ startState = startPos
 /\ \A k \in ({"full"} \X Full(startPos)) \cup ({"window"} \X History(startPos)) :
      Resolves(k[1], startTable[k]) /\ pool[k[1]][startTable[k].slot].length = B
TailPrivateZero == \A kind \in Kinds : \A s \in Slots(kind) : pool[kind][s].length < B =>
 /\ Cardinality({l \in Lanes : \E b \in Blocks :
        table[l][<<kind, b>>] # NoEntry /\ table[l][<<kind, b>>].slot = s
        /\ phase[l] # "done"}) <= 1
 /\ \A i \in pool[kind][s].length..(B-1) : pool[kind][s].data[i] = 0
TableIndex == \A l \in Lanes, k \in Keys : table[l][k] # NoEntry =>
 pool[k[1]][table[l][k].slot].block = k[2]

LiveEntries(kind) ==
  ({table[l][<<kind, b>>] : l \in {q \in Lanes : phase[q] # "done"}, b \in Blocks} \ {NoEntry})
  \cup (IF startKept THEN {startTable[<<kind, b>>] : b \in Blocks} \ {NoEntry} ELSE {})
ValidOffset(kind, e) ==
 /\ e.offset.hi >= 0
 /\ e.offset.mib \in 0..(GiB4MiB - 1)
 /\ e.offset.sub \in 0..(MiBBytes - 1)
 /\ e.offset = Offset(kind, e.slot)
 /\ (PoolIndex(e.slot) > 0 => OffsetLt(ZeroOffset, e.offset))
 /\ (PoolIndex(e.slot) * StrideMiB(kind) >= GiB4MiB => e.offset.hi >= 1)
 /\ OffsetLe(AddMiB(e.offset, StrideMiB(kind)), PoolEnd(kind))
 /\ OffsetLt(AddBytes(e.offset, (B - 1) * Kvb), AddMiB(e.offset, StrideMiB(kind)))
 /\ AddBytes(e.offset, 0) = e.offset
 /\ \A i \in 0..(B - 2) :
      OffsetLt(AddBytes(e.offset, i * Kvb), AddBytes(e.offset, (i + 1) * Kvb))
 /\ \A i \in 0..(B - 1) :
      /\ AddBytes(e.offset, i * Kvb).sub = (e.offset.sub + i * Kvb) % MiBBytes
      /\ AddBytes(e.offset, i * Kvb) = AddBytes(Offset(kind, e.slot), i * Kvb)
Address64 ==
 /\ \A l \in Lanes, k \in Keys :
      table[l][k] # NoEntry => ValidOffset(k[1], table[l][k])
 /\ startKept => \A k \in Keys :
      startTable[k] # NoEntry => ValidOffset(k[1], startTable[k])
 /\ \A kind \in Kinds : \A e1, e2 \in LiveEntries(kind) :
      e1.slot < e2.slot => OffsetLe(AddMiB(e1.offset, StrideMiB(kind)), e2.offset)

Spec == Init /\ [][Next]_vars
=============================================================================
