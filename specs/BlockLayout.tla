---------------------------- MODULE BlockLayout ----------------------------
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Lanes, Capacity, B, W, MaxPos, MaxSpan, MaxGeneration, Bug
ASSUME /\ Capacity >= 3 /\ B > 0 /\ W > 0
       /\ MaxPos > B /\ MaxSpan > 0 /\ MaxGeneration > 1
       /\ Bug \in {"none", "stale", "window", "span", "start"}

Kinds == {"full", "window"}
Slots == 1..Capacity
Blocks == 0..((MaxPos - 1) \div B)
Keys == Kinds \X Blocks
Cells == Slots
NoEntry == [slot |-> 0, generation |-> 0, offset |-> 0]
EmptyTable == [k \in Keys |-> NoEntry]
\* A base at 2^33 catches any accidental 32-bit offset truncation.
BaseHigh == 2
Stride == 4096
Offset(s) == [hi |-> BaseHigh, lo |-> (s - 1) * Stride]
AddBytes(o, n) == [hi |-> o.hi, lo |-> o.lo + n]
Ceil(n, d) == (n + d - 1) \div d
SpanBound(c) == Ceil(W - 1 + c, B) + 1
Window(lo, hi) == IF hi < lo THEN {} ELSE {p \div B : p \in lo..hi}
History(n) == Window(IF n > W THEN n - W ELSE 0, n - 1)
SpanWindow(n, c) == Window(IF n >= W THEN n - W + 1 ELSE 0, n + c - 1)
Full(n) == Window(0, n - 1)
Needed(n, c) == ({"full"} \X Full(n + c))
                 \cup ({"window"} \X SpanWindow(n, c))
Handle(kind, e) == <<kind, e.slot, e.generation>>
Handles(t) == {Handle(k[1], t[k]) : k \in {j \in Keys : t[j] # NoEntry}}
Restrict(t, ks) == [k \in Keys |-> IF k \in ks THEN t[k] ELSE NoEntry]

VARIABLES pool, table, pos, phase, target, spanStart, budget, pin,
          startKept, startPos, startTable, startState
vars == <<pool, table, pos, phase, target, spanStart, budget, pin,
          startKept, startPos, startTable, startState>>

Init ==
 /\ pool = [x \in Cells |-> [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0,
                            data |-> [i \in 0..(B-1) |-> 0]]]
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
Free(s) == ~\E h \in Owned : h[2] = s
Resolves(kind, e) == /\ e # NoEntry
                      /\ e.generation = pool[e.slot].generation
                      /\ pool[e.slot].kind = kind
                      /\ e.offset = Offset(e.slot)

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
 /\ table[l][k] = NoEntry /\ s \in Slots /\ Free(s)
 /\ pool[s].generation < MaxGeneration
 /\ \A r \in 1..(s-1) : ~Free(r) \/ pool[r].generation = MaxGeneration
 /\ LET g == pool[s].generation + 1 IN
     /\ pool' = [pool EXCEPT ![s] =
           [generation |-> g, kind |-> k[1], block |-> k[2], length |-> 0, data |-> [i \in 0..(B-1) |-> 0]]]
     /\ table' = [table EXCEPT ![l][k] = [slot |-> s, generation |-> g, offset |-> Offset(s)]]
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
        cells == {table[l][<<kind,b>>].slot : kind \in Kinds}
    IN pool' = [x \in Cells |-> IF x \in cells THEN
           [pool[x] EXCEPT !.length = @ + 1, !.data[pos[l] % B] = 1]
           ELSE pool[x]]
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
          \/ (\E k \in Keys, s \in Slots : Alloc(l,k,s))
          \/ BeginStep(l) \/ Write(l) \/ EndStep(l) \/ SlideWindow(l)
          \/ KeepStart(l) \/ Match(l) \/ Finish(l)) \/ DropStart

TypeOK ==
 /\ pos \in [Lanes -> 0..MaxPos]
 /\ phase \in [Lanes -> {"idle", "prepare", "step", "done"}]
 /\ \A x \in Cells : /\ pool[x].generation \in 0..MaxGeneration
                          /\ pool[x].kind \in Kinds
                          /\ pool[x].length \in 0..B
                          /\ pool[x].data \in [0..(B-1) -> {0,1}]
GenerationMatches == \A l \in Lanes, k \in Keys :
 table[l][k] # NoEntry => Resolves(k[1],table[l][k])
WindowCovered == \A l \in Lanes : phase[l] # "done" =>
 \A b \in History(pos[l]) : table[l][<<"window",b>>] # NoEntry
FullCovered == \A l \in Lanes : phase[l] # "done" =>
 \A b \in Full(pos[l]) : table[l][<<"full",b>>] # NoEntry
SpanPinned == \A l \in Lanes : phase[l] = "step" =>
 /\ budget[l] >= SpanBound(target[l]-spanStart[l])
 /\ Handles(Restrict(table[l], Needed(spanStart[l],target[l]-spanStart[l]))) \subseteq pin[l]
 /\ \A h \in pin[l] : h[3] = pool[h[2]].generation
StartRetained == startKept =>
 /\ startState = startPos
 /\ \A k \in ({"full"} \X Full(startPos)) \cup ({"window"} \X History(startPos)) :
      Resolves(k[1], startTable[k]) /\ pool[startTable[k].slot].length = B
TailPrivateZero == \A x \in Cells : pool[x].length < B =>
 /\ Cardinality({l \in Lanes : \E k \in Keys :
        table[l][k] # NoEntry /\ table[l][k].slot = x
        /\ phase[l] # "done"}) <= 1
 /\ \A i \in pool[x].length..(B-1) : pool[x].data[i] = 0
TableIndex == \A l \in Lanes, k \in Keys : table[l][k] # NoEntry =>
 pool[table[l][k].slot].block = k[2]
Address64 == \A l \in Lanes, k \in Keys : table[l][k] # NoEntry =>
 /\ table[l][k].offset.hi > 0
 \* A stronger signed-31-bit bound ensures no low-32-bit carry in TLC.
 /\ table[l][k].offset.lo >= 0
 /\ table[l][k].offset.lo <= 2147483647 - (B-1)*2
 /\ \A i \in 0..(B-1) :
      AddBytes(table[l][k].offset, i * 2) = AddBytes(Offset(table[l][k].slot), i * 2)

Spec == Init /\ [][Next]_vars
=============================================================================
