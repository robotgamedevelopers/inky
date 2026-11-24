---------------------------- MODULE PoisonGate ----------------------------
\* The poisoned-slot gate's slot arithmetic (src/poolgate.rs, poison_plan and
\* apply/restore), for one kind's pool; the code runs it once per kind and the
\* kinds share no slots. Lanes take, share and release slots between steps;
\* before a step the gate poisons every slot no stepping lane references and
\* saves the poisoned slots some live lane references; the step writes its
\* lanes' slots; the gate then puts the saved slots back. A slot handed out by
\* the pool is zeroed (the pool's rule, DESIGN.md "pool slots zeroed on reuse").
\*
\* Bug selects a reversed witness, each a guard the code has:
\*   "unguarded" the first stepping lane's slots are not protected
\*               (INK_POOLGATE_SABOTAGE): NoFalsePoison fails.
\*   "nosave"    nothing is saved before the poison: LiveIntact fails.
\*   "ownedonly" only referenced slots are poisoned, free ones are skipped:
\*               Coverage fails (the gate could not see a pool that forgets
\*               to zero a reused slot).
EXTENDS Integers, FiniteSets

CONSTANTS Lanes, Capacity, MaxWrite, Bug
ASSUME /\ Capacity \in 1..6 /\ MaxWrite \in 1..3
       /\ Bug \in {"none", "unguarded", "nosave", "ownedonly"}

Slots == 1..Capacity
Poison == -1
None == -2
\* the code's "first stepping lane": any fixed choice will do
First(S) == CHOOSE l \in S : TRUE

VARIABLES table, data, truth, saved, stepping, phase
vars == <<table, data, truth, saved, stepping, phase>>

Owned == UNION {table[l] : l \in Lanes}
Read(S) == UNION {table[l] : l \in S}
Protect(S) == IF Bug = "unguarded" THEN Read(S \ {First(S)}) ELSE Read(S)
PoisonSet(S) == IF Bug = "ownedonly" THEN Owned \ Protect(S) ELSE Slots \ Protect(S)
SaveSet(S) == IF Bug = "nosave" THEN {} ELSE PoisonSet(S) \cap Owned

TypeOK ==
  /\ table \in [Lanes -> SUBSET Slots]
  /\ data \in [Slots -> {Poison} \cup 0..MaxWrite]
  /\ truth \in [Slots -> 0..MaxWrite]
  /\ saved \in [Slots -> {None, Poison} \cup 0..MaxWrite]
  /\ stepping \in SUBSET Lanes
  /\ phase \in {"idle", "poisoned"}

Init ==
  /\ table = [l \in Lanes |-> {}]
  /\ data = [s \in Slots |-> Poison]      \* never-written memory is arbitrary
  /\ truth = [s \in Slots |-> 0]
  /\ saved = [s \in Slots |-> None]
  /\ stepping = {}
  /\ phase = "idle"

\* the pool hands a free slot to a lane, zeroed
Take(l, s) ==
  /\ phase = "idle" /\ s \notin Owned
  /\ table' = [table EXCEPT ![l] = @ \cup {s}]
  /\ data' = [data EXCEPT ![s] = 0]
  /\ truth' = [truth EXCEPT ![s] = 0]
  /\ UNCHANGED <<saved, stepping, phase>>

\* a lane matches another lane's (sealed) slot
Share(l, s) ==
  /\ phase = "idle" /\ s \in Owned \ table[l]
  /\ table' = [table EXCEPT ![l] = @ \cup {s}]
  /\ UNCHANGED <<data, truth, saved, stepping, phase>>

\* a lane ends, or its window slides past a block
Release(l, s) ==
  /\ phase = "idle" /\ s \in table[l]
  /\ table' = [table EXCEPT ![l] = @ \ {s}]
  /\ UNCHANGED <<data, truth, saved, stepping, phase>>

\* poison_plan, then apply: save, then poison
PoisonStep(S) ==
  /\ phase = "idle" /\ S # {}
  /\ saved' = [s \in Slots |-> IF s \in SaveSet(S) THEN data[s] ELSE None]
  /\ data' = [s \in Slots |-> IF s \in PoisonSet(S) THEN Poison ELSE data[s]]
  /\ stepping' = S /\ phase' = "poisoned"
  /\ UNCHANGED <<table, truth>>

\* the step writes a slot of one of its lanes
Write(l, s) ==
  /\ phase = "poisoned" /\ l \in stepping /\ s \in table[l] /\ truth[s] < MaxWrite
  /\ data' = [data EXCEPT ![s] = truth[s] + 1]
  /\ truth' = [truth EXCEPT ![s] = @ + 1]
  /\ UNCHANGED <<table, saved, stepping, phase>>

\* restore: the saved slots go back
Restore ==
  /\ phase = "poisoned"
  /\ data' = [s \in Slots |-> IF saved[s] # None THEN saved[s] ELSE data[s]]
  /\ saved' = [s \in Slots |-> None]
  /\ stepping' = {} /\ phase' = "idle"
  /\ UNCHANGED <<table, truth>>

Next ==
  \/ \E l \in Lanes, s \in Slots : Take(l, s) \/ Share(l, s) \/ Release(l, s) \/ Write(l, s)
  \/ \E S \in SUBSET Lanes : PoisonStep(S)
  \/ Restore

Spec == Init /\ [][Next]_vars

\* no slot a stepping lane reads holds poison: the gate never fails a correct pool
NoFalsePoison == phase = "poisoned" => \A s \in Read(stepping) : data[s] # Poison
\* every slot the step must not read holds poison: the gate is as strong as it says
Coverage == phase = "poisoned" => \A s \in Slots \ Read(stepping) : data[s] = Poison
\* between steps every referenced slot holds exactly what its lanes wrote
LiveIntact == phase = "idle" => \A s \in Owned : data[s] = truth[s]
=============================================================================
