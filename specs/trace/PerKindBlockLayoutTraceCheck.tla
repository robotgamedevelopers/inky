--------------------- MODULE PerKindBlockLayoutTraceCheck ---------------------
\* Direct state-assigning trace replay for specs/PerKindBlockLayout.tla.
\* Passes the Rust allocator's literal slot, generation, and 64-bit byte offset
\* straight into `table` with zero Python remapping:
\*   - Truncated 32-bit offsets or wrong kind strides violate `Address64` (exit 12).
\*   - Stale generations on reuse violate `GenerationMatches` (exit 12).
\*   - Non-lowest-free slot selection or premature reuse violates `Conforms` (exit 12).

EXTENDS PerKindBlockLayout, PerKindBlockLayoutTraceData, Sequences

ASSUME Len(Events) >= 1

VARIABLE stepIdx

TraceVars == <<vars, stepIdx>>

ApplyBlockEvent(ev) ==
    CASE ev.op = "prepare" ->
            /\ phase' = [phase EXCEPT ![ev.lane] = "prepare"]
            /\ target' = [target EXCEPT ![ev.lane] = pos[ev.lane] + ev.tokens]
            /\ spanStart' = [spanStart EXCEPT ![ev.lane] = pos[ev.lane]]
            /\ budget' = [budget EXCEPT ![ev.lane] =
                            IF ev.hasBudget THEN ev.budgetVal ELSE SpanBound(ev.tokens)]
            /\ UNCHANGED <<pool, table, pos, pin, startKept, startPos, startTable, startState>>
      [] ev.op = "alloc" ->
            /\ LET kd == ev.kind
                   s  == ev.slot
                   canBump == /\ kd \in Kinds
                              /\ s \in Slots(kd)
                              /\ Free(kd, s)
                              /\ pool[kd][s].generation < MaxGeneration
               IN /\ pool' = IF canBump
                             THEN [pool EXCEPT ![kd][s] =
                                    [generation |-> pool[kd][s].generation + 1,
                                     kind |-> kd, block |-> ev.block, length |-> 0,
                                     data |-> [i \in 0..(B - 1) |-> 0]]]
                             ELSE pool
                  /\ table' = IF kd \in Kinds /\ ev.block \in Blocks
                              THEN [table EXCEPT ![ev.lane][<<kd, ev.block>>] =
                                      [slot |-> s, generation |-> ev.gen, offset |-> ev.offset, kind |-> kd]]
                              ELSE table
            /\ UNCHANGED <<pos, phase, target, spanStart, budget, pin,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "begin_step" ->
            /\ phase' = [phase EXCEPT ![ev.lane] = "step"]
            /\ pin' = [pin EXCEPT ![ev.lane] =
                        Handles(Restrict(table[ev.lane], Needed(spanStart[ev.lane], target[ev.lane] - spanStart[ev.lane])))]
            /\ UNCHANGED <<pool, table, pos, target, spanStart, budget,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "write" ->
            /\ LET blkIdx == pos[ev.lane] \div B
                   off    == pos[ev.lane] % B
               IN pool' = [kd \in Kinds |-> [s \in Slots(kd) |->
                      IF blkIdx \in Blocks /\ table[ev.lane][<<kd, blkIdx>>].slot = s
                         /\ pool[kd][s].length < B /\ off \in 0..(B - 1)
                      THEN [pool[kd][s] EXCEPT !.length = @ + 1, !.data[off] = 1]
                      ELSE pool[kd][s]]]
            /\ pos' = [pos EXCEPT ![ev.lane] = IF ev.hasPos THEN ev.posVal ELSE @ + 1]
            /\ UNCHANGED <<table, phase, target, spanStart, budget, pin,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "end_step" ->
            /\ phase' = [phase EXCEPT ![ev.lane] = "idle"]
            /\ pin' = [pin EXCEPT ![ev.lane] = {}]
            /\ UNCHANGED <<pool, table, pos, target, spanStart, budget,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "slide_window" ->
            /\ LET t == Restrict(table[ev.lane], ({"full"} \X Blocks) \cup ({"window"} \X History(pos[ev.lane])))
               IN table' = [table EXCEPT ![ev.lane] = t]
            /\ UNCHANGED <<pool, pos, phase, target, spanStart, budget, pin,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "keep_start" ->
            /\ startKept' = TRUE
            /\ startPos' = IF ev.hasPos THEN ev.posVal ELSE pos[ev.lane]
            /\ startState' = pos[ev.lane]
            /\ startTable' = Restrict(table[ev.lane], ({"full"} \X Full(pos[ev.lane])) \cup ({"window"} \X History(pos[ev.lane])))
            /\ UNCHANGED <<pool, table, pos, phase, target, spanStart, budget, pin>>
      [] ev.op = "drop_start" ->
            /\ startKept' = FALSE
            /\ startTable' = EmptyTable
            /\ UNCHANGED <<pool, table, pos, phase, target, spanStart, budget, pin, startPos, startState>>
      [] ev.op = "match" ->
            /\ table' = [table EXCEPT ![ev.lane] = startTable]
            /\ pos' = [pos EXCEPT ![ev.lane] = IF ev.hasPos THEN ev.posVal ELSE startPos]
            /\ UNCHANGED <<pool, phase, target, spanStart, budget, pin,
                           startKept, startPos, startTable, startState>>
      [] ev.op = "finish" ->
            /\ phase' = [phase EXCEPT ![ev.lane] = "done"]
            /\ table' = [table EXCEPT ![ev.lane] = EmptyTable]
            /\ UNCHANGED <<pool, pos, target, spanStart, budget, pin,
                           startKept, startPos, startTable, startState>>
      [] OTHER -> FALSE

TraceInit ==
    /\ Init
    /\ stepIdx = 1

TraceNext ==
    \/ /\ stepIdx <= Len(Events)
       /\ ApplyBlockEvent(Events[stepIdx])
       /\ stepIdx' = stepIdx + 1
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED TraceVars

TraceSpec == TraceInit /\ [][TraceNext]_TraceVars

StepConforms ==
    \/ /\ stepIdx <= Len(Events)
       /\ Next
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED vars

Conforms == [][StepConforms]_TraceVars

=============================================================================
