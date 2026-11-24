---- MODULE BlockLayout_TTrace_1795485409 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, BlockLayout, BlockLayout_TEConstants

_expression ==
    LET BlockLayout_TEExpression == INSTANCE BlockLayout_TEExpression
    IN BlockLayout_TEExpression!expression
----

_trace ==
    LET BlockLayout_TETrace == INSTANCE BlockLayout_TETrace
    IN BlockLayout_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        phase = ((a :> "done" @@ b :> "prepare"))
        /\
        startKept = (FALSE)
        /\
        spanStart = ((a :> 0 @@ b :> 0))
        /\
        pin = ((a :> {} @@ b :> {}))
        /\
        pos = ((a :> 1 @@ b :> 0))
        /\
        pool = (<<[generation |-> 2, kind |-> "window", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>)
        /\
        startTable = ((<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]))
        /\
        startState = (0)
        /\
        startPos = (0)
        /\
        table = ((a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 1, generation |-> 2, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])))
        /\
        target = ((a :> 1 @@ b :> 1))
        /\
        budget = ((a :> 2 @@ b :> 2))
    )
----

_init ==
    /\ phase = _TETrace[1].phase
    /\ startPos = _TETrace[1].startPos
    /\ pos = _TETrace[1].pos
    /\ pool = _TETrace[1].pool
    /\ table = _TETrace[1].table
    /\ startState = _TETrace[1].startState
    /\ target = _TETrace[1].target
    /\ startKept = _TETrace[1].startKept
    /\ pin = _TETrace[1].pin
    /\ spanStart = _TETrace[1].spanStart
    /\ startTable = _TETrace[1].startTable
    /\ budget = _TETrace[1].budget
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ phase  = _TETrace[i].phase
        /\ phase' = _TETrace[j].phase
        /\ startPos  = _TETrace[i].startPos
        /\ startPos' = _TETrace[j].startPos
        /\ pos  = _TETrace[i].pos
        /\ pos' = _TETrace[j].pos
        /\ pool  = _TETrace[i].pool
        /\ pool' = _TETrace[j].pool
        /\ table  = _TETrace[i].table
        /\ table' = _TETrace[j].table
        /\ startState  = _TETrace[i].startState
        /\ startState' = _TETrace[j].startState
        /\ target  = _TETrace[i].target
        /\ target' = _TETrace[j].target
        /\ startKept  = _TETrace[i].startKept
        /\ startKept' = _TETrace[j].startKept
        /\ pin  = _TETrace[i].pin
        /\ pin' = _TETrace[j].pin
        /\ spanStart  = _TETrace[i].spanStart
        /\ spanStart' = _TETrace[j].spanStart
        /\ startTable  = _TETrace[i].startTable
        /\ startTable' = _TETrace[j].startTable
        /\ budget  = _TETrace[i].budget
        /\ budget' = _TETrace[j].budget

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("BlockLayout_TTrace_1795485409.json", _TETrace)

=============================================================================

 Note that you can extract this module `BlockLayout_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `BlockLayout_TEExpression.tla` file takes precedence 
  over the module `BlockLayout_TEExpression` below).

---- MODULE BlockLayout_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, BlockLayout, BlockLayout_TEConstants

expression == 
    [
        \* To hide variables of the `BlockLayout` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        phase |-> phase
        ,startPos |-> startPos
        ,pos |-> pos
        ,pool |-> pool
        ,table |-> table
        ,startState |-> startState
        ,target |-> target
        ,startKept |-> startKept
        ,pin |-> pin
        ,spanStart |-> spanStart
        ,startTable |-> startTable
        ,budget |-> budget
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_phaseUnchanged |-> phase = phase'
        
        \* Format the `phase` variable as Json value.
        \* ,_phaseJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(phase)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_phaseModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].phase # _TETrace[s-1].phase
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE BlockLayout_TETrace ----
\*EXTENDS IOUtils, TLC, BlockLayout, BlockLayout_TEConstants
\*
\*trace == IODeserialize("BlockLayout_TTrace_1795485409.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE BlockLayout_TETrace ----
EXTENDS TLC, BlockLayout, BlockLayout_TEConstants

trace == 
    <<
    ([phase |-> (a :> "idle" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 0 @@ b :> 0),pool |-> <<[generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 0 @@ b :> 0),budget |-> (a :> 0 @@ b :> 0)]),
    ([phase |-> (a :> "prepare" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 0 @@ b :> 0),pool |-> <<[generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "prepare" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 0 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "prepare" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 0 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "step" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {<<"window", 1, 1>>, <<"full", 2, 1>>} @@ b :> {}),pos |-> (a :> 0 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "step" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {<<"window", 1, 1>>, <<"full", 2, 1>>} @@ b :> {}),pos |-> (a :> 1 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "idle" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 1 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "done" @@ b :> "idle"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 1 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 0),budget |-> (a :> 2 @@ b :> 0)]),
    ([phase |-> (a :> "done" @@ b :> "prepare"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 1 @@ b :> 0),pool |-> <<[generation |-> 1, kind |-> "window", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 1),budget |-> (a :> 2 @@ b :> 2)]),
    ([phase |-> (a :> "done" @@ b :> "prepare"),startKept |-> FALSE,spanStart |-> (a :> 0 @@ b :> 0),pin |-> (a :> {} @@ b :> {}),pos |-> (a :> 1 @@ b :> 0),pool |-> <<[generation |-> 2, kind |-> "window", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)], [generation |-> 1, kind |-> "full", block |-> 0, length |-> 1, data |-> (0 :> 1 @@ 1 :> 0)], [generation |-> 0, kind |-> "full", block |-> 0, length |-> 0, data |-> (0 :> 0 @@ 1 :> 0)]>>,startTable |-> (<<"window", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]),startState |-> 0,startPos |-> 0,table |-> (a :> (<<"window", 0>> :> [slot |-> 1, generation |-> 1, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 2, generation |-> 1, offset |-> [hi |-> 2, lo |-> 4096]] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0]) @@ b :> (<<"window", 0>> :> [slot |-> 1, generation |-> 2, offset |-> [hi |-> 2, lo |-> 0]] @@ <<"window", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 0>> :> [slot |-> 0, generation |-> 0, offset |-> 0] @@ <<"full", 1>> :> [slot |-> 0, generation |-> 0, offset |-> 0])),target |-> (a :> 1 @@ b :> 1),budget |-> (a :> 2 @@ b :> 2)])
    >>
----


=============================================================================

---- MODULE BlockLayout_TEConstants ----
EXTENDS BlockLayout

CONSTANTS a, b

=============================================================================

---- CONFIG BlockLayout_TTrace_1795485409 ----
CONSTANTS
    Lanes = { a , b }
    Capacity = 3
    B = 2
    W = 2
    MaxPos = 4
    MaxSpan = 2
    MaxGeneration = 2
    Bug = "stale"
    a = a
    b = b

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Mon Nov 23 20:56:50 EST 2026