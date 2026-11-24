---- MODULE SplitArena_TTrace_1795485562 ----
EXTENDS Sequences, TLCExt, SplitArena, Toolbox, Naturals, TLC

_expression ==
    LET SplitArena_TEExpression == INSTANCE SplitArena_TEExpression
    IN SplitArena_TEExpression!expression
----

_trace ==
    LET SplitArena_TETrace == INSTANCE SplitArena_TETrace
    IN SplitArena_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        p0 = (0)
        /\
        phase = ("comb")
        /\
        t = (5)
        /\
        start = (0)
        /\
        arena = ((0 :> <<0, 0, 0>> @@ 2 :> <<0, 1, 0>> @@ 4 :> <<1, 0, 0>> @@ 6 :> <<1, 1, 0>> @@ 8 :> <<2, 0, 0>> @@ 10 :> <<2, 1, 0>> @@ 12 :> <<3, 0, 0>> @@ 14 :> <<3, 1, 0>>))
        /\
        swa = (FALSE)
    )
----

_init ==
    /\ p0 = _TETrace[1].p0
    /\ swa = _TETrace[1].swa
    /\ t = _TETrace[1].t
    /\ phase = _TETrace[1].phase
    /\ start = _TETrace[1].start
    /\ arena = _TETrace[1].arena
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ p0  = _TETrace[i].p0
        /\ p0' = _TETrace[j].p0
        /\ swa  = _TETrace[i].swa
        /\ swa' = _TETrace[j].swa
        /\ t  = _TETrace[i].t
        /\ t' = _TETrace[j].t
        /\ phase  = _TETrace[i].phase
        /\ phase' = _TETrace[j].phase
        /\ start  = _TETrace[i].start
        /\ start' = _TETrace[j].start
        /\ arena  = _TETrace[i].arena
        /\ arena' = _TETrace[j].arena

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("SplitArena_TTrace_1795485562.json", _TETrace)

=============================================================================

 Note that you can extract this module `SplitArena_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `SplitArena_TEExpression.tla` file takes precedence 
  over the module `SplitArena_TEExpression` below).

---- MODULE SplitArena_TEExpression ----
EXTENDS Sequences, TLCExt, SplitArena, Toolbox, Naturals, TLC

expression == 
    [
        \* To hide variables of the `SplitArena` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        p0 |-> p0
        ,swa |-> swa
        ,t |-> t
        ,phase |-> phase
        ,start |-> start
        ,arena |-> arena
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_p0Unchanged |-> p0 = p0'
        
        \* Format the `p0` variable as Json value.
        \* ,_p0Json |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(p0)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_p0ModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].p0 # _TETrace[s-1].p0
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE SplitArena_TETrace ----
\*EXTENDS IOUtils, SplitArena, TLC
\*
\*trace == IODeserialize("SplitArena_TTrace_1795485562.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE SplitArena_TETrace ----
EXTENDS SplitArena, TLC

trace == 
    <<
    ([p0 |-> 0,phase |-> "attn",t |-> 5,start |-> 0,arena |-> <<>>,swa |-> FALSE]),
    ([p0 |-> 0,phase |-> "comb",t |-> 5,start |-> 0,arena |-> (0 :> <<0, 0, 0>> @@ 2 :> <<0, 1, 0>> @@ 4 :> <<1, 0, 0>> @@ 6 :> <<1, 1, 0>> @@ 8 :> <<2, 0, 0>> @@ 10 :> <<2, 1, 0>> @@ 12 :> <<3, 0, 0>> @@ 14 :> <<3, 1, 0>>),swa |-> FALSE])
    >>
----


=============================================================================

---- CONFIG SplitArena_TTrace_1795485562 ----
CONSTANTS
    Capacity = 4
    Bug = "rowsUnbounded"

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
\* Generated on Mon Nov 23 20:59:22 EST 2026