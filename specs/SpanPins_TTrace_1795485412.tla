---- MODULE SpanPins_TTrace_1795485412 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, SpanPins

_expression ==
    LET SpanPins_TEExpression == INSTANCE SpanPins_TEExpression
    IN SpanPins_TEExpression!expression
----

_trace ==
    LET SpanPins_TETrace == INSTANCE SpanPins_TETrace
    IN SpanPins_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        pin = ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
        /\
        offset = (63)
        /\
        selected = (TRUE)
    )
----

_init ==
    /\ pin = _TETrace[1].pin
    /\ selected = _TETrace[1].selected
    /\ offset = _TETrace[1].offset
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ pin  = _TETrace[i].pin
        /\ pin' = _TETrace[j].pin
        /\ selected  = _TETrace[i].selected
        /\ selected' = _TETrace[j].selected
        /\ offset  = _TETrace[i].offset
        /\ offset' = _TETrace[j].offset

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("SpanPins_TTrace_1795485412.json", _TETrace)

=============================================================================

 Note that you can extract this module `SpanPins_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `SpanPins_TEExpression.tla` file takes precedence 
  over the module `SpanPins_TEExpression` below).

---- MODULE SpanPins_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, SpanPins

expression == 
    [
        \* To hide variables of the `SpanPins` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        pin |-> pin
        ,selected |-> selected
        ,offset |-> offset
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_pinUnchanged |-> pin = pin'
        
        \* Format the `pin` variable as Json value.
        \* ,_pinJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(pin)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_pinModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].pin # _TETrace[s-1].pin
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE SpanPins_TETrace ----
\*EXTENDS IOUtils, TLC, SpanPins
\*
\*trace == IODeserialize("SpanPins_TTrace_1795485412.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE SpanPins_TETrace ----
EXTENDS TLC, SpanPins

trace == 
    <<
    ([pin |-> {},offset |-> 0,selected |-> FALSE]),
    ([pin |-> {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},offset |-> 63,selected |-> TRUE])
    >>
----


=============================================================================

---- CONFIG SpanPins_TTrace_1795485412 ----
CONSTANTS
    B = 64
    W = 1024
    C = 512
    Bug = TRUE

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
\* Generated on Mon Nov 23 20:56:53 EST 2026