---- MODULE GroupedLayout_TTrace_1795485415 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, GroupedLayout

_expression ==
    LET GroupedLayout_TEExpression == INSTANCE GroupedLayout_TEExpression
    IN GroupedLayout_TEExpression!expression
----

_trace ==
    LET GroupedLayout_TETrace == INSTANCE GroupedLayout_TETrace
    IN GroupedLayout_TETrace!trace
----

_prop ==
    ~<>[](
        loaded = ({<<0, 1>>, <<0, 2>>})
        /\
        shared = (0)
        /\
        visits = (<<<<<<0, <<0, 1>>, 0>>>>, <<<<0, <<0, 2>>, 0>>>>>>)
        /\
        lengths = (<<1, 1>>)
        /\
        tile = (0)
        /\
        windowed = (FALSE)
        /\
        staged = (<<<<0, <<0, 1>>>>, <<0, <<0, 2>>>>>>)
        /\
        partials = (<<<<<<0, 0>>>>, <<<<0, 0>>>>>>)
    )
----

_init ==
    /\ shared = _TETrace[1].shared
    /\ tile = _TETrace[1].tile
    /\ windowed = _TETrace[1].windowed
    /\ visits = _TETrace[1].visits
    /\ partials = _TETrace[1].partials
    /\ loaded = _TETrace[1].loaded
    /\ lengths = _TETrace[1].lengths
    /\ staged = _TETrace[1].staged
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ shared  = _TETrace[i].shared
        /\ shared' = _TETrace[j].shared
        /\ tile  = _TETrace[i].tile
        /\ tile' = _TETrace[j].tile
        /\ windowed  = _TETrace[i].windowed
        /\ windowed' = _TETrace[j].windowed
        /\ visits  = _TETrace[i].visits
        /\ visits' = _TETrace[j].visits
        /\ partials  = _TETrace[i].partials
        /\ partials' = _TETrace[j].partials
        /\ loaded  = _TETrace[i].loaded
        /\ loaded' = _TETrace[j].loaded
        /\ lengths  = _TETrace[i].lengths
        /\ lengths' = _TETrace[j].lengths
        /\ staged  = _TETrace[i].staged
        /\ staged' = _TETrace[j].staged

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("GroupedLayout_TTrace_1795485415.json", _TETrace)

=============================================================================

 Note that you can extract this module `GroupedLayout_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `GroupedLayout_TEExpression.tla` file takes precedence 
  over the module `GroupedLayout_TEExpression` below).

---- MODULE GroupedLayout_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, GroupedLayout

expression == 
    [
        \* To hide variables of the `GroupedLayout` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        shared |-> shared
        ,tile |-> tile
        ,windowed |-> windowed
        ,visits |-> visits
        ,partials |-> partials
        ,loaded |-> loaded
        ,lengths |-> lengths
        ,staged |-> staged
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_sharedUnchanged |-> shared = shared'
        
        \* Format the `shared` variable as Json value.
        \* ,_sharedJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(shared)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_sharedModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].shared # _TETrace[s-1].shared
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE GroupedLayout_TETrace ----
\*EXTENDS IOUtils, TLC, GroupedLayout
\*
\*trace == IODeserialize("GroupedLayout_TTrace_1795485415.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE GroupedLayout_TETrace ----
EXTENDS TLC, GroupedLayout

trace == 
    <<
    ([loaded |-> {},shared |-> 0,visits |-> <<<<>>, <<>>>>,lengths |-> <<1, 1>>,tile |-> 0,windowed |-> FALSE,staged |-> <<>>,partials |-> <<<<>>, <<>>>>]),
    ([loaded |-> {<<0, 1>>},shared |-> 0,visits |-> <<<<<<0, <<0, 1>>, 0>>>>, <<>>>>,lengths |-> <<1, 1>>,tile |-> 0,windowed |-> FALSE,staged |-> <<<<0, <<0, 1>>>>>>,partials |-> <<<<<<0, 0>>>>, <<>>>>]),
    ([loaded |-> {<<0, 1>>, <<0, 2>>},shared |-> 0,visits |-> <<<<<<0, <<0, 1>>, 0>>>>, <<<<0, <<0, 2>>, 0>>>>>>,lengths |-> <<1, 1>>,tile |-> 0,windowed |-> FALSE,staged |-> <<<<0, <<0, 1>>>>, <<0, <<0, 2>>>>>>,partials |-> <<<<<<0, 0>>>>, <<<<0, 0>>>>>>])
    >>
----


=============================================================================

---- CONFIG GroupedLayout_TTrace_1795485415 ----
CONSTANTS
    Capacity = 3
    Bug = "stall"

PROPERTY
    _prop

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
\* Generated on Mon Nov 23 20:56:56 EST 2026