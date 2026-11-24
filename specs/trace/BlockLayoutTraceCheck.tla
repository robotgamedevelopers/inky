------------------------ MODULE BlockLayoutTraceCheck ------------------------
\* Behavioral trace-conformance checker for specs/BlockLayout.tla.
\* Replays observed NDJSON engine events step-by-step against the unmodified
\* BlockLayout specification. Every step 1..Len(Events) must satisfy both
\* MatchBlockEvent(Events[stepIdx]) and BlockLayout!Next, all 9 BlockLayout
\* safety invariants must hold at every state, and CHECK_DEADLOCK TRUE ensures
\* TLC reaches the terminal stuttering state stepIdx = Len(Events) + 1.

EXTENDS BlockLayout, BlockLayoutTraceData, Sequences

ASSUME Len(Events) >= 1

VARIABLE stepIdx

TraceVars == <<vars, stepIdx>>

MatchBlockEvent(ev) ==
    CASE ev.op = "prepare" ->
            /\ Prepare(ev.lane, ev.tokens)
            /\ (ev.hasBudget => budget'[ev.lane] = ev.budgetVal)
            /\ (ev.hasPos => pos'[ev.lane] = ev.posVal)
      [] ev.op = "alloc" ->
            /\ Alloc(ev.lane, <<ev.kind, ev.block>>, ev.slot)
            /\ (ev.hasGen => table'[ev.lane][<<ev.kind, ev.block>>].generation = ev.gen)
            /\ (ev.hasOffset =>
                    table'[ev.lane][<<ev.kind, ev.block>>].offset =
                        [hi |-> ev.offsetHi, lo |-> ev.offsetLo])
      [] ev.op = "begin_step" ->
            /\ BeginStep(ev.lane)
            /\ (ev.hasPinCount => Cardinality(pin'[ev.lane]) = ev.pinCount)
      [] ev.op = "write" ->
            /\ Write(ev.lane)
            /\ (ev.hasPos => pos'[ev.lane] = ev.posVal)
      [] ev.op = "end_step" ->
            /\ EndStep(ev.lane)
            /\ (ev.hasPos => pos'[ev.lane] = ev.posVal)
      [] ev.op = "slide_window" ->
            /\ SlideWindow(ev.lane)
      [] ev.op = "keep_start" ->
            /\ KeepStart(ev.lane)
            /\ (ev.hasPos => startPos' = ev.posVal)
      [] ev.op = "drop_start" ->
            /\ DropStart
      [] ev.op = "match" ->
            /\ Match(ev.lane)
            /\ (ev.hasPos => pos'[ev.lane] = ev.posVal)
      [] ev.op = "finish" ->
            /\ Finish(ev.lane)
      [] OTHER -> FALSE

TraceInit ==
    /\ Init
    /\ stepIdx = 1

TraceNext ==
    \/ /\ stepIdx <= Len(Events)
       /\ MatchBlockEvent(Events[stepIdx])
       /\ stepIdx' = stepIdx + 1
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED TraceVars

TraceSpec == TraceInit /\ [][TraceNext]_TraceVars

StepConforms ==
    \/ /\ stepIdx <= Len(Events)
       /\ Next
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED vars

Conforms == Init /\ [][StepConforms]_TraceVars

=============================================================================
