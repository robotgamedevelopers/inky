---------------------- MODULE CacheSchedulerTraceCheck ----------------------
\* Behavioral trace-conformance checker for specs/CacheScheduler.tla.
\* Replays observed NDJSON engine/scheduler events step-by-step against the
\* unmodified CacheScheduler specification. Every step 1..Len(Events) must
\* satisfy MatchSchedEvent(Events[stepIdx]) and CacheScheduler!Next, all 12
\* safety invariants and all 4 action properties must hold, and CHECK_DEADLOCK
\* TRUE ensures TLC reaches stepIdx = Len(Events) + 1.

EXTENDS CacheScheduler, CacheSchedulerTraceData, Sequences

ASSUME Len(Events) >= 1

VARIABLE stepIdx

TraceVars == <<vars, stepIdx>>

TreeBlockOf(ev) == <<"tree", ev.prefix, ev.kind, ev.block>>

MatchSchedEvent(ev) ==
    CASE ev.op = "arrive" ->
            /\ Arrive(ev.lane, ev.prefix, ev.target)
            /\ (ev.hasState => laneState'[ev.lane] = ev.stateVal)
      [] ev.op = "refuse429" ->
            /\ Arrive(ev.lane, ev.prefix, ev.target)
            /\ laneState'[ev.lane] = "Refused"
      [] ev.op = "admit" ->
            /\ Admit(ev.lane)
            /\ (ev.hasWait => waitTicks'[ev.lane] = ev.waitVal)
      [] ev.op = "begin_restore" ->
            /\ BeginRestore(ev.lane)
            /\ (ev.hasPos => lanePos'[ev.lane] = ev.posVal)
            /\ (ev.hasRes => reservation'[ev.lane] = ev.resVal)
      [] ev.op = "restore_block" ->
            /\ RestoreBlock(ev.lane, TreeBlockOf(ev))
      [] ev.op = "finish_restore" ->
            /\ FinishRestore(ev.lane)
      [] ev.op = "evict_mem" ->
            /\ EvictCleanOrUnrefMem
            /\ (ev.hasBlock => blocks'[TreeBlockOf(ev)].tier \notin {"memory", "both"})
      [] ev.op = "start_writeback" ->
            /\ StartColdWriteback
            /\ (ev.hasBlock => stepIO' = {TreeBlockOf(ev)})
      [] ev.op = "complete_step_io" ->
            /\ CompleteStepIO
            /\ (ev.hasTier => blocks'[TreeBlockOf(ev)].tier = ev.tierVal)
      [] ev.op = "reclaim_disk" ->
            /\ ReclaimColdDisk
      [] ev.op = "drop_unreferenced" ->
            /\ DropUnreferenced(TreeBlockOf(ev))
      [] ev.op = "slide_window" ->
            /\ SlideWindow(ev.lane)
      [] ev.op = "build_step" ->
            /\ BuildStep(ev.chosen, ev.spans)
            /\ (ev.hasPinCount => Cardinality(stepPins') = ev.pinCount)
      [] ev.op = "gpu_execute" ->
            /\ GpuExecute
      [] ev.op = "finish_step" ->
            /\ FinishStep
            /\ (ev.hasPos => lanePos'[ev.lane] = ev.posVal)
      [] ev.op = "finish" ->
            /\ FinishReady(ev.lane)
      [] ev.op = "cancel" ->
            /\ Cancel(ev.lane)
            /\ (ev.hasState => laneState'[ev.lane] = ev.stateVal)
      [] OTHER -> FALSE

TraceInit ==
    /\ Init
    /\ stepIdx = 1

TraceNext ==
    \/ /\ stepIdx <= Len(Events)
       /\ MatchSchedEvent(Events[stepIdx])
       /\ stepIdx' = stepIdx + 1
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED TraceVars

TraceSpec == TraceInit /\ [][TraceNext]_TraceVars

ExactCacheModelStep(ev) ==
    CASE ev.op \in {"arrive", "refuse429"} ->
            /\ ev.lane \in Lanes /\ ev.prefix \in Prefixes /\ ev.target \in RequestLengths
            /\ Arrive(ev.lane, ev.prefix, ev.target)
      [] ev.op = "admit"            -> ev.lane \in Lanes /\ Admit(ev.lane)
      [] ev.op = "begin_restore"    -> ev.lane \in Lanes /\ BeginRestore(ev.lane)
      [] ev.op = "restore_block"    -> ev.lane \in Lanes /\ TreeBlockOf(ev) \in BlockIds /\ RestoreBlock(ev.lane, TreeBlockOf(ev))
      [] ev.op = "finish_restore"   -> ev.lane \in Lanes /\ FinishRestore(ev.lane)
      [] ev.op = "evict_mem"        -> EvictCleanOrUnrefMem
      [] ev.op = "start_writeback"  -> StartColdWriteback
      [] ev.op = "complete_step_io" -> CompleteStepIO
      [] ev.op = "reclaim_disk"     -> ReclaimColdDisk
      [] ev.op = "drop_unreferenced" -> TreeBlockOf(ev) \in BlockIds /\ DropUnreferenced(TreeBlockOf(ev))
      [] ev.op = "slide_window"     -> ev.lane \in Lanes /\ SlideWindow(ev.lane)
      [] ev.op = "build_step"       ->
            /\ ev.chosen \in ((SUBSET Lanes) \ {{}})
            /\ ev.spans \in [Lanes -> 0..MaxSpan]
            /\ BuildStep(ev.chosen, ev.spans)
      [] ev.op = "gpu_execute"      -> GpuExecute
      [] ev.op = "finish_step"      -> FinishStep
      [] ev.op = "finish"           -> ev.lane \in Lanes /\ FinishReady(ev.lane)
      [] ev.op = "cancel"           -> ev.lane \in Lanes /\ Cancel(ev.lane)
      [] OTHER -> FALSE

StepConforms ==
    \/ /\ stepIdx <= Len(Events)
       /\ ExactCacheModelStep(Events[stepIdx])
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED vars

Conforms == [][StepConforms]_TraceVars

=============================================================================
