----------------------- MODULE ServeSchedulerTraceCheck -----------------------
\* Direct state-assigning replay of src/serve/scheduler_trace.rs JSONL traces
\* against specs/ServeScheduler.tla:
\*   - Missing 429 refusal beyond LANES + QUEUE violates `InFlightBounded` (exit 12).
\*   - Over-chunk prefill or multi-token decode violates `StepBudgetAndModeValid` (exit 12).
\*   - Broken prefill/decode alternation, mid-step cancel, or out-of-order admit
\*     violates `Conforms` (exit 12).

EXTENDS ServeScheduler, ServeSchedulerTraceData, Sequences

ASSUME Len(Events) >= 1

VARIABLE stepIdx

TraceVars == <<vars, stepIdx>>

ClearPendingIfNotRefuse ==
    pendingRefusal' = 0

ApplyServeEvent(ev) ==
    CASE ev.op = "arrive" ->
            /\ seenJobs'  = seenJobs \cup {ev.job}
            /\ jobPrompt' = [jobPrompt EXCEPT ![ev.job] = ev.prompt]
            /\ jobMax'    = [jobMax EXCEPT ![ev.job] = ev.max]
            /\ inFlight'  = inFlight \cup {ev.job}
            /\ IF Cardinality(EffectiveInFlight) >= NumLanes + MaxQueue
               THEN /\ pendingRefusal' = ev.job
                    /\ UNCHANGED queue
               ELSE /\ pendingRefusal' = 0
                    /\ queue' = Append(queue, ev.job)
            /\ UNCHANGED <<laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                           disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "refuse429" ->
            /\ inFlight'       = inFlight \ {ev.job}
            /\ pendingRefusal' = 0
            /\ UNCHANGED <<seenJobs, queue, laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                           jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "refuse_ctx" ->
            /\ ClearPendingIfNotRefuse
            /\ queue'    = IF queue # <<>> /\ Head(queue) = ev.job THEN Tail(queue) ELSE queue
            /\ inFlight' = inFlight \ {ev.job}
            /\ UNCHANGED <<seenJobs, laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                           jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "admit" ->
            /\ ClearPendingIfNotRefuse
            /\ queue'      = IF queue # <<>> /\ Head(queue) = ev.job THEN Tail(queue) ELSE queue
            /\ laneJob'    = [laneJob EXCEPT ![ev.lane] = ev.job]
            /\ lanePos'    = [lanePos EXCEPT ![ev.lane] = ev.cached]
            /\ lanePrompt' = [lanePrompt EXCEPT ![ev.lane] = jobPrompt[ev.job]]
            /\ laneGen'    = [laneGen EXCEPT ![ev.lane] = 0]
            /\ keptPrompt' = [keptPrompt EXCEPT ![ev.lane] = 0]
            /\ UNCHANGED <<seenJobs, inFlight, jobPrompt, jobMax,
                           disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "build_step" ->
            /\ ClearPendingIfNotRefuse
            /\ stepActive'     = TRUE
            /\ stepMode'       = ev.mode
            /\ stepSpans'      = ev.spans
            /\ lastWasPrefill' = (ev.mode = "prefill")
            /\ UNCHANGED <<seenJobs, inFlight, queue,
                           laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                           jobPrompt, jobMax, disconnected>>
      [] ev.op = "finish_step" ->
            /\ ClearPendingIfNotRefuse
            /\ lanePos' = [l \in LaneIds |->
                            IF \E sp \in stepSpans : sp.lane = l
                            THEN LET sp == CHOOSE s \in stepSpans : s.lane = l
                                 IN sp.pos + sp.tokens
                            ELSE lanePos[l]]
            /\ keptPrompt' = [l \in LaneIds |->
                                IF stepMode = "prefill" /\
                                   (\E sp \in stepSpans : sp.lane = l /\ sp.pos + sp.tokens = lanePrompt[l])
                                THEN lanePrompt[l]
                                ELSE keptPrompt[l]]
            /\ laneGen' = [l \in LaneIds |->
                            IF \E sp \in stepSpans :
                                sp.lane = l /\ (stepMode = "decode" \/ sp.pos + sp.tokens = lanePrompt[l])
                            THEN laneGen[l] + 1
                            ELSE laneGen[l]]
            /\ stepActive' = FALSE
            /\ stepMode'   = "none"
            /\ stepSpans'  = {}
            /\ UNCHANGED <<seenJobs, inFlight, queue,
                           laneJob, lanePrompt, jobPrompt, jobMax, disconnected, lastWasPrefill>>
      [] ev.op = "disconnect" ->
            /\ ClearPendingIfNotRefuse
            /\ disconnected' = [disconnected EXCEPT ![ev.job] = TRUE]
            /\ inFlight'     = inFlight \ {ev.job}
            /\ UNCHANGED <<seenJobs, queue,
                           laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                           jobPrompt, jobMax, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "cancel" ->
            /\ ClearPendingIfNotRefuse
            /\ laneJob'    = [laneJob EXCEPT ![ev.lane] = 0]
            /\ lanePos'    = [lanePos EXCEPT ![ev.lane] = 0]
            /\ lanePrompt' = [lanePrompt EXCEPT ![ev.lane] = 0]
            /\ laneGen'    = [laneGen EXCEPT ![ev.lane] = 0]
            /\ inFlight'   = inFlight \ {ev.job}
            /\ UNCHANGED <<seenJobs, queue, keptPrompt,
                           jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] ev.op = "finish" ->
            /\ ClearPendingIfNotRefuse
            /\ laneJob'    = [laneJob EXCEPT ![ev.lane] = 0]
            /\ lanePos'    = [lanePos EXCEPT ![ev.lane] = 0]
            /\ lanePrompt' = [lanePrompt EXCEPT ![ev.lane] = 0]
            /\ laneGen'    = [laneGen EXCEPT ![ev.lane] = 0]
            /\ inFlight'   = inFlight \ {ev.job}
            /\ UNCHANGED <<seenJobs, queue, keptPrompt,
                           jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>
      [] OTHER -> FALSE

\* Note: bind `Next` directly to the trace event at `stepIdx` so TLC does not
\* branch over exponentially many `SUBSET [job: Jobs, ...]` disjuncts when
\* checking `Conforms`!
ExactModelStep(ev) ==
    CASE ev.op = "arrive"      -> Arrive(ev.job, ev.prompt, ev.max)
      [] ev.op = "refuse429"   -> Refuse429(ev.job)
      [] ev.op = "refuse_ctx"  -> RefuseCtx(ev.job)
      [] ev.op = "admit"       -> Admit(ev.job, ev.lane, ev.cached, ev.srcKind, ev.fromLane)
      [] ev.op = "build_step"  -> BuildStep(ev.mode, ev.spans)
      [] ev.op = "finish_step" -> FinishStep
      [] ev.op = "disconnect"  -> Disconnect(ev.job)
      [] ev.op = "cancel"      -> Cancel(ev.job, ev.lane, ev.generated)
      [] ev.op = "finish"      -> Finish(ev.job, ev.lane, ev.reason, ev.generated)
      [] OTHER -> FALSE

TraceInit ==
    /\ Init
    /\ stepIdx = 1

TraceNext ==
    \/ /\ stepIdx <= Len(Events)
       /\ ApplyServeEvent(Events[stepIdx])
       /\ stepIdx' = stepIdx + 1
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED TraceVars

TraceSpec == TraceInit /\ [][TraceNext]_TraceVars

StepConforms ==
    \/ /\ stepIdx <= Len(Events)
       /\ ExactModelStep(Events[stepIdx])
    \/ /\ stepIdx = Len(Events) + 1
       /\ UNCHANGED vars

Conforms == [][StepConforms]_TraceVars

=============================================================================
