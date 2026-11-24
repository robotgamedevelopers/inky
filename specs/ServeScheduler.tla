--------------------------- MODULE ServeScheduler ---------------------------
\* Behavioral model of the production serve scheduler (src/serve.rs +
\* src/serve/scheduler_trace.rs, schema versions 1 and 2):
\*   - Atomic HTTP capacity limit `NumLanes + MaxQueue` (`InFlightBounded`),
\*     answering `refuse429` immediately when full;
\*   - Context-overflow refusal `refuse_ctx` when `prompt + max + 2 > MaxCtx`;
\*   - FIFO queue drain into free lanes (`admit` with cold/own/copy prefix reuse);
\*   - Single step in flight (`build_step` -> `finish_step`), chunk-bounded
\*     prefill (`<= ChunkBudget`), 1-token decode, and 1-for-1 prefill/decode
\*     alternation (`lastWasPrefill`) when both phases are ready;
\*   - Client `disconnect` followed by step-boundary `cancel`, and `finish`.

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Jobs, NumLanes, MaxQueue, ChunkBudget, MaxCtx

ASSUME /\ Jobs # {} /\ 0 \notin Jobs
       /\ NumLanes >= 1 /\ MaxQueue >= 0
       /\ ChunkBudget >= 1 /\ MaxCtx >= 4

LaneIds == 0..(NumLanes - 1)

RECURSIVE SeqToSet(_)
SeqToSet(sq) ==
    IF sq = <<>> THEN {}
    ELSE {Head(sq)} \cup SeqToSet(Tail(sq))

RECURSIVE SumSpanTokens(_)
SumSpanTokens(S) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE v \in S : TRUE
         IN x.tokens + SumSpanTokens(S \ {x})

VARIABLES
    seenJobs,        \* Subset of Jobs that have arrived so far
    inFlight,        \* Subset of Jobs holding an InFlight capacity permit
    pendingRefusal,  \* 0 or job id awaiting immediate atomic refuse429
    queue,           \* FIFO Seq(Jobs) waiting for lane admission
    laneJob,         \* [LaneIds -> Jobs \cup {0}]
    lanePos,         \* [LaneIds -> 0..MaxCtx]
    lanePrompt,      \* [LaneIds -> 0..MaxCtx]
    laneGen,         \* [LaneIds -> 0..MaxCtx]
    keptPrompt,      \* [LaneIds -> 0..MaxCtx] (0 = no kept conv state)
    jobPrompt,       \* [Jobs -> 0..MaxCtx]
    jobMax,          \* [Jobs -> 0..MaxCtx]
    disconnected,    \* [Jobs -> BOOLEAN]
    stepActive,      \* BOOLEAN: batch step in flight on engine
    stepMode,        \* "none" | "prefill" | "decode"
    stepSpans,       \* Set of [job, lane, pos, tokens] records in current step
    lastWasPrefill   \* BOOLEAN: 1-for-1 alternation between prefill and decode

vars == <<
    seenJobs, inFlight, pendingRefusal, queue,
    laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
    jobPrompt, jobMax, disconnected,
    stepActive, stepMode, stepSpans, lastWasPrefill
>>

Init ==
    /\ seenJobs       = {}
    /\ inFlight       = {}
    /\ pendingRefusal = 0
    /\ queue          = <<>>
    /\ laneJob        = [l \in LaneIds |-> 0]
    /\ lanePos        = [l \in LaneIds |-> 0]
    /\ lanePrompt     = [l \in LaneIds |-> 0]
    /\ laneGen        = [l \in LaneIds |-> 0]
    /\ keptPrompt     = [l \in LaneIds |-> 0]
    /\ jobPrompt      = [j \in Jobs |-> 0]
    /\ jobMax         = [j \in Jobs |-> 0]
    /\ disconnected   = [j \in Jobs |-> FALSE]
    /\ stepActive     = FALSE
    /\ stepMode       = "none"
    /\ stepSpans      = {}
    /\ lastWasPrefill = FALSE

HasFreeLane      == \E l \in LaneIds : laneJob[l] = 0
HasReadyPrefill  == \E l \in LaneIds : laneJob[l] # 0 /\ lanePos[l] < lanePrompt[l]
HasReadyDecode   == \E l \in LaneIds : laneJob[l] # 0 /\ lanePos[l] >= lanePrompt[l]
EffectiveInFlight == inFlight \ (IF pendingRefusal = 0 THEN {} ELSE {pendingRefusal})

Arrive(j, promptLen, maxLen) ==
    /\ pendingRefusal = 0
    /\ j \in Jobs /\ j \notin seenJobs
    /\ promptLen \in 1..MaxCtx /\ maxLen \in 1..MaxCtx
    /\ seenJobs'  = seenJobs \cup {j}
    /\ jobPrompt' = [jobPrompt EXCEPT ![j] = promptLen]
    /\ jobMax'    = [jobMax EXCEPT ![j] = maxLen]
    /\ inFlight'  = inFlight \cup {j}
    /\ IF Cardinality(inFlight) >= NumLanes + MaxQueue
       THEN /\ pendingRefusal' = j
            /\ UNCHANGED queue
       ELSE /\ pendingRefusal' = 0
            /\ queue' = Append(queue, j)
    /\ UNCHANGED <<laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                   disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

Refuse429(j) ==
    /\ pendingRefusal = j
    /\ j \in inFlight
    /\ inFlight'       = inFlight \ {j}
    /\ pendingRefusal' = 0
    /\ UNCHANGED <<seenJobs, queue, laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                   jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

RefuseCtx(j) ==
    /\ pendingRefusal = 0
    /\ ~stepActive
    /\ queue # <<>> /\ Head(queue) = j
    /\ HasFreeLane
    /\ jobPrompt[j] + jobMax[j] + 2 > MaxCtx
    /\ queue'    = Tail(queue)
    /\ inFlight' = inFlight \ {j}
    /\ UNCHANGED <<seenJobs, pendingRefusal, laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                   jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

Admit(j, l, cached, srcKind, fromLane) ==
    /\ pendingRefusal = 0
    /\ ~stepActive
    /\ queue # <<>> /\ Head(queue) = j
    /\ l \in LaneIds /\ laneJob[l] = 0
    /\ jobPrompt[j] + jobMax[j] + 2 <= MaxCtx
    /\ cached \in 0..(jobPrompt[j] - 1)
    /\ CASE srcKind = "any"  -> TRUE
         [] srcKind = "cold" -> cached = 0 /\ fromLane = l
         [] srcKind = "own"  -> fromLane = l /\ keptPrompt[l] = cached /\ cached > 0
         [] srcKind = "copy" -> /\ fromLane \in LaneIds /\ fromLane # l
                                /\ laneJob[fromLane] # 0
                                /\ lanePos[fromLane] = cached
                                /\ lanePrompt[fromLane] = cached
                                /\ cached > 0
         [] OTHER -> FALSE
    /\ queue'      = Tail(queue)
    /\ laneJob'    = [laneJob EXCEPT ![l] = j]
    /\ lanePos'    = [lanePos EXCEPT ![l] = cached]
    /\ lanePrompt' = [lanePrompt EXCEPT ![l] = jobPrompt[j]]
    /\ laneGen'    = [laneGen EXCEPT ![l] = 0]
    /\ keptPrompt' = [keptPrompt EXCEPT ![l] = 0]
    /\ UNCHANGED <<seenJobs, inFlight, pendingRefusal, jobPrompt, jobMax,
                   disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

BuildStep(mode, spans) ==
    /\ pendingRefusal = 0
    /\ ~stepActive
    /\ spans # {}
    /\ \A sp \in spans :
        /\ sp.lane \in LaneIds
        /\ laneJob[sp.lane] = sp.job
        /\ lanePos[sp.lane] = sp.pos
        /\ sp.tokens >= 1
    /\ \A s1, s2 \in spans : s1.lane = s2.lane => s1 = s2
    /\ CASE mode = "prefill" ->
            /\ HasReadyPrefill
            /\ (~HasReadyDecode \/ ~lastWasPrefill)
            /\ SumSpanTokens(spans) <= ChunkBudget
            /\ \A sp \in spans :
                /\ lanePos[sp.lane] < lanePrompt[sp.lane]
                /\ sp.pos + sp.tokens <= lanePrompt[sp.lane]
            /\ lastWasPrefill' = TRUE
         [] mode = "decode" ->
            /\ HasReadyDecode
            /\ (~HasReadyPrefill \/ lastWasPrefill)
            /\ \A sp \in spans :
                /\ lanePos[sp.lane] >= lanePrompt[sp.lane]
                /\ sp.tokens = 1
                /\ sp.pos + 1 <= MaxCtx
            /\ lastWasPrefill' = FALSE
         [] OTHER -> FALSE
    /\ stepActive' = TRUE
    /\ stepMode'   = mode
    /\ stepSpans'  = spans
    /\ UNCHANGED <<seenJobs, inFlight, pendingRefusal, queue,
                   laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                   jobPrompt, jobMax, disconnected>>

FinishStep ==
    /\ pendingRefusal = 0
    /\ stepActive
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
    /\ UNCHANGED <<seenJobs, inFlight, pendingRefusal, queue,
                   laneJob, lanePrompt, jobPrompt, jobMax, disconnected, lastWasPrefill>>

Disconnect(j) ==
    /\ pendingRefusal = 0
    /\ j \in inFlight
    /\ ~disconnected[j]
    /\ disconnected' = [disconnected EXCEPT ![j] = TRUE]
    /\ inFlight'     = inFlight \ {j}
    /\ UNCHANGED <<seenJobs, pendingRefusal, queue,
                   laneJob, lanePos, lanePrompt, laneGen, keptPrompt,
                   jobPrompt, jobMax, stepActive, stepMode, stepSpans, lastWasPrefill>>

Cancel(j, l, genCnt) ==
    /\ pendingRefusal = 0
    /\ ~stepActive
    /\ l \in LaneIds /\ laneJob[l] = j
    /\ disconnected[j] = TRUE
    /\ laneGen[l] = genCnt
    /\ laneJob'    = [laneJob EXCEPT ![l] = 0]
    /\ lanePos'    = [lanePos EXCEPT ![l] = 0]
    /\ lanePrompt' = [lanePrompt EXCEPT ![l] = 0]
    /\ laneGen'    = [laneGen EXCEPT ![l] = 0]
    /\ inFlight'   = inFlight \ {j}
    /\ UNCHANGED <<seenJobs, pendingRefusal, queue, keptPrompt,
                   jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

Finish(j, l, reason, genCnt) ==
    /\ pendingRefusal = 0
    /\ ~stepActive
    /\ l \in LaneIds /\ laneJob[l] = j
    /\ lanePos[l] >= lanePrompt[l]
    /\ laneGen[l] = genCnt
    /\ reason \in {"length", "stop"}
    /\ (reason = "length" => (genCnt >= jobMax[j] \/ lanePos[l] + 2 >= MaxCtx))
    /\ laneJob'    = [laneJob EXCEPT ![l] = 0]
    /\ lanePos'    = [lanePos EXCEPT ![l] = 0]
    /\ lanePrompt' = [lanePrompt EXCEPT ![l] = 0]
    /\ laneGen'    = [laneGen EXCEPT ![l] = 0]
    /\ inFlight'   = inFlight \ {j}
    /\ UNCHANGED <<seenJobs, pendingRefusal, queue, keptPrompt,
                   jobPrompt, jobMax, disconnected, stepActive, stepMode, stepSpans, lastWasPrefill>>

Next ==
    \/ \E j \in Jobs, p \in 1..MaxCtx, m \in 1..MaxCtx : Arrive(j, p, m)
    \/ \E j \in Jobs : Refuse429(j) \/ RefuseCtx(j) \/ Disconnect(j)
    \/ \E j \in Jobs, l \in LaneIds, c \in 0..(MaxCtx - 1),
          sk \in {"any", "cold", "own", "copy"}, fl \in LaneIds : Admit(j, l, c, sk, fl)
    \/ \E m \in {"prefill", "decode"},
          sps \in ((SUBSET [job: Jobs, lane: LaneIds, pos: 0..MaxCtx, tokens: 1..MaxCtx]) \ {{}}) :
          BuildStep(m, sps)
    \/ FinishStep
    \/ \E j \in Jobs, l \in LaneIds, g \in 0..MaxCtx : Cancel(j, l, g)
    \/ \E j \in Jobs, l \in LaneIds, r \in {"length", "stop"}, g \in 1..MaxCtx : Finish(j, l, r, g)

\* ---- Safety Invariants ----------------------------------------------------

TypeOK ==
    /\ seenJobs \subseteq Jobs
    /\ inFlight \subseteq seenJobs
    /\ pendingRefusal \in ({0} \cup Jobs)
    /\ laneJob \in [LaneIds -> ({0} \cup Jobs)]
    /\ stepActive \in BOOLEAN
    /\ stepMode \in {"none", "prefill", "decode"}
    /\ lastWasPrefill \in BOOLEAN

InFlightBounded ==
    Cardinality(EffectiveInFlight) <= NumLanes + MaxQueue

StepBudgetAndModeValid ==
    stepActive =>
        /\ stepSpans # {}
        /\ (stepMode = "prefill" => SumSpanTokens(stepSpans) <= ChunkBudget)
        /\ (stepMode = "decode" => \A sp \in stepSpans : sp.tokens = 1)

ActiveLanesConsistent ==
    /\ \A l \in LaneIds :
        laneJob[l] # 0 =>
            /\ laneJob[l] \in seenJobs
            /\ (laneJob[l] \in EffectiveInFlight \/ disconnected[laneJob[l]])
            /\ laneJob[l] \notin SeqToSet(queue)
            /\ lanePos[l] <= MaxCtx
    /\ \A l1, l2 \in LaneIds :
        (laneJob[l1] # 0 /\ laneJob[l1] = laneJob[l2]) => l1 = l2

Spec == Init /\ [][Next]_vars

=============================================================================
