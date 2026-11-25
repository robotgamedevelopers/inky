-------------------------- MODULE PoisonGateTrace --------------------------
\* Replays deterministic walks recorded from src/poolgate.rs (`poisoned`,
\* `plans`, `poison_plan`, `apply`, `restore`, `table_slots`, `poison_bytes`)
\* by scripts/poisongate-traces.rs against specs/PoisonGate.tla, including
\* `Take`, `Share`, and `Release` both between steps and inside the poisoned
\* phase (`U64, U179`).
EXTENDS PoisonGate, PoisonGateTraceData, Sequences

VARIABLES trace, at

TS == Traces[trace][at]

Hydrate(s) ==
  /\ table    = [l \in Lanes |-> s.table[l + 1]]
  /\ data     = [x \in Slots |-> s.data[x]]
  /\ truth    = [x \in Slots |-> s.truth[x]]
  /\ saved    = [x \in Slots |-> s.saved[x]]
  /\ stepping = s.stepping
  /\ phase    = s.phase

HydrateNext(s) ==
  /\ table'    = [l \in Lanes |-> s.table[l + 1]]
  /\ data'     = [x \in Slots |-> s.data[x]]
  /\ truth'    = [x \in Slots |-> s.truth[x]]
  /\ saved'    = [x \in Slots |-> s.saved[x]]
  /\ stepping' = s.stepping
  /\ phase'    = s.phase

ReplayInit ==
  /\ trace \in 1..Len(Traces)
  /\ at = 1
  /\ Hydrate(TS)

ReplayNext ==
  /\ at < Len(Traces[trace])
  /\ UNCHANGED trace
  /\ at' = at + 1
  /\ HydrateNext(Traces[trace][at + 1])

Replay == ReplayInit /\ [][ReplayNext]_<<vars, trace, at>>

InitConforms == (at = 1) => Init

DispatchAction(act) ==
  IF act.action = "Take"            THEN Take(act.argLane, act.argSlot)
  ELSE IF act.action = "Share"      THEN Share(act.argLane, act.argSlot)
  ELSE IF act.action = "Release"    THEN Release(act.argLane, act.argSlot)
  ELSE IF act.action = "PoisonStep" THEN PoisonStep(act.argSet)
  ELSE IF act.action = "Write"      THEN Write(act.argLane, act.argSlot)
  ELSE IF act.action = "Restore"    THEN Restore
  ELSE FALSE

StepActionCheck ==
  (at' = at + 1) =>
    \E act \in {Traces[trace][at + 1]} : DispatchAction(act)

ActionConforms == [][StepActionCheck]_<<vars, trace, at>>
=============================================================================
