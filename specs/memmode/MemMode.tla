--------------------------- MODULE MemMode ---------------------------
\* Inky mode's helper (src/memd.rs): a root daemon that raises the GPU wired-memory limit while any inky process is
\* connected and restores the limit it found when the last connection closes (unless the owner's limit was already at
\* or above Target, in which case the helper leaves it untouched). A client's connection closes on a clean exit, a
\* crash, or kill -9 alike: one Close action for all of them. Local Unix sockets stay open across macOS sleep, so the
\* limit stays held while any client remains connected across sleep. The helper itself may crash at any moment:
\* every connection dies with it, and launchd starts it again. The limit it found is on disk while it is raised, and a
\* starting helper restores a limit it finds there.
\*
\* The syscalls and the saved file are fallible, and the model says so: keeping the saved limit may fail (the raise
\* is refused before it happens), setting the limit may fail after moving it (a refused connect rolls it back, and
\* keeps the save only while the limit is still moved), restoring may fail (the save is kept, in memory and on
\* disk, across the idle time until the next start or SIGTERM attempts it again), and forgetting the file may fail
\* after a confirmed restore (the in-memory save is still cleared, so the next connect reads the owner's limit
\* afresh; the stale file is overwritten by that connect's save). A starting helper whose restore fails seeds
\* its in-memory save from the file, so its first connect restores the seeded limit instead of mistaking the
\* still-raised limit for the found one. The owner changing the limit by hand is modelled while no restore is
\* owed: a stale in-memory save left by a confirmed restore does not block it (the running helper's next connect
\* is what overwrites it), while a hand change racing an owed restore, or one made while only the saved file
\* remains, is outside the model — there the helper's saved value wins by design until its restore is confirmed
\* or the next connect overwrites the file.
EXTENDS Integers, FiniteSets
NONE == -1     \* nothing saved
CONSTANTS Clients, Values, Target, Bug
ASSUME Target \in Values /\ Bug \in {"none", "restore_default", "release_early", "no_persist", "lost_seed", "stale_saved", "no_rollback", "lower_higher"}
VARIABLES limit,      \* the OS setting iogpu.wired_limit_mb
          saved,      \* the limit the running helper found when it raised it (in memory), or NONE
          disk,       \* the same, in the saved file, or NONE
          conns,      \* the clients connected to the running helper
          alive,      \* the helper is running
          owed,       \* a restore of the saved limit is still owed (not yet confirmed)
          rollback,   \* last refused-raise rollback observation; cleared by other operations
          owner       \* what the owner set by hand last, outside inky mode
NoRollback == [needed |-> FALSE, attempted |-> FALSE, restored |-> FALSE, found |-> NONE]
vars == <<limit, saved, disk, conns, alive, owed, owner, rollback>>
Init == /\ limit \in Values /\ owner = limit /\ saved = NONE /\ disk = NONE /\ conns = {} /\ alive = TRUE /\ owed = FALSE /\ rollback = NoRollback

\* the owner changes the limit by hand outside inky mode, while no restore is owed (a stale in-memory save
\* does not block it; a save still owed, or one left only on disk, does — see the header)
OwnerSets(v) == rollback' = NoRollback /\ alive /\ conns = {} /\ ~owed /\ (saved # NONE \/ disk = NONE) /\ v \in Values /\ limit' = v /\ owner' = v /\ UNCHANGED <<saved, disk, conns, alive, owed>>
\* A failed raise can leave either its input or target. Reads may fail, and an
\* attempted rollback can fail before or after assignment. Keep these outcomes
\* explicit: an omitted rollback must not hide inside an arbitrary final limit.
RefusedConnect ==
    LET found == IF saved = NONE THEN limit ELSE saved
        kept == IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
    IN \E raised \in {limit, Target} :
       \E readRaised \in {NONE, raised}, writes \in BOOLEAN :
        LET needed == readRaised # found
            attempted == needed /\ Bug # "no_rollback"
            restored == attempted /\ writes
            after == IF restored THEN found ELSE raised
        IN \E readFinal \in {NONE, after}, forgotten \in BOOLEAN :
            /\ limit' = after /\ conns' = {}
            /\ rollback' = [needed |-> needed, attempted |-> attempted,
                             restored |-> restored, found |-> found]
            /\ IF readFinal = found
                  THEN /\ owed' = FALSE
                       /\ disk' = IF forgotten THEN NONE ELSE kept
                       /\ saved' = IF Bug = "stale_saved" /\ ~forgotten THEN found ELSE NONE
                  ELSE /\ owed' = TRUE /\ saved' = found /\ disk' = kept
Connect(c) == /\ alive /\ c \notin conns
              /\ IF conns # {} THEN /\ conns' = conns \cup {c} /\ rollback' = NoRollback
                                    /\ UNCHANGED <<saved, limit, disk, owed>>
                 ELSE \/ \* failure reading the current value or persisting it refuses before any raise
                         /\ saved = NONE /\ rollback' = NoRollback /\ UNCHANGED <<conns, saved, limit, disk, owed>>
                      \/ \* owner's current limit is already at or above Target: admit without lowering or saving
                         /\ saved = NONE /\ limit >= Target /\ Bug # "lower_higher"
                         /\ conns' = {c} /\ rollback' = NoRollback
                         /\ UNCHANGED <<saved, limit, disk, owed>>
                      \/ \* admitted: persist the found limit before raising
                         /\ (saved # NONE \/ limit < Target \/ Bug = "lower_higher")
                         /\ conns' = {c} /\ limit' = Target /\ owed' = TRUE /\ rollback' = NoRollback
                         /\ saved' = IF saved = NONE THEN limit ELSE saved
                         /\ disk' = IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
                      \/ /\ (saved # NONE \/ limit < Target \/ Bug = "lower_higher")
                         /\ RefusedConnect
              /\ UNCHANGED <<alive, owner>>
Close(c) == /\ rollback' = NoRollback /\ alive /\ c \in conns /\ conns' = conns \ {c}
            /\ IF conns' # {} /\ Bug # "release_early"
                  THEN UNCHANGED <<limit, saved, disk, owed>>
                  ELSE IF saved = NONE THEN UNCHANGED <<limit, saved, disk, owed>>
                       ELSE \/ /\ limit' \in {limit, saved} /\ UNCHANGED <<saved, disk, owed>>
                               \* failed or unconfirmed restore may have moved the limit: keep the save
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ saved' = NONE /\ disk' = NONE
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ disk' = disk
                               /\ saved' = IF Bug = "stale_saved" THEN saved ELSE NONE     \* forgetting the file fails; Bug "stale_saved" also keeps the in-memory save
            /\ UNCHANGED <<alive, owner>>
\* the helper dies (kill -9, a panic): its memory and every connection go with it
Crash == rollback' = NoRollback /\ alive /\ alive' = FALSE /\ conns' = {} /\ saved' = NONE /\ owed' = FALSE /\ UNCHANGED <<limit, disk, owner>>
\* launchd starts it again: a limit on disk means it died while raised. A failed restore seeds the in-memory
\* save from the file (Bug "lost_seed" is the helper that forgets to); a confirmed one may still fail to
\* forget the file.
Restart == /\ rollback' = NoRollback /\ ~alive /\ alive' = TRUE
           /\ IF disk = NONE THEN UNCHANGED <<limit, disk, saved, owed>>
              ELSE \/ /\ limit' = disk /\ saved' = NONE /\ owed' = FALSE /\ (disk' = NONE \/ disk' = disk)
                   \/ /\ limit' \in {limit, disk} /\ UNCHANGED disk /\ saved' = (IF Bug = "lost_seed" THEN NONE ELSE disk) /\ owed' = (Bug # "lost_seed")
           /\ UNCHANGED <<conns, owner>>
Next == (\E c \in Clients : Connect(c) \/ Close(c)) \/ (\E v \in Values : OwnerSets(v)) \/ Crash \/ Restart
Spec == Init /\ [][Next]_vars /\ WF_vars(Restart) /\ WF_vars(\E c \in Clients : Close(c))

\* in inky mode while someone is connected: raised to Target, or left at the owner's higher limit
RaisedWhileConnected == alive /\ conns # {} => limit = IF owner >= Target THEN owner ELSE Target
\* a running helper with no clients and nothing saved anywhere has given the owner's limit back (a save kept
\* across the idle time means a restore is still pending, not that the limit was forgotten)
RestoredWhenIdle == alive /\ conns = {} /\ saved = NONE /\ disk = NONE => limit = owner
\* the limit found is on disk while inky mode raised it; an already-sufficient owner limit needs no raise or save
SavedIsOwner == alive /\ conns # {} =>
    IF owner < Target THEN saved = owner /\ disk = owner
    ELSE saved = NONE /\ disk = NONE
\* a save held by a running helper is always a restore still owed; a save left behind by a confirmed
\* restore or a forgotten failed raise (Bug "stale_saved") breaks this
SavedIsOwed == alive /\ saved # NONE => owed
\* Every refused raise not confirmed at the saved value attempts rollback.
\* A write that assigns the saved value restores it; a failed write can leave a
\* moved limit only with the restore obligation retained. Crashes and failed
\* final-close restores make an unconditional no-clients => not-raised false.
RollbackAttempted == rollback.needed => rollback.attempted
RefusedRollbackSafe ==
    /\ (rollback.restored => limit = rollback.found)
    /\ (rollback.needed /\ limit # rollback.found =>
          rollback.attempted /\ saved = rollback.found /\ owed)
\* whatever happens, once the helper runs with no clients the owner's limit is back, or a restore of it is
\* still pending in memory and has not been silently dropped
EventuallyRestored == <>[](conns = {}) ~> (limit = owner \/ saved # NONE)
=============================================================================
