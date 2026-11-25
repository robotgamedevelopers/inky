--------------------------- MODULE MemMode ---------------------------
\* Inky mode's helper (src/memd.rs): a root daemon that raises the GPU wired-memory limit while any inky process is
\* connected and restores the limit it found when the last connection closes (unless the owner's limit was already at
\* or above Target, in which case the helper leaves it untouched). A client's connection closes on a clean exit, a
\* crash, or kill -9 alike: one Close action for all of them. Local Unix sockets stay open across macOS sleep, so the
\* limit stays held while any client remains connected across sleep. The helper itself may crash at any moment:
\* every connection dies with it, and launchd starts it again. The limit it found is on disk while it is raised, keyed
\* on macOS's immutable per-boot kern.bootsessionuuid, so a helper restarting on the same boot restores it even across
\* wall-clock jumps while a helper starting after an OS reboot ignores and removes a stale pre-reboot file.
\*
\* The syscalls and the saved file are fallible, and the model says so: keeping the saved limit may fail (the raise
\* is refused before it happens), setting the limit may fail after moving it (a refused connect rolls it back, and
\* keeps the save only while the limit is still moved), restoring may fail (the save is kept, in memory and on
\* disk, across the idle time until the next start or SIGTERM attempts it again), and forgetting the file may fail
\* after a confirmed restore (the in-memory save is still cleared, so the next connect reads the owner's limit
\* afresh; the stale file is overwritten by that connect's save). A starting helper whose restore fails seeds
\* its in-memory save from the file, so its first connect restores the seeded limit instead of mistaking the
\* still-raised limit for the found one. The owner changing the limit by hand is modelled while no restore is
\* owed: a stale in-memory save left by a confirmed restore or a stale pre-reboot file does not block it, while a
\* hand change racing an owed restore, or one made while a current-boot saved file remains, is outside the model.
EXTENDS Integers, FiniteSets
NONE == -1     \* nothing saved
CONSTANTS Clients, Values, Target, Bug
ASSUME Target \in Values /\ Bug \in {"none", "restore_default", "release_early", "no_persist", "lost_seed", "stale_saved", "no_rollback", "restore_stale", "stale_boot", "lower_higher"}
VARIABLES limit,      \* the OS setting iogpu.wired_limit_mb
          saved,      \* the limit the running helper found when it raised it (in memory), or NONE
          disk,       \* the same, in the saved file, or NONE
          diskStale,  \* TRUE iff disk # NONE and its boot session UUID belongs to an earlier OS boot
          conns,      \* the clients connected to the running helper
          alive,      \* the helper is running
          owed,       \* a restore of the saved limit is still owed (not yet confirmed)
          rollback,   \* last refused-raise rollback observation; cleared by other operations
          owner       \* what the owner set by hand last, outside inky mode
NoRollback == [needed |-> FALSE, attempted |-> FALSE, restored |-> FALSE, found |-> NONE]
vars == <<limit, saved, disk, diskStale, conns, alive, owed, owner, rollback>>
Init == /\ limit \in Values /\ owner = limit /\ saved = NONE /\ disk = NONE /\ diskStale = FALSE
        /\ conns = {} /\ alive = TRUE /\ owed = FALSE /\ rollback = NoRollback

\* the owner changes the limit by hand outside inky mode, while no restore is owed (a stale pre-reboot file
\* does not block it; a save still owed, or a current-boot file left on disk, does — see the header)
OwnerSets(v) == rollback' = NoRollback /\ alive /\ conns = {} /\ ~owed /\ (saved # NONE \/ disk = NONE \/ diskStale)
                /\ v \in Values /\ limit' = v /\ owner' = v /\ UNCHANGED <<saved, disk, diskStale, conns, alive, owed>>
\* A failed raise can leave either its input or target. Reads may fail, and an
\* attempted rollback can fail before or after assignment. Keep these outcomes
\* explicit: an omitted rollback must not hide inside an arbitrary final limit.
RefusedConnect ==
    LET found == IF saved = NONE THEN limit ELSE saved
        kept == IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
        keptStale == IF saved = NONE THEN FALSE ELSE diskStale
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
                       /\ diskStale' = IF forgotten THEN FALSE ELSE keptStale
                       /\ saved' = IF Bug = "stale_saved" /\ ~forgotten THEN found ELSE NONE
                  ELSE /\ owed' = TRUE /\ saved' = found /\ disk' = kept /\ diskStale' = keptStale
Connect(c) == /\ alive /\ c \notin conns
              /\ IF conns # {} THEN /\ conns' = conns \cup {c} /\ rollback' = NoRollback
                                    /\ UNCHANGED <<saved, limit, disk, diskStale, owed>>
                 ELSE \/ \* failure reading the current value or persisting it refuses before any raise
                         /\ saved = NONE /\ rollback' = NoRollback /\ UNCHANGED <<conns, saved, limit, disk, diskStale, owed>>
                      \/ \* owner's current limit is already at or above Target: admit without lowering or saving
                         /\ saved = NONE /\ limit >= Target /\ Bug # "lower_higher"
                         /\ conns' = {c} /\ rollback' = NoRollback
                         /\ UNCHANGED <<saved, limit, disk, diskStale, owed>>
                      \/ \* admitted: persist the found limit under the current boot UUID before raising
                         /\ (saved # NONE \/ limit < Target \/ Bug = "lower_higher")
                         /\ conns' = {c} /\ limit' = Target /\ owed' = TRUE /\ rollback' = NoRollback
                         /\ saved' = IF saved = NONE THEN limit ELSE saved
                         /\ disk' = IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
                         /\ diskStale' = IF saved = NONE THEN FALSE ELSE diskStale
                      \/ /\ (saved # NONE \/ limit < Target \/ Bug = "lower_higher")
                         /\ RefusedConnect
              /\ UNCHANGED <<alive, owner>>
Close(c) == /\ rollback' = NoRollback /\ alive /\ c \in conns /\ conns' = conns \ {c}
            /\ IF conns' # {} /\ Bug # "release_early"
                  THEN UNCHANGED <<limit, saved, disk, diskStale, owed>>
                  ELSE IF saved = NONE THEN UNCHANGED <<limit, saved, disk, diskStale, owed>>
                       ELSE \/ /\ limit' \in {limit, saved} /\ UNCHANGED <<saved, disk, diskStale, owed>>
                               \* failed or unconfirmed restore may have moved the limit: keep the save
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ saved' = NONE /\ disk' = NONE /\ diskStale' = FALSE
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ UNCHANGED <<disk, diskStale>>
                               /\ saved' = IF Bug = "stale_saved" THEN saved ELSE NONE     \* forgetting the file fails; Bug "stale_saved" also keeps the in-memory save
            /\ UNCHANGED <<alive, owner>>
\* the helper dies on the same boot (kill -9, a panic): its memory and every connection go with it; boot UUID is unchanged
Crash == rollback' = NoRollback /\ alive /\ alive' = FALSE /\ conns' = {} /\ saved' = NONE /\ owed' = FALSE
         /\ UNCHANGED <<limit, disk, diskStale, owner>>
\* macOS reboots into a new boot session UUID with kernel limit v: any file left on disk is now from an earlier boot
Reboot(v) == rollback' = NoRollback /\ v \in Values /\ alive' = FALSE /\ conns' = {} /\ saved' = NONE /\ owed' = FALSE
             /\ limit' = v /\ owner' = v /\ UNCHANGED disk /\ diskStale' = (disk # NONE)
\* launchd starts the helper (restore_on_startup_for_boot): a current-boot file on disk restores or seeds Mode.saved;
\* an earlier-boot file (diskStale) is discarded without touching the new boot's limit.
Restart == /\ rollback' = NoRollback /\ ~alive /\ alive' = TRUE
           /\ IF disk = NONE THEN UNCHANGED <<limit, disk, diskStale, saved, owed>>
              ELSE IF diskStale
                   THEN IF Bug = "restore_stale"
                        THEN /\ limit' = disk /\ saved' = NONE /\ owed' = FALSE /\ disk' = NONE /\ diskStale' = FALSE
                        ELSE /\ UNCHANGED limit /\ saved' = NONE /\ owed' = FALSE
                             /\ ((disk' = NONE /\ diskStale' = FALSE) \/ UNCHANGED <<disk, diskStale>>)
                   ELSE IF Bug = "stale_boot"
                        THEN /\ UNCHANGED limit /\ saved' = NONE /\ owed' = FALSE /\ disk' = NONE /\ diskStale' = FALSE
                        ELSE \/ /\ limit' = disk /\ saved' = NONE /\ owed' = FALSE
                                /\ ((disk' = NONE /\ diskStale' = FALSE) \/ UNCHANGED <<disk, diskStale>>)
                             \/ /\ limit' \in {limit, disk} /\ UNCHANGED <<disk, diskStale>>
                                /\ saved' = (IF Bug = "lost_seed" THEN NONE ELSE disk) /\ owed' = (Bug # "lost_seed")
           /\ UNCHANGED <<conns, owner>>
Next == (\E c \in Clients : Connect(c) \/ Close(c)) \/ (\E v \in Values : OwnerSets(v) \/ Reboot(v)) \/ Crash \/ Restart
Spec == Init /\ [][Next]_vars /\ WF_vars(Restart) /\ WF_vars(\E c \in Clients : Close(c))
\* Reversed liveness witness (`UnfairRestart.cfg`): omitting weak fairness of
\* `Restart` lets a crashed helper stay down forever with the limit raised.
SpecUnfair == Init /\ [][Next]_vars /\ WF_vars(\E c \in Clients : Close(c))

\* in inky mode while someone is connected: raised to Target, or left at the owner's higher limit
RaisedWhileConnected == alive /\ conns # {} => limit = IF owner >= Target THEN owner ELSE Target
\* a running helper with no clients and no current-boot save has given the owner's limit back (a save kept
\* across the idle time means a restore is still pending, not that the limit was forgotten)
RestoredWhenIdle == alive /\ conns = {} /\ saved = NONE /\ (disk = NONE \/ diskStale) => limit = owner
\* the limit found is on disk under the current boot UUID while inky mode raised it; an already-sufficient owner limit needs no raise or save
SavedIsOwner == alive /\ conns # {} =>
    IF owner < Target THEN saved = owner /\ disk = owner /\ ~diskStale
    ELSE saved = NONE /\ (disk = NONE \/ diskStale)
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
\* whatever happens, once no clients remain connected, the owner's limit is back (via close, restart, or reboot),
\* or a restore of it is still pending in memory and has not been silently dropped
EventuallyRestored == <>[](conns = {}) ~> (limit = owner \/ saved # NONE)
=============================================================================
