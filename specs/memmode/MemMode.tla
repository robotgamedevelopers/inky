--------------------------- MODULE MemMode ---------------------------
\* Inky mode's helper (src/memd.rs): a root daemon that raises the GPU wired-memory limit while any inky process is
\* connected and restores the limit it found when the last connection closes. A client's connection closes on a clean
\* exit, a crash, kill -9 or sleep alike: one Close action for all of them. The helper itself may crash at any moment:
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
ASSUME Target \in Values /\ Bug \in {"none", "restore_default", "release_early", "no_persist", "lost_seed", "stale_saved"}
VARIABLES limit,      \* the OS setting iogpu.wired_limit_mb
          saved,      \* the limit the running helper found when it raised it (in memory), or NONE
          disk,       \* the same, in the saved file, or NONE
          conns,      \* the clients connected to the running helper
          alive,      \* the helper is running
          owed,       \* a restore of the saved limit is still owed (not yet confirmed)
          owner       \* what the owner set by hand last, outside inky mode
vars == <<limit, saved, disk, conns, alive, owed, owner>>
Init == /\ limit \in Values /\ owner = limit /\ saved = NONE /\ disk = NONE /\ conns = {} /\ alive = TRUE /\ owed = FALSE

\* the owner changes the limit by hand outside inky mode, while no restore is owed (a stale in-memory save
\* does not block it; a save still owed, or one left only on disk, does — see the header)
OwnerSets(v) == alive /\ conns = {} /\ ~owed /\ (saved # NONE \/ disk = NONE) /\ v \in Values /\ limit' = v /\ owner' = v /\ UNCHANGED <<saved, disk, conns, alive, owed>>
Connect(c) == /\ alive /\ c \notin conns
              /\ IF conns # {} THEN /\ conns' = conns \cup {c} /\ UNCHANGED <<saved, limit, disk, owed>>
                 ELSE \/ \* admitted: the limit found (or the seeded save) is kept on disk, then the limit is raised
                         /\ conns' = {c} /\ limit' = Target /\ owed' = TRUE
                         /\ saved' = IF saved = NONE THEN limit ELSE saved
                         /\ disk' = IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
                      \/ \* refused: the raise moved the limit and rolling it back failed too, so the save is kept
                         \* with no clients, for the helper's exit paths
                         /\ conns' = {} /\ limit' = Target /\ owed' = TRUE
                         /\ saved' = IF saved = NONE THEN limit ELSE saved
                         /\ disk' = IF saved = NONE THEN (IF Bug = "no_persist" THEN NONE ELSE limit) ELSE disk
                      \/ \* refused and forgotten: the raise never moved the limit, or rolling it back is
                         \* confirmed (the limit ends at the value found), so no client is counted and the
                         \* save is forgotten — in memory always, on disk too unless forgetting it fails
                         \* (a kept save that could not be written ends unchanged and is not a separate
                         \* step). Modelled only while no restore is owed at entry; Bug "stale_saved"
                         \* keeps the in-memory save when forgetting the file fails, as it does after a
                         \* confirmed restore.
                         /\ (saved = NONE \/ ~owed) /\ conns' = {} /\ owed' = FALSE
                         /\ limit' = IF saved = NONE THEN limit ELSE saved
                         /\ saved' = NONE /\ disk' = NONE
                      \/ /\ (saved = NONE \/ ~owed) /\ conns' = {} /\ owed' = FALSE
                         /\ limit' = IF saved = NONE THEN limit ELSE saved
                         /\ disk' = IF saved = NONE THEN limit ELSE saved
                         /\ saved' = IF Bug = "stale_saved" THEN (IF saved = NONE THEN limit ELSE saved) ELSE NONE
              /\ UNCHANGED <<alive, owner>>
Close(c) == /\ alive /\ c \in conns /\ conns' = conns \ {c}
            /\ IF conns' # {} /\ Bug # "release_early"
                  THEN UNCHANGED <<limit, saved, disk, owed>>
                  ELSE IF saved = NONE THEN UNCHANGED <<limit, saved, disk, owed>>
                       ELSE \/ UNCHANGED <<limit, saved, disk, owed>>     \* the restore fails: the save is kept
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ saved' = NONE /\ disk' = NONE
                            \/ /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                               /\ owed' = FALSE /\ disk' = disk
                               /\ saved' = IF Bug = "stale_saved" THEN saved ELSE NONE     \* forgetting the file fails; Bug "stale_saved" also keeps the in-memory save
            /\ UNCHANGED <<alive, owner>>
\* the helper dies (kill -9, a panic): its memory and every connection go with it
Crash == alive /\ alive' = FALSE /\ conns' = {} /\ saved' = NONE /\ owed' = FALSE /\ UNCHANGED <<limit, disk, owner>>
\* launchd starts it again: a limit on disk means it died while raised. A failed restore seeds the in-memory
\* save from the file (Bug "lost_seed" is the helper that forgets to); a confirmed one may still fail to
\* forget the file.
Restart == /\ ~alive /\ alive' = TRUE
           /\ IF disk = NONE THEN UNCHANGED <<limit, disk, saved, owed>>
              ELSE \/ /\ limit' = disk /\ saved' = NONE /\ owed' = FALSE /\ (disk' = NONE \/ disk' = disk)
                   \/ /\ UNCHANGED <<limit, disk>> /\ saved' = (IF Bug = "lost_seed" THEN NONE ELSE disk) /\ owed' = (Bug # "lost_seed")
           /\ UNCHANGED <<conns, owner>>
Next == (\E c \in Clients : Connect(c) \/ Close(c)) \/ (\E v \in Values : OwnerSets(v)) \/ Crash \/ Restart
Spec == Init /\ [][Next]_vars /\ WF_vars(Restart) /\ WF_vars(\E c \in Clients : Close(c))

\* in inky mode exactly while someone is connected to a running helper
RaisedWhileConnected == alive /\ conns # {} => limit = Target
\* a running helper with no clients and nothing saved anywhere has given the owner's limit back (a save kept
\* across the idle time means a restore is still pending, not that the limit was forgotten)
RestoredWhenIdle == alive /\ conns = {} /\ saved = NONE /\ disk = NONE => limit = owner
\* the limit found is on disk exactly while it is raised
SavedIsOwner == alive /\ conns # {} => saved = owner /\ disk = owner
\* a save held by a running helper is always a restore still owed; a save left behind by a confirmed
\* restore or a forgotten failed raise (Bug "stale_saved") breaks this
SavedIsOwed == alive /\ saved # NONE => owed
\* whatever happens, once the helper runs with no clients the owner's limit is back, or a restore of it is
\* still pending in memory and has not been silently dropped
EventuallyRestored == <>[](conns = {}) ~> (limit = owner \/ saved # NONE)
=============================================================================
