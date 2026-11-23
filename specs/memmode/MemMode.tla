--------------------------- MODULE MemMode ---------------------------
\* Inky mode's helper (src/memd.rs): a root daemon that raises the GPU wired-memory limit while any inky process is
\* connected and restores the limit it found when the last connection closes. A client's connection closes on a clean
\* exit, a crash, kill -9 or sleep alike: one Close action for all of them. The helper itself may crash at any moment:
\* every connection dies with it, and launchd starts it again. The limit it found is on disk while it is raised, and a
\* starting helper restores a limit it finds there.
EXTENDS Integers, FiniteSets
NONE == -1     \* nothing saved
CONSTANTS Clients, Values, Target, Bug
ASSUME Target \in Values /\ Bug \in {"none", "restore_default", "release_early", "no_persist"}
VARIABLES limit,      \* the OS setting iogpu.wired_limit_mb
          saved,      \* the limit the running helper found when it raised it (in memory), or NONE
          disk,       \* the same, in the saved file, or NONE
          conns,      \* the clients connected to the running helper
          alive,      \* the helper is running
          owner       \* what the owner set by hand last, outside inky mode
vars == <<limit, saved, disk, conns, alive, owner>>
Init == /\ limit \in Values /\ owner = limit /\ saved = NONE /\ disk = NONE /\ conns = {} /\ alive = TRUE

\* the owner changes the limit by hand outside inky mode
OwnerSets(v) == alive /\ conns = {} /\ v \in Values /\ limit' = v /\ owner' = v /\ UNCHANGED <<saved, disk, conns, alive>>
Connect(c) == /\ alive /\ c \notin conns /\ conns' = conns \cup {c}
              /\ IF conns = {} THEN /\ saved' = limit /\ limit' = Target
                                    /\ disk' = IF Bug = "no_persist" THEN NONE ELSE limit
                               ELSE UNCHANGED <<saved, limit, disk>>
              /\ UNCHANGED <<alive, owner>>
Close(c) == /\ alive /\ c \in conns /\ conns' = conns \ {c}
            /\ IF conns' = {} \/ Bug = "release_early"
                  THEN /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                       /\ saved' = NONE /\ disk' = NONE
                  ELSE UNCHANGED <<saved, limit, disk>>
            /\ UNCHANGED <<alive, owner>>
\* the helper dies (kill -9, a panic): its memory and every connection go with it
Crash == alive /\ alive' = FALSE /\ conns' = {} /\ saved' = NONE /\ UNCHANGED <<limit, disk, owner>>
\* launchd starts it again: a limit on disk means it died while raised
Restart == /\ ~alive /\ alive' = TRUE
           /\ IF disk # NONE THEN limit' = disk /\ disk' = NONE ELSE UNCHANGED <<limit, disk>>
           /\ UNCHANGED <<saved, conns, owner>>
Next == (\E c \in Clients : Connect(c) \/ Close(c)) \/ (\E v \in Values : OwnerSets(v)) \/ Crash \/ Restart
Spec == Init /\ [][Next]_vars /\ WF_vars(Restart) /\ WF_vars(\E c \in Clients : Close(c))

\* in inky mode exactly while someone is connected to a running helper
RaisedWhileConnected == alive /\ conns # {} => limit = Target
\* a running helper with no clients has given the owner's limit back
RestoredWhenIdle == alive /\ conns = {} => limit = owner /\ saved = NONE /\ disk = NONE
\* the limit found is on disk exactly while it is raised
SavedIsOwner == alive /\ conns # {} => saved = owner /\ disk = owner
\* whatever happens, the owner's limit comes back once the helper runs with no clients
EventuallyRestored == <>[](conns = {}) ~> (limit = owner)
=============================================================================
