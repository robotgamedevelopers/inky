--------------------------- MODULE MemMode ---------------------------
\* Inky mode's helper (src/memd.rs): a root daemon that raises the GPU wired-memory limit while any inky process is
\* connected and restores the limit it found when the last connection closes. A connection closes on a clean exit, a
\* crash, kill -9 or sleep alike: the model has one Close action for all of them.
EXTENDS Naturals, FiniteSets
CONSTANTS Clients, Values, Target, Bug
ASSUME Target \in Values /\ Bug \in {"none", "restore_default", "release_early"}
VARIABLES limit,      \* the OS setting iogpu.wired_limit_mb
          saved,      \* the limit the helper found when it raised it (or "none")
          conns,      \* the set of connected clients
          owner       \* what the owner set by hand last, before or between inky sessions
vars == <<limit, saved, conns, owner>>
Init == /\ limit \in Values /\ owner = limit /\ saved = "none" /\ conns = {}

\* the owner changes the limit by hand while nothing is connected
OwnerSets(v) == conns = {} /\ v \in Values /\ limit' = v /\ owner' = v /\ UNCHANGED <<saved, conns>>
Connect(c) == /\ c \notin conns /\ conns' = conns \cup {c}
              /\ IF conns = {} THEN saved' = limit /\ limit' = Target ELSE UNCHANGED <<saved, limit>>
              /\ UNCHANGED owner
Close(c) == /\ c \in conns /\ conns' = conns \ {c}
            /\ IF conns' = {} \/ Bug = "release_early"
                  THEN /\ limit' = IF Bug = "restore_default" THEN 0 ELSE saved
                       /\ saved' = "none"
                  ELSE UNCHANGED <<saved, limit>>
            /\ UNCHANGED owner
Next == (\E c \in Clients : Connect(c) \/ Close(c)) \/ (\E v \in Values : OwnerSets(v))
Spec == Init /\ [][Next]_vars /\ WF_vars(\E c \in Clients : Close(c))

\* in inky mode exactly while someone is connected
RaisedWhileConnected == conns # {} => limit = Target
\* out of inky mode, the limit is what the owner left it at
RestoredWhenIdle == conns = {} => limit = owner /\ saved = "none"
\* the helper remembers the owner's value for exactly as long as it holds the limit
SavedIsOwner == conns # {} => saved = owner
=============================================================================
