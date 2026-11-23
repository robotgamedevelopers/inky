# MemMode

The model of inky mode's helper (`src/memd.rs`): it raises the GPU wired-memory limit while any inky process is connected
and restores the limit it found when the last connection closes.

```sh
java -cp tla2tools.jar tlc2.TLC -config MemMode.cfg -workers 1 MemMode.tla          # passes: 40 distinct states, with the liveness property
java -cp tla2tools.jar tlc2.TLC -config RestoreDefault.cfg -workers 1 MemMode.tla   # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config ReleaseEarly.cfg -workers 1 MemMode.tla     # witness: RaisedWhileConnected violated
java -cp tla2tools.jar tlc2.TLC -config NoPersist.cfg -workers 1 MemMode.tla        # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config LostSeed.cfg -workers 1 MemMode.tla        # witness: SavedIsOwner violated
java -cp tla2tools.jar tlc2.TLC -config StaleSaved.cfg -workers 1 MemMode.tla       # witness: SavedIsOwed violated
```

- `RestoreDefault`: restoring the macOS default (0) instead of the value found throws away a limit the owner set by hand.
- `ReleaseEarly`: releasing when any client closes leaves the others running without the memory.
- `NoPersist`: a helper that keeps the limit it found only in memory, then dies while it is raised, leaves the limit raised for good once launchd restarts it.
- `LostSeed`: a helper that restarts after a failed startup restore without seeding its in-memory save from the saved file: its first connect reads the still-raised limit as the found one and overwrites the file with it, so the owner's limit is never restored.
- `StaleSaved`: a helper that keeps its in-memory save after a confirmed restore (when forgetting the file fails): the save is no longer owed a restore, yet the next connect would skip reading the owner's current limit and later restore the stale value over it.

What the model covers: clients connecting and closing (however a connection closes), the owner setting the limit outside inky mode while no save is pending, the helper crashing at any moment and launchd restarting it, the saved limit on disk, and the fallible steps — writing the saved file, setting the limit (including a raise that moves it and is then rolled back), restoring, forgetting the file, and a starting helper seeding its save from the file when its restore fails. The save is held across the idle time while a restore is still owed. Safety, and the liveness that the owner's limit comes back or a restore of it is still pending and has not been silently dropped. What it does not: the socket and its permissions, a hand change of the limit racing a pending restore (the helper's saved value wins there by design until its restore is confirmed), the choice of target, or a check that the Rust follows the model step by step (there is no trace check for the helper; its unit tests in src/memd.rs cover the same transitions).

TLA+ tools 2.19. Checked with three clients and three limit values; small enough for any machine.
