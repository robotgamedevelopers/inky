# MemMode

The model of inky mode's helper (`src/memd.rs`): it raises the GPU wired-memory limit while any inky process is connected
and restores the limit it found when the last connection closes.

```sh
java -cp tla2tools.jar tlc2.TLC -config MemMode.cfg -workers 1 MemMode.tla          # passes: 30 distinct states, with the liveness property
java -cp tla2tools.jar tlc2.TLC -config RestoreDefault.cfg -workers 1 MemMode.tla   # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config ReleaseEarly.cfg -workers 1 MemMode.tla     # witness: RaisedWhileConnected violated
java -cp tla2tools.jar tlc2.TLC -config NoPersist.cfg -workers 1 MemMode.tla        # witness: RestoredWhenIdle violated
```

- `RestoreDefault`: restoring the macOS default (0) instead of the value found throws away a limit the owner set by hand.
- `ReleaseEarly`: releasing when any client closes leaves the others running without the memory.
- `NoPersist`: a helper that keeps the limit it found only in memory, then dies while it is raised, leaves the limit raised for good once launchd restarts it.

What the model covers: clients connecting and closing (however a connection closes), the owner setting the limit outside inky mode, the helper crashing at any moment and launchd restarting it, the saved limit on disk; safety, and the liveness that the owner's limit comes back. What it does not: the socket and its permissions, a sysctl call failing (the helper refuses to raise a limit it cannot read), the choice of target, or a check that the Rust follows the model step by step (there is no trace check for the helper; its unit tests in src/memd.rs cover the same transitions).

TLA+ tools 2.19. Checked with three clients and three limit values; small enough for any machine.
