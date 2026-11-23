# MemMode

The model of inky mode's helper (`src/memd.rs`): it raises the GPU wired-memory limit while any inky process is connected
and restores the limit it found when the last connection closes.

```sh
java -cp tla2tools.jar tlc2.TLC -config MemMode.cfg -workers 1 MemMode.tla          # passes: 24 distinct states
java -cp tla2tools.jar tlc2.TLC -config RestoreDefault.cfg -workers 1 MemMode.tla   # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config ReleaseEarly.cfg -workers 1 MemMode.tla     # witness: RaisedWhileConnected violated
```

- `RestoreDefault`: restoring the macOS default (0) instead of the value found throws away a limit the owner set by hand.
- `ReleaseEarly`: releasing when any client closes leaves the others running without the memory.

TLA+ tools 2.19. Checked with three clients and three limit values; the model is small enough for any machine.
