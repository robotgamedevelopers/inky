# MemMode

The model of inky mode's helper (`src/memd.rs`): it raises the GPU wired-memory limit while any inky process is connected
and restores the limit it found when the last connection closes.

```sh
java -cp tla2tools.jar tlc2.TLC -config MemMode.cfg -workers 1 MemMode.tla          # passes: 40 distinct states, with the liveness property
java -cp tla2tools.jar tlc2.TLC -config RestoreDefault.cfg -workers 1 MemMode.tla   # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config ReleaseEarly.cfg -workers 1 MemMode.tla     # witness: RaisedWhileConnected violated
java -cp tla2tools.jar tlc2.TLC -config NoPersist.cfg -workers 1 MemMode.tla        # witness: RestoredWhenIdle violated
java -cp tla2tools.jar tlc2.TLC -config LostSeed.cfg -workers 1 MemMode.tla        # witness: SavedIsOwner violated
java -cp tla2tools.jar tlc2.TLC -config StaleSaved.cfg -workers 1 MemMode.tla       # witness: RestoredWhenIdle violated
```

- `RestoreDefault`: restoring the macOS default (0) instead of the value found throws away a limit the owner set by hand.
- `ReleaseEarly`: releasing when any client closes leaves the others running without the memory.
- `NoPersist`: a helper that keeps the limit it found only in memory, then dies while it is raised, leaves the limit raised for good once launchd restarts it.
- `LostSeed`: a helper that restarts after a failed startup restore without seeding its in-memory save from the saved file: its first connect reads the still-raised limit as the found one and overwrites the file with it, so the owner's limit is never restored.
- `StaleSaved`: a helper that keeps its in-memory save after a confirmed restore, or after a failed raise it should have forgotten (when forgetting the file fails): the owner then changes the limit by hand, and the next connect skips reading it — an admitted one later restores the stale value over the owner's, a refused one rolls back to it. The witness checks the owner's limit (`RestoredWhenIdle`), so TLC runs past the stale state itself (which also breaks `SavedIsOwed`, checked in the main config) to the overwrite.

What the model covers: clients connecting and closing (however a connection closes), the owner setting the limit outside inky mode while no restore is owed, the helper crashing at any moment and launchd restarting it, the saved limit on disk, and the fallible steps — writing the saved file, setting the limit (including a raise that moves it and is then rolled back, and one that is forgotten once the rollback is confirmed), restoring, forgetting the file, and a starting helper seeding its save from the file when its restore fails. The save is held across the idle time while a restore is still owed. Safety, and the liveness that the owner's limit comes back or a restore of it is still pending and has not been silently dropped. What it does not: the socket and its permissions, a hand change of the limit racing an owed restore, or one made while only the saved file remains (the helper's saved value wins there by design until its restore is confirmed or the next connect overwrites the file), the choice of target, or the real socket/sysctl integration. The CPU trace check below connects the actual Rust Mode transitions to the model.

TLA+ tools 2.19. Checked with three clients and three limit values; small enough for any machine.


## Production Mode trace conformance

Run `TLA2TOOLS_JAR=/path/to/tla2tools.jar scripts/check-memmode-traces.sh` from the repository root. It needs Java, `rustc`, Python 3 and Bash, with no Metal, root privileges or system-setting changes. The CPU TLA runner invokes it beside the model checks.

The script extracts `Mode` and the daemon startup restore block and its actual `Mode` initialization expression directly from `src/memd.rs` (bounded by exact source markers, failing if those move), compiles them with `scripts/memmode-traces.rs`, and records 32 reproducible walks of 192 operations. Every walk begins with an unfailing two-client connect/close sequence, then randomizes connects, closes, failures, permitted idle owner changes, crashes and restarts. A crash clears process memory and clients; restart executes the production startup code against an actual temporary saved file. The fake injects unreadable gets, failed sets before or after assignment, failed saved-file writes/removal, and successful sets without assignment during close. Connect/startup retain the production `set_limit` contract that success confirms the write; close's extra read-back is tested with the weaker injected setter.

Each state includes the operation and the ordered closure calls, arguments, results and assignment effects. `TraceProtocol.tla` ties the observed calls to the corresponding success/failure branch, including rollback, read-back and startup seeding, and checks the resulting OS/disk values against the recorded effects. `owed` is tracked independently from `Mode.saved`: successful persistence/raising creates an obligation; a confirmed restore releases it even when forgetting the file fails. Startup derives the obligation from the observed restore result. Closing maps the client count to `{1, ..., clients}`.

`TraceCheck.tla` replays every recorded state and checks initial/next-state conformance from **MemMode itself**, its safety invariants, and the operation protocol. This checks finite prefixes, including simulated crash/restart and the real startup restore block, but does not establish infinite fairness/liveness, socket handling, process supervision, or real syscall behavior. The separate exhaustive model run checks the model's liveness.

Five committed negative controls mutate temporary extracts of the production Rust: early release, omitted rollback, omitted close read-back, lost startup seed, and omitted persistence. Each must produce a TLC action-property or invariant counterexample (exit 13 or 12); compiler, parser or infrastructure failures cannot count as successful controls. The generator separately requires coverage of all six fault classes, failed saved-file restarts, and successful restore calls that leave the limit at the wrong value.

The initial traces exposed existing implementation behaviors missing from the model: a failed raise/read-back can retain a save while the original limit is intact, an unconfirmed restore can move the limit, and a previously owed save can be restored by rollback. Production Rust is unchanged. The main model and all five model bug witnesses remain discriminating.
