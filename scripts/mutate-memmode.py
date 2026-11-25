"""Named discriminating controls applied only to temporary production extracts."""
import pathlib
import sys

mutant, directory = sys.argv[1:]
startup_mutants = {"lost-seed", "restore-stale-boot", "stale-same-boot"}
path = pathlib.Path(directory) / ("startup.rs" if mutant in startup_mutants else "mode.rs")
changes = {
    "early-release": ("self.clients -= 1;", "self.clients = 0;"),
    "no-rollback": ("if get() != Some(found) { let _ = set(found); }", "let _ = get();"),
    "no-readback": ("Ok(()) if get() == Some(v) =>", "Ok(()) =>"),
    "lost-seed": ("Mode { saved: seed, clients: 0 }", "Mode { saved: None, clients: 0 }"),
    "restore-stale-boot": (
        "SavedRecord::StaleBoot(old_boot, old_val) => {\n            eprintln!(\"[inky memd] ignoring stale saved limit {old_val} from earlier boot {old_boot}\");\n            if let Err(e) = keep(None) { eprintln!(\"[inky memd] {e}\"); }\n            Ok(Mode::default())\n        }",
        "SavedRecord::StaleBoot(_, _) => restore_on_startup(saved_path, set, keep),",
    ),
    "stale-same-boot": ("if boot_tag != now_boot {", "if true {"),
    "skip-keep": ('keep(Some(found)).map_err(|e| format!("the saved limit could not be written; not raising it: {e}"))?;', "// omitted persistence"),
}
old, new = changes[mutant]
source = path.read_text()
assert source.count(old) == 1, f"mutation marker changed: {mutant}"
path.write_text(source.replace(old, new))
