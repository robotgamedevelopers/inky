"""Named discriminating controls applied only to temporary production extracts."""
import pathlib
import sys

mutant, directory = sys.argv[1:]
path = pathlib.Path(directory) / ("startup.rs" if mutant == "lost-seed" else "mode.rs")
changes = {
    "early-release": ("self.clients -= 1;", "self.clients = 0;"),
    "no-rollback": ("if get() != Some(found) { let _ = set(found); }", "let _ = get();"),
    "no-readback": ("Ok(()) if get() == Some(v) =>", "Ok(()) =>"),
    "lost-seed": ("Mode { saved: seed, clients: 0 }", "Mode { saved: None, clients: 0 }"),
    "skip-keep": ('keep(Some(found)).map_err(|e| format!("the saved limit could not be written; not raising it: {e}"))?;', "// omitted persistence"),
    "lower-higher": ("if found >= target {", "if false && found >= target {"),
}
old, new = changes[mutant]
source = path.read_text()
assert source.count(old) == 1, f"mutation marker changed: {mutant}"
path.write_text(source.replace(old, new))
