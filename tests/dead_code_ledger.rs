use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> =
        fs::read_dir(dir).expect("read src directory").map(|e| e.expect("dir entry").path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn actual_allow_dead_code_sites(repo_root: &Path) -> BTreeSet<(String, String)> {
    let mut files = Vec::new();
    collect_rs_files(&repo_root.join("src"), &mut files);
    let mut sites = BTreeSet::new();
    for path in files {
        let rel = path.strip_prefix(repo_root).unwrap().to_string_lossy().replace('\\', "/");
        let content = fs::read_to_string(&path).expect("read rust source");
        let lines: Vec<&str> = content.lines().collect();
        for (idx, raw) in lines.iter().enumerate() {
            let trimmed = raw.trim();
            if trimmed.starts_with("//") || !trimmed.contains("allow(dead_code)") {
                continue;
            }
            if trimmed.starts_with("#![") {
                sites.insert((rel.clone(), "#![allow(dead_code)]".to_string()));
                continue;
            }
            // Find the annotated item's first non-attribute, non-comment signature line.
            let mut target = None;
            for next in &lines[idx + 1..] {
                let nt = next.trim();
                if nt.is_empty() || nt.starts_with("//") || nt.starts_with("#[") {
                    continue;
                }
                target = Some(nt.to_string());
                break;
            }
            let target =
                target.unwrap_or_else(|| panic!("{rel}:{}: allow(dead_code) has no following item line", idx + 1));
            sites.insert((rel.clone(), target));
        }
    }
    sites
}

fn parse_ledger_sites(ledger_md: &str) -> BTreeSet<(String, String)> {
    let start_marker = "<!-- DEAD_CODE_LEDGER_START -->";
    let end_marker = "<!-- DEAD_CODE_LEDGER_END -->";
    let start =
        ledger_md.find(start_marker).expect("missing DEAD_CODE_LEDGER_START in docs/DEAD-CODE.md") + start_marker.len();
    let end = ledger_md[start..].find(end_marker).expect("missing DEAD_CODE_LEDGER_END in docs/DEAD-CODE.md") + start;
    let mut sites = BTreeSet::new();
    for line in ledger_md[start..end].lines() {
        let t = line.trim();
        if !t.starts_with('|') || t.contains("| File |") || t.starts_with("|---") {
            continue;
        }
        let cols: Vec<&str> = t.split('|').map(str::trim).collect();
        assert!(cols.len() >= 6, "malformed ledger row: {t}");
        let file = cols[1].trim_matches('`').to_string();
        let target = cols[2].trim_matches('`').to_string();
        let category = cols[3].trim_matches('`');
        let reason = cols[4];
        assert!(!file.is_empty() && !target.is_empty(), "empty file/target in row: {t}");
        assert!(
            matches!(category, "trace-reference" | "kept-spec" | "kept-api" | "held-open-pr"),
            "unexpected dead-code category '{category}' in row: {t}"
        );
        assert!(!reason.is_empty(), "missing reason in row: {t}");
        assert!(sites.insert((file.clone(), target.clone())), "duplicate ledger entry for ({file}, {target})");
    }
    sites
}

#[test]
fn every_allow_dead_code_in_src_matches_docs_dead_code_ledger() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ledger_path = root.join("docs/DEAD-CODE.md");
    let ledger_md = fs::read_to_string(&ledger_path).expect("read docs/DEAD-CODE.md");
    let expected = parse_ledger_sites(&ledger_md);
    let actual = actual_allow_dead_code_sites(&root);

    let undocumented: Vec<_> = actual.difference(&expected).collect();
    let stale_in_doc: Vec<_> = expected.difference(&actual).collect();
    assert!(
        undocumented.is_empty() && stale_in_doc.is_empty(),
        "allow(dead_code) drift detected!\n  Undocumented in src/: {undocumented:#?}\n  Stale in docs/DEAD-CODE.md: {stale_in_doc:#?}"
    );
}
