//! Tunings: one machine's settings, matched by chip and memory. Each is a file in tunings/ (compiled in); contributors
//! add theirs. A machine with no tuning runs on defaults derived from its memory.

pub struct Tuning {
    pub name: String,
    pub chip: String,
    pub memory_gb: u64,
    pub wired_limit_mb: u64,
    pub lane_margin_gib: f64,
    pub default_lanes: usize,
    /// Whether Metal 4 / MPP `tensor_ops` are enabled (`false` on Metal 3 GPUs like Apple M3 Ultra).
    pub use_mpp: bool,
    /// Target share (percent) of KV pool budget allocated to sliding-window blocks (remainder goes to Full).
    pub kv_window_pct: u64,
    /// Default decode attention split-K target when `INK_SPLITS` is unset.
    pub decode_splits: usize,
}

const FILES: &[&str] = &[
    include_str!("../tunings/m5-max-128gb.tuning"),
    include_str!("../tunings/m5-pro-64gb.tuning"),
    include_str!("../tunings/m3-ultra-192gb.tuning"),
    include_str!("../tunings/m3-ultra-512gb.tuning"),
];

/// Parse a tuning file: `key = value` lines, `#` comments. Refuses a file missing a key.
pub fn parse(text: &str) -> Result<Tuning, String> {
    let mut kv = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() { continue; }
        let (k, v) = line.split_once('=').ok_or(format!("not key = value: {line}"))?;
        kv.insert(k.trim().to_string(), v.trim().to_string());
    }
    let get = |k: &str| kv.get(k).cloned().ok_or(format!("missing {k}"));
    let num = |k: &str| get(k)?.parse::<f64>().map_err(|e| format!("{k}: {e}"));
    let default_lanes = match kv.get("default_lanes") {
        Some(v) => v.parse::<usize>().map_err(|e| format!("default_lanes: {e}"))?,
        None => 64,
    };
    let use_mpp = match kv.get("use_mpp").map(String::as_str) {
        Some("true") => true,
        Some("false") => false,
        Some(other) => return Err(format!("use_mpp: expected true or false, got {other:?}")),
        None => true,
    };
    let kv_window_pct = match kv.get("kv_window_pct") {
        Some(v) => {
            let pct = v.parse::<u64>().map_err(|e| format!("kv_window_pct: {e}"))?;
            if !(1..=45).contains(&pct) {
                return Err(format!("kv_window_pct: expected 1..=45, got {pct}"));
            }
            pct
        }
        None => 15,
    };
    let decode_splits = match kv.get("decode_splits") {
        Some(v) => {
            let splits = v.parse::<usize>().map_err(|e| format!("decode_splits: {e}"))?;
            if !(1..=256).contains(&splits) {
                return Err(format!("decode_splits: expected 1..=256, got {splits}"));
            }
            splits
        }
        None => 8,
    };
    Ok(Tuning { name: get("name")?, chip: get("chip")?, memory_gb: num("memory_gb")? as u64, wired_limit_mb: num("wired_limit_mb")? as u64,
                lane_margin_gib: num("lane_margin_gib")?, default_lanes, use_mpp, kv_window_pct, decode_splits })
}

/// The tuning for a machine with this chip and memory, or the defaults for its memory.
pub fn select(chip: &str, memory_bytes: u64) -> Tuning {
    let gb = (memory_bytes as f64 / (1u64 << 30) as f64).round() as u64;
    for f in FILES {
        let t = parse(f).expect("a tuning in tunings/ does not parse");
        if t.chip == chip && t.memory_gb == gb { return t; }
    }
    // defaults: leave max(8 GB, 10%) for macOS
    let mb = memory_bytes / (1 << 20);
    let use_mpp = !chip.starts_with("Apple M1")
        && !chip.starts_with("Apple M2")
        && !chip.starts_with("Apple M3")
        && !chip.starts_with("Apple M4");
    Tuning { name: "default".into(), chip: chip.into(), memory_gb: gb, wired_limit_mb: mb.saturating_sub((mb / 10).max(8192)),
             lane_margin_gib: 4.0, default_lanes: 64, use_mpp, kv_window_pct: 15, decode_splits: 8 }
}

/// This machine's chip name and memory, from sysctl.
pub fn this_machine() -> (String, u64) {
    (sysctl_string("machdep.cpu.brand_string").unwrap_or_default(), sysctl_u64("hw.memsize").unwrap_or(0))
}

pub fn sysctl_string(name: &str) -> Option<String> {
    let n = std::ffi::CString::new(name).ok()?;
    let mut len: libc::size_t = 0;
    unsafe {
        if libc::sysctlbyname(n.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) != 0 { return None; }
        let mut buf = vec![0u8; len];
        if libc::sysctlbyname(n.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) != 0 { return None; }
        Some(String::from_utf8_lossy(&buf[..len.saturating_sub(1)]).to_string())
    }
}

pub fn sysctl_u64(name: &str) -> Option<u64> {
    let n = std::ffi::CString::new(name).ok()?;
    let mut v: u64 = 0;
    let mut len: libc::size_t = 8;
    let r = unsafe { libc::sysctlbyname(n.as_ptr(), &mut v as *mut u64 as *mut libc::c_void, &mut len, std::ptr::null_mut(), 0) };
    if r != 0 { return None; }
    Some(if len == 4 { v & 0xffff_ffff } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn every_tuning_parses() { for f in FILES { parse(f).unwrap(); } }
    #[test] fn selects_the_macbook_pro() {
        let t = select("Apple M5 Max", 128 << 30);
        assert_eq!((t.name.as_str(), t.wired_limit_mb, t.default_lanes, t.use_mpp, t.kv_window_pct, t.decode_splits),
                   ("m5-max-128gb", 118000, 64, true, 15, 8));
        let p = select("Apple M5 Pro", 64 << 30);
        assert_eq!((p.name.as_str(), p.wired_limit_mb, p.default_lanes, p.use_mpp, p.kv_window_pct, p.decode_splits),
                   ("m5-pro-64gb", 57344, 32, true, 15, 8));
    }
    #[test] fn selects_m3_ultra_across_192gb_and_512gb() {
        let u192 = select("Apple M3 Ultra", 192 << 30);
        assert_eq!(u192.name, "m3-ultra-192gb");
        assert_eq!(u192.chip, "Apple M3 Ultra");
        assert_eq!(u192.memory_gb, 192);
        assert_eq!(u192.wired_limit_mb, 180224);
        assert_eq!(u192.lane_margin_gib, 4.0);
        assert_eq!(u192.default_lanes, 64);
        assert!(!u192.use_mpp);
        assert_eq!(u192.kv_window_pct, 20);
        assert_eq!(u192.decode_splits, 16);

        let u512 = select("Apple M3 Ultra", 512 << 30);
        assert_eq!(u512.name, "m3-ultra-512gb");
        assert_eq!(u512.chip, "Apple M3 Ultra");
        assert_eq!(u512.memory_gb, 512);
        assert_eq!(u512.wired_limit_mb, 491520);
        assert_eq!(u512.lane_margin_gib, 4.0);
        assert_eq!(u512.default_lanes, 128);
        assert!(!u512.use_mpp);
        assert_eq!(u512.kv_window_pct, 15);
        assert_eq!(u512.decode_splits, 16);

        // Unmatched M3 memory size falls back safely to non-MPP defaults with OS headroom.
        let u96 = select("Apple M3 Max", 96 << 30);
        assert_eq!(u96.name, "default");
        assert!(!u96.use_mpp);
        assert_eq!(u96.lane_margin_gib, 4.0);
    }
    #[test] fn rejects_invalid_optional_tuning_fields() {
        let base = "name = x\nchip = y\nmemory_gb = 64\nwired_limit_mb = 57344\nlane_margin_gib = 4\n";
        assert!(parse(&format!("{base}use_mpp = maybe\n")).is_err());
        assert!(parse(&format!("{base}kv_window_pct = 0\n")).is_err());
        assert!(parse(&format!("{base}kv_window_pct = 46\n")).is_err());
        assert!(parse(&format!("{base}decode_splits = 0\n")).is_err());
        assert!(parse(&format!("{base}decode_splits = 257\n")).is_err());
    }
    #[test] fn defaults_leave_room_for_macos() {
        let t = select("Apple M9", 64 << 30);
        assert_eq!(t.name, "default");
        assert_eq!(t.wired_limit_mb, 65536 - 8192);
    }
    #[test] fn default_tuning_saturates_to_zero_on_small_or_unreadable_memory() {
        for bytes in [0u64, 512 << 20, 4 << 30, 8 << 30] {
            let t = select("Unknown Mac", bytes);
            assert_eq!(t.name, "default");
            assert_eq!(t.wired_limit_mb, 0, "memory_bytes={bytes} must saturate wired_limit_mb to 0 without underflow");
        }
        let nine_gib = select("Unknown Mac", 9 << 30);
        assert_eq!(nine_gib.wired_limit_mb, 9216 - 8192);
    }
    #[test] fn refuses_a_missing_key() { assert!(parse("name = x\nchip = y\n").is_err()); }
}
