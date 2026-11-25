mod blocks;
mod cache_scheduler;
mod cache_trace;
#[allow(dead_code)]
mod blockbook;
mod engine;
mod gguf;
#[allow(dead_code)] // not yet called by the scheduler; the pool and table workers land next
mod prefix;
mod metal;
mod sampler;
mod scheduler;
mod session;
mod memd;
mod poolgate;
mod record;
mod serve;
mod spill;
mod stream_text;
mod tuning;
mod tokenizer;
mod unicode_tables;
mod verify;

use engine::{Engine, Lane};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use sampler::Sampler;
use session::{Renderer, Session};
use std::io::{BufRead, Write};
use std::net::ToSocketAddrs;
use tokenizer::Tokenizer;

const DEFAULT_MODEL: &str = "~/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf";
const CONTEXT_ALIGNMENT: usize = 64;
// Inkling 2 Mini's eight full-attention slots address 16,384 K/V elements per token with u32 offsets.
const MAX_CONTEXT: usize = ((u32::MAX as usize - 1) / 16_384 / CONTEXT_ALIGNMENT) * CONTEXT_ALIGNMENT;

const USAGE: &str = "inky - a hand-fused Metal inference engine for Thinking Machines Inkling 2 Mini

usage: inky <command> [options]

commands:
  chat              interactive multi-turn chat (KV cache reused across turns)
  run -p TEXT       one prompt, streamed
  bench             greedy prefill + decode benchmark
  check --ref P     compare logits against a llama.cpp reference dump (tools/ref_logits)
  serve             OpenAI Chat Completions and Anthropic Messages (see inky serve --help)

options:
  -m, --model PATH  GGUF file (default: LM Studio's Inkling-2-Mini Q4_K_M)
  -p, --prompt TEXT prompt text (run/bench)      -f FILE  read the prompt from a file
  -n N              max new tokens (default 2048, bench 128; 0 skips generation)
  -c, --ctx N       context length (default 16384, maximum 262080, rounded up to a multiple of 64)
  --effort E        reasoning effort: none|minimal|low|medium|high|xhigh|<0..0.99> (default high)
  --system TEXT     system prompt
  --temp T          temperature (default 1.0; 0 = greedy, fully GPU-pipelined)
  --top-p P, --top-k K, --seed S
  --greedy          same as --temp 0
  --hide-thinking   do not print the reasoning stream
  --kv-disk         keep only half of the (exact) K/V cache in memory while generating; the older half lives in a
                    temporary file and is read back for every token (slower generation)
  --kv8             8-bit K/V cache: half the cache memory, NOT exact (the default fp16 cache is)
  --loopcd OMEGA    adaptive LoopCD-Logits contrast weight in [0, 4] (default 0 = disabled)
  --raw             do not apply the chat template; print special tokens";


const SERVE_USAGE: &str = "usage: inky serve [options]

  -m, --model PATH  GGUF file (default: LM Studio's Inkling-2-Mini Q4_K_M)
  -c, --ctx N       context per lane (default 65536, maximum 262080, rounded up to a multiple of 64)
  --bind ADDR       listen address (default 127.0.0.1:8090)
  --key-env NAME    use a nonempty key from the named environment variable
  --api-key-env NAME
                    alias for --key-env
  --allow-unauthenticated-non-loopback
                    explicitly allow a non-loopback bind without a key
  --lanes N         active inference lanes (defaults to this machine's tuning)
  --queue N         waiting requests (default 64)
  --effort E        default reasoning effort (default medium)
  --thinking-budget N
                    hard cap on thinking tokens before forcing <|end_message|> (default 0 = unlimited)
  --loopcd OMEGA    default adaptive LoopCD-Logits contrast weight in [0, 4] (default 0 = disabled)
  --record-dir DIR  record requests and response metrics as cards-YYYYMMDD.jsonl in DIR
  --record-max-bytes N
                    max bytes per segment before rotating to .1 (default 104857600); larger rows are dropped
  --spill-dir DIR   directory for lazy LRU SSD block spill (mode 0700; files 0600)
  --max-spill-bytes N
                    maximum bytes kept in --spill-dir before LRU disk GC (default 64 GiB)
  --kv8             8-bit K/V cache: doubles BlockPool capacity and halves KV bandwidth (NOT exact)
  -n MAX            positive default Chat Completions reply budget (default 16384)";

fn cli_error(error: String) -> ! {
    eprintln!("{error}");
    std::process::exit(2);
}

fn report_cli_result(command: &str, result: Result<(), String>) {
    if let Err(err) = result {
        eprintln!("{command}: {err}");
        std::process::exit(1);
    }
}

fn wait_gate(cb: &metal::CmdBuf, stage: &str, pos: usize) {
    if let Err(err) = metal::wait_completed(cb, stage, pos) {
        eprintln!("[gate] {err}");
        std::process::exit(1);
    }
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("{flag} requires a nonnegative integer"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value.parse().map_err(|_| format!("{flag} requires a nonnegative integer"))
}

fn parse_u64_positive(flag: &str, value: &str) -> Result<u64, String> {
    let v: u64 = value.parse().map_err(|_| format!("{flag} requires a positive integer"))?;
    if v == 0 {
        return Err(format!("{flag} must be greater than 0"));
    }
    Ok(v)
}

fn parse_context(flag: &str, value: &str) -> Result<usize, String> {
    let ctx = parse_usize(flag, value)?;
    if !(1..=MAX_CONTEXT).contains(&ctx) {
        return Err(format!("{flag} must be between 1 and {MAX_CONTEXT} tokens"));
    }
    Ok(ctx.div_ceil(CONTEXT_ALIGNMENT) * CONTEXT_ALIGNMENT)
}

fn parse_loopcd_omega(flag: &str, value: &str) -> Result<f32, String> {
    let omega: f32 = value
        .parse()
        .map_err(|_| format!("{flag} must be a finite number between 0 and 4"))?;
    if !omega.is_finite() || !(0.0..=4.0).contains(&omega) {
        return Err(format!("{flag} must be a finite number between 0 and 4"));
    }
    Ok(omega)
}

fn parse_temp(flag: &str, value: &str) -> Result<f32, String> {
    let temp: f32 = value
        .parse()
        .map_err(|_| format!("{flag} must be a finite nonnegative number"))?;
    if !temp.is_finite() || temp < 0.0 {
        return Err(format!("{flag} must be a finite nonnegative number"));
    }
    Ok(if temp == 0.0 { 0.0 } else { temp })
}

fn parse_top_p(flag: &str, value: &str) -> Result<f32, String> {
    let top_p: f32 = value
        .parse()
        .map_err(|_| format!("{flag} must be a finite number in (0, 1]"))?;
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(format!("{flag} must be a finite number in (0, 1]"));
    }
    Ok(top_p)
}

fn effort_value(e: &str) -> Result<f64, String> {
    let trimmed = e.trim();
    match trimmed {
        "none" => Ok(0.0),
        "minimal" => Ok(0.1),
        "low" => Ok(0.2),
        "medium" => Ok(0.7),
        "high" => Ok(0.9),
        "xhigh" | "max" => Ok(0.99),
        _ => {
            let v: f64 = trimmed.parse().map_err(|_| {
                format!("--effort must be none|minimal|low|medium|high|xhigh|max or a finite number in [0, 1], got {e:?}")
            })?;
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(format!("--effort must be none|minimal|low|medium|high|xhigh|max or a finite number in [0, 1], got {e:?}"));
            }
            Ok(v.clamp(0.0, 0.99))
        }
    }
}

fn parse_effort(flag: &str, value: &str) -> Result<String, String> {
    let _ = effort_value(value).map_err(|e| e.replacen("--effort", flag, 1))?;
    Ok(value.trim().to_string())
}

fn is_known_cli_command(cmd: &str) -> bool {
    matches!(
        cmd,
        "help" | "-h" | "--help"
            | "check" | "rows" | "rollback" | "verify" | "binv" | "crosspath" | "lanes" | "spans" | "xcopy" | "poolgate"
            | "loopcdgate" | "blockedinv" | "temp1" | "prefixbench"
            | "lanebench" | "dbatch" | "sharedkv" | "maxctx" | "debug" | "prof" | "routes" | "vcost" | "prof_pf"
            | "kmoe" | "kgemm" | "kmoepf" | "ksg" | "qclip" | "lazytest" | "kvprune" | "kvscreen" | "kvdump"
            | "kpeak" | "kmode" | "bw" | "kbench" | "run" | "chat" | "bench"
    )
}

fn check_prompt_fits_ctx(prompt_tokens: usize, ctx: usize) -> Result<(), String> {
    if prompt_tokens == 0 {
        return Err("prompt encoded to 0 tokens".into());
    }
    if prompt_tokens >= ctx {
        return Err(format!(
            "prompt of {prompt_tokens} tokens exceeds context headroom ({} max prompt tokens for -c {ctx}); pass a larger -c/--ctx",
            ctx.saturating_sub(1)
        ));
    }
    Ok(())
}

fn entropy_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    nanos ^ (std::process::id() as u64).rotate_left(32)
}

fn resolve_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(entropy_seed)
}

fn default_model() -> String {
    DEFAULT_MODEL.replace('~', &std::env::var("HOME").unwrap_or_default())
}

fn parse_serve(args: &[String], default_lanes: usize, read_key: impl Fn(&str) -> Result<String, std::env::VarError>) -> Result<serve::ServeOpts, String> {
    let mut o = serve::ServeOpts {
        model: default_model(), ctx: 65536, bind: Vec::new(), lanes: default_lanes, effort: "medium".into(),
        max_tokens: 16384, key: None, queue: 64, thinking_budget: 0, loopcd_omega: 0.0, record_dir: None,
        record_max_bytes: record::DEFAULT_MAX_FILE_BYTES,
        spill_dir: None, max_spill_bytes: spill::DEFAULT_MAX_SPILL_BYTES,
        kv8: false,
    };
    let mut bind = "127.0.0.1:8090";
    let mut allow_unauthenticated = false;
    let mut args = args.iter().map(String::as_str);
    while let Some(flag) = args.next() {
        let mut value = || args.next().filter(|value| !value.is_empty() && !value.starts_with("--") && !matches!(*value, "-m" | "-c" | "-n" | "-h"))
            .ok_or_else(|| format!("missing value for {flag}"));
        match flag {
            "-m" | "--model" => o.model = value()?.into(),
            "-c" | "--ctx" => o.ctx = parse_context(flag, value()?)?,
            "--bind" => bind = value()?,
            "--lanes" => o.lanes = parse_usize(flag, value()?)?,
            "--effort" => o.effort = parse_effort(flag, value()?)?,
            "-n" => o.max_tokens = parse_usize(flag, value()?)?,
            "--key-env" | "--api-key-env" => {
                let name = value()?;
                let key = read_key(name).map_err(|error| match error {
                    std::env::VarError::NotPresent => format!("{flag} {name}: variable is not set"),
                    std::env::VarError::NotUnicode(_) => format!("{flag} {name}: variable must contain valid UTF-8"),
                })?;
                if key.is_empty() || key.trim() != key || key.bytes().any(|b| b.is_ascii_control()) {
                    return Err(format!("{flag} {name} must contain a nonempty HTTP header value without surrounding whitespace or control characters"));
                }
                o.key = Some(key);
            }
            "--queue" => o.queue = parse_usize(flag, value()?)?,
            "--thinking-budget" => o.thinking_budget = parse_usize(flag, value()?)?,
            "--loopcd" => o.loopcd_omega = parse_loopcd_omega(flag, value()?)?,
            "--record-dir" => o.record_dir = Some(std::path::PathBuf::from(value()?)),
            "--record-max-bytes" => o.record_max_bytes = parse_u64_positive(flag, value()?)?,
            "--spill-dir" => o.spill_dir = Some(std::path::PathBuf::from(value()?)),
            "--max-spill-bytes" => o.max_spill_bytes = parse_u64(flag, value()?)?,
            "--kv8" => o.kv8 = true,
            "--allow-unauthenticated-non-loopback" => allow_unauthenticated = true,
            _ => return Err(format!("unknown serve option {flag}\n\n{SERVE_USAGE}")),
        }
    }
    o.bind = bind.to_socket_addrs().map_err(|e| format!("cannot resolve --bind {bind}: {e}"))?.collect();
    serve::validate_bind_addrs(&o.bind, o.key.is_some(), allow_unauthenticated)?;
    Ok(o)
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn loopcd_weak_logits_require_both_finite_references() {
        assert_eq!(weak_loopcd_mismatch(&[1.0, 2.0], &[1.25, 2.0], &[1.0, 2.0]), Some(0.25));
        assert_eq!(weak_loopcd_mismatch(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.25]), Some(0.25));
        assert_eq!(weak_loopcd_mismatch(&[1.0], &[f32::NAN], &[1.0]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[1.0], &[1.0], &[f32::NAN]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[], &[], &[]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[f32::NAN], &[1.0], &[1.0]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[f32::INFINITY], &[f32::INFINITY], &[f32::INFINITY]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[1.0, 2.0], &[1.0], &[1.0, 2.0]), Some(f32::INFINITY));
        assert_eq!(weak_loopcd_mismatch(&[1.0], &[1.0000001], &[0.9999999]), None);
    }

    fn serve_args(args: &[&str]) -> Result<serve::ServeOpts, String> {
        parse_serve(&args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(), 64, |name| match name {
            "PRESENT" => Ok("configured-key".into()),
            "EMPTY" => Ok(String::new()),
            "BLANK" => Ok(" \t".into()),
            "NONUNICODE" => Err(std::env::VarError::NotUnicode("secret data".into())),
            _ => Err(std::env::VarError::NotPresent),
        })
    }

    #[test]
    fn serving_uses_machine_lane_default_unless_overridden() {
        let lanes = tuning::select("Apple M5 Pro", 64 << 30).default_lanes;
        assert_eq!(parse_serve(&[], lanes, |_| unreachable!()).unwrap().lanes, 32);
        let args = ["--lanes".to_string(), "8".to_string()];
        assert_eq!(parse_serve(&args, lanes, |_| unreachable!()).unwrap().lanes, 8);
    }

    #[test]
    fn cli_context_limit_applies_before_rounding() {
        assert_eq!(MAX_CONTEXT, 262080);
        assert_eq!(parse_context("--ctx", "1"), Ok(64));
        assert_eq!(parse_context("--ctx", "262079"), Ok(MAX_CONTEXT));
        assert_eq!(parse_context("--ctx", "262080"), Ok(MAX_CONTEXT));
        for invalid in ["0", "262081", "262144", "1000000", &usize::MAX.to_string(), "text"] {
            assert!(parse_context("--ctx", invalid).is_err());
        }
    }

    #[test]
    fn serve_rejects_incomplete_or_invalid_arguments() {
        for (args, message) in [
            (&["--key-env"][..], "missing value for --key-env"),
            (&["--key-env", "--lanes", "1"], "missing value for --key-env"),
            (&["--key-env", "ABSENT"], "--key-env ABSENT"),
            (&["--key-env", "EMPTY"], "nonempty HTTP header"),
            (&["--key-env", "BLANK", "--allow-unauthenticated-non-loopback"], "nonempty HTTP header"),
            (&["--key-env", "NONUNICODE"], "variable must contain valid UTF-8"),
            (&["--api-key-env"], "missing value for --api-key-env"),
            (&["--api-key-env", "ABSENT"], "--api-key-env ABSENT"),
            (&["--api-key-env", "EMPTY"], "nonempty HTTP header"),
            (&["--api-key-env", "NONUNICODE"], "variable must contain valid UTF-8"),
            (&["--bind"], "missing value for --bind"),
            (&["--record-dir"], "missing value for --record-dir"),
            (&["--record-max-bytes"], "missing value for --record-max-bytes"),
            (&["--record-max-bytes", "0"], "--record-max-bytes must be greater than 0"),
            (&["--record-max-bytes", "abc"], "--record-max-bytes requires a positive integer"),
            (&["--spill-dir"], "missing value for --spill-dir"),
            (&["--max-spill-bytes"], "missing value for --max-spill-bytes"),
            (&["--max-spill-bytes", "bad"], "--max-spill-bytes requires a nonnegative integer"),
            (&["--lanes", "bad"], "--lanes requires a nonnegative integer"),
            (&["--ctx", "262144"], "--ctx must be between 1 and 262080"),
            (&["--loopcd"], "missing value for --loopcd"),
            (&["--loopcd", "-0.1"], "--loopcd must be a finite number between 0 and 4"),
            (&["--loopcd", "nan"], "--loopcd must be a finite number between 0 and 4"),
            (&["--loopcd", "inf"], "--loopcd must be a finite number between 0 and 4"),
            (&["--loopcd", "-inf"], "--loopcd must be a finite number between 0 and 4"),
            (&["--loopcd", "5.0"], "--loopcd must be a finite number between 0 and 4"),
            (&["--loopcd", "abc"], "--loopcd must be a finite number between 0 and 4"),
            (&["--typo"], "unknown serve option --typo"),
        ] {
            let err = serve_args(args).err().unwrap();
            assert!(err.contains(message), "{args:?}: {err}");
            assert!(!err.contains("secret data"));
        }
    }

    #[test]
    fn serve_checks_the_addresses_it_will_bind() {
        assert_eq!(serve_args(&[]).unwrap().lanes, 64);
        assert_eq!(serve_args(&[]).unwrap().record_dir, None);
        assert_eq!(serve_args(&[]).unwrap().record_max_bytes, record::DEFAULT_MAX_FILE_BYTES);
        assert_eq!(serve_args(&["--record-dir", "cards-out"]).unwrap().record_dir, Some(std::path::PathBuf::from("cards-out")));
        assert_eq!(serve_args(&["--record-max-bytes", "4096"]).unwrap().record_max_bytes, 4096);
        assert_eq!(serve_args(&[]).unwrap().spill_dir, None);
        assert_eq!(serve_args(&[]).unwrap().max_spill_bytes, spill::DEFAULT_MAX_SPILL_BYTES);
        assert!(!serve_args(&[]).unwrap().kv8);
        assert!(serve_args(&["--kv8"]).unwrap().kv8);
        let spilled = serve_args(&["--spill-dir", "spill-out", "--max-spill-bytes", "1048576", "--kv8"]).unwrap();
        assert_eq!(spilled.spill_dir, Some(std::path::PathBuf::from("spill-out")));
        assert_eq!(spilled.max_spill_bytes, 1_048_576);
        assert!(spilled.kv8);
        assert_eq!(serve_args(&[]).unwrap().effort, "medium");
        assert_eq!(serve_args(&["--effort", "high"]).unwrap().effort, "high");
        assert_eq!(serve_args(&[]).unwrap().bind, ["127.0.0.1:8090".parse::<std::net::SocketAddr>().unwrap()]);
        assert_eq!(serve_args(&["--bind", "[::1]:0"]).unwrap().bind, ["[::1]:0".parse::<std::net::SocketAddr>().unwrap()]);
        assert!(serve_args(&["--bind", "[::ffff:127.0.0.1]:0"]).is_ok());
        for bind in [
            "10.0.0.1:0",
            "172.16.0.1:0",
            "192.168.1.1:0",
            "100.64.0.1:18096",
            "169.254.1.1:0",
            "[fd7a:115c:a1e0::1]:18096",
            "[fe80::1]:0",
            "[::ffff:100.64.0.1]:0",
        ] {
            let args = ["--bind", bind];
            assert!(serve_args(&args).err().unwrap().contains("a non-loopback --bind requires"), "{bind}");
            let keyed = serve_args(&["--bind", bind, "--key-env", "PRESENT"]).unwrap();
            assert_eq!(keyed.bind, [bind.parse::<std::net::SocketAddr>().unwrap()]);
            assert_eq!(keyed.key.as_deref(), Some("configured-key"));
            let api_keyed = serve_args(&["--bind", bind, "--api-key-env", "PRESENT"]).unwrap();
            assert_eq!(api_keyed.bind, keyed.bind);
            assert_eq!(api_keyed.key.as_deref(), Some("configured-key"));
            let unkeyed = serve_args(&["--bind", bind, "--allow-unauthenticated-non-loopback"]).unwrap();
            assert_eq!(unkeyed.bind, keyed.bind);
            assert!(unkeyed.key.is_none());
        }
        for rejected in [
            "0.0.0.0:0",
            "0.0.0.0:8096",
            "[::]:0",
            "[::]:8096",
            "8.8.8.8:8096",
            "192.0.2.1:0",
            "[::ffff:192.0.2.1]:0",
            "[::ffff:0.0.0.0]:0",
            "[2001:4860:4860::8888]:8096",
        ] {
            for extra in [&[][..], &["--key-env", "PRESENT"][..], &["--allow-unauthenticated-non-loopback"][..]] {
                let mut argv = vec!["--bind", rejected];
                argv.extend_from_slice(extra);
                let err = serve_args(&argv).err().unwrap_or_else(|| panic!("expected {argv:?} to be refused"));
                assert!(err.contains("neither loopback nor a private"), "{argv:?}: {err}");
            }
        }
        assert!(serve::is_private_ip("100.64.0.1".parse().unwrap()));
    }

    #[test]
    fn loopcd_cli_and_serve_flag_parsing() {
        assert_eq!(serve_args(&[]).unwrap().loopcd_omega, 0.0);
        assert_eq!(serve_args(&["--loopcd", "0"]).unwrap().loopcd_omega, 0.0);
        assert_eq!(serve_args(&["--loopcd", "0.3"]).unwrap().loopcd_omega, 0.3);
        assert_eq!(serve_args(&["--loopcd", "4.0"]).unwrap().loopcd_omega, 4.0);

        let cli = |args: &[&str]| {
            parse_cli_args(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
        };
        assert_eq!(cli(&["run", "-p", "hi"]).unwrap().1.loopcd_omega, 0.0);
        assert_eq!(cli(&["run", "--loopcd", "0"]).unwrap().1.loopcd_omega, 0.0);
        assert_eq!(cli(&["run", "--loopcd", "0.3"]).unwrap().1.loopcd_omega, 0.3);
        assert_eq!(cli(&["chat", "--loopcd", "1.5"]).unwrap().1.loopcd_omega, 1.5);
        assert_eq!(cli(&["bench", "--loopcd", "4.0"]).unwrap().1.loopcd_omega, 4.0);
        for bad in ["-0.1", "nan", "NaN", "inf", "-inf", "4.01", "5.0", "bad"] {
            let err = cli(&["run", "--loopcd", bad]).err().unwrap();
            assert!(err.contains("--loopcd must be a finite number between 0 and 4"), "{bad}: {err}");
        }
        assert!(cli(&["run", "--loopcd"]).err().unwrap().contains("missing value for --loopcd"));
        assert!(cli(&["run", "--loopcd", "0.3", "--kv-disk"]).err().unwrap().contains("--kv-disk"));
    }

    #[test]
    fn cli_validates_sampling_effort_seed_zero_kv_disk_and_subcommands() {
        let cli = |args: &[&str]| {
            parse_cli_args(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
        };

        // --temp and --top-p bounds + NaN/Inf rejection before loading weights
        assert_eq!(cli(&["run", "--temp", "0"]).unwrap().1.temp, 0.0);
        assert_eq!(cli(&["run", "--temp", "-0.0"]).unwrap().1.temp, 0.0);
        assert_eq!(cli(&["run", "--temp", "1.25"]).unwrap().1.temp, 1.25);
        for bad_temp in ["-0.01", "-1", "nan", "NaN", "inf", "-inf", "abc"] {
            assert!(cli(&["run", "--temp", bad_temp]).is_err(), "accepted bad --temp {bad_temp}");
        }
        assert_eq!(cli(&["run", "--top-p", "1.0"]).unwrap().1.top_p, 1.0);
        assert_eq!(cli(&["run", "--top-p", "0.05"]).unwrap().1.top_p, 0.05);
        for bad_top_p in ["0", "0.0", "-0.5", "1.01", "nan", "NaN", "inf", "-inf", "abc"] {
            assert!(cli(&["run", "--top-p", bad_top_p]).is_err(), "accepted bad --top-p {bad_top_p}");
        }

        // --effort accepts named tiers and finite [0, 1] numbers, rejecting NaN, typos, and out-of-range floats
        for (eff, rendered) in [
            ("none", "0"),
            ("minimal", "0.1"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
            ("0", "0"),
            ("0.5", "0.5"),
            ("1.0", "0.99"),
        ] {
            let (_, o) = cli(&["run", "--effort", eff]).unwrap();
            assert_eq!(effort_str(&o.effort), rendered);
            assert!(serve_args(&["--effort", eff]).is_ok());
        }
        for bad_eff in ["nan", "NaN", "inf", "-inf", "-0.1", "1.01", "meduim", "hig", "turbo"] {
            assert!(cli(&["run", "--effort", bad_eff]).is_err(), "CLI accepted bad --effort {bad_eff}");
            assert!(serve_args(&["--effort", bad_eff]).is_err(), "serve accepted bad --effort {bad_eff}");
        }

        // --seed 0 is preserved as an explicit deterministic seed, distinct from omitted --seed
        assert_eq!(cli(&["run"]).unwrap().1.seed, None);
        assert_eq!(cli(&["run", "--seed", "0"]).unwrap().1.seed, Some(0));
        assert_eq!(resolve_seed(Some(0)), 0);
        let mut s0_a = Sampler::new(1.0, 8, 0.95, 20.0, resolve_seed(Some(0)));
        let mut s0_b = Sampler::new(1.0, 8, 0.95, 20.0, resolve_seed(Some(0)));
        let row = [1.0f32, 1.2, 0.9, 1.1, 0.8, 1.15];
        for _ in 0..16 {
            assert_eq!(s0_a.sample_next(&row), s0_b.sample_next(&row));
        }

        // --kv-disk + --kv8 refused cleanly at CLI parse
        assert!(cli(&["run", "--kv-disk", "--kv8"]).err().unwrap().contains("--kv-disk cannot be combined with --kv8"));
        assert!(cli(&["check", "--kv8", "--kv-disk"]).err().unwrap().contains("--kv-disk cannot be combined with --kv8"));

        // Unknown subcommands fail with error instead of printing usage with exit 0
        for unknown in ["bogus", "sever", "runs", "--unknown-first"] {
            let err = cli(&[unknown]).err().unwrap();
            assert!(err.contains("unknown command"), "{unknown}: {err}");
        }
        assert_eq!(cli(&[]).unwrap().0, "help");
        assert_eq!(cli(&["--help"]).unwrap().0, "help");
        assert_eq!(cli(&["crosspath", "--ref", "ref/long.tokens"]).unwrap().0, "crosspath");

        // Context headroom check refuses prompts >= ctx cleanly
        assert!(check_prompt_fits_ctx(0, 64).is_err());
        assert!(check_prompt_fits_ctx(63, 64).is_ok());
        assert!(check_prompt_fits_ctx(64, 64).is_err());
        assert!(check_prompt_fits_ctx(120, 64).is_err());
    }
}

struct Opts {
    model: String, prompt: String, n: usize, ctx: usize, raw: bool, effort: String, reference: String, quiet: bool, stop: usize,
    system: String, temp: f32, top_p: f32, top_k: usize, seed: Option<u64>, hide_thinking: bool, n_set: bool,
    loopcd_omega: f32, kv8: bool, kv_disk: bool, allow_mismatch: bool,
}

fn parse_cli_args(raw_args: &[String]) -> Result<(String, Opts), String> {
    let mut args = raw_args.iter().map(String::as_str);
    let raw_cmd = args.next().unwrap_or("help");
    if !is_known_cli_command(raw_cmd) {
        return Err(format!("unknown command {raw_cmd}\n\n{USAGE}"));
    }
    let cmd = if matches!(raw_cmd, "help" | "-h" | "--help") { "help".to_string() } else { raw_cmd.to_string() };
    let mut o = Opts { model: default_model(), prompt: "Write a haiku about the ocean.".into(), n: 2048, ctx: 16384,
                       raw: false, effort: "high".into(), reference: String::new(), quiet: false, stop: 1, system: String::new(),
                       temp: 1.0, top_p: 0.95, top_k: 64, seed: None, hide_thinking: false, n_set: false,
                       loopcd_omega: 0.0, kv8: false, kv_disk: false,
                       allow_mismatch: std::env::var("INK_ALLOW_MISMATCH").is_ok_and(|v| !v.is_empty() && v != "0") };
    while let Some(a) = args.next() {
        let mut v = || args.next().ok_or_else(|| format!("missing value for {a}"));
        match a {
            "-m" | "--model" => o.model = v()?.into(),
            "-p" | "--prompt" => o.prompt = v()?.into(),
            "-f" | "--file" => {
                let path = v()?;
                o.prompt = std::fs::read_to_string(path).map_err(|e| format!("read prompt file {path}: {e}"))?;
            }
            "-n" => { o.n = parse_usize(a, v()?)?; o.n_set = true; }
            "-c" | "--ctx" => o.ctx = parse_context(a, v()?)?,
            "--raw" => o.raw = true,
            "--kv8" => o.kv8 = true,
            "--kv-disk" => o.kv_disk = true,
            "--allow-mismatch" => o.allow_mismatch = true,
            "--effort" => o.effort = parse_effort(a, v()?)?,
            "--system" => o.system = v()?.into(),
            "--temp" => o.temp = parse_temp(a, v()?)?,
            "--top-p" => o.top_p = parse_top_p(a, v()?)?,
            "--top-k" => o.top_k = parse_usize(a, v()?)?,
            "--seed" => o.seed = Some(parse_u64(a, v()?)?),
            "--greedy" => o.temp = 0.0,
            "--hide-thinking" => o.hide_thinking = true,
            "--loopcd" => o.loopcd_omega = parse_loopcd_omega(a, v()?)?,
            "--ref" => o.reference = v()?.into(),
            "-q" => o.quiet = true,
            "--stop" => o.stop = parse_usize(a, v()?)?,
            "-h" | "--help" => return Ok(("help".into(), o)),
            _ => return Err(format!("unknown option {a}\n\n{USAGE}")),
        }
    }
    if o.kv_disk && o.kv8 {
        return Err("--kv-disk cannot be combined with --kv8".into());
    }
    if o.loopcd_omega > 0.0 && o.kv_disk {
        return Err("--loopcd cannot be combined with --kv-disk".into());
    }
    Ok((cmd, o))
}

fn parse() -> (String, Opts) {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, mut o) = parse_cli_args(&raw).unwrap_or_else(|e| cli_error(e));
    if cmd != "help" {
        if o.kv8 { engine::KV8.store(true, std::sync::atomic::Ordering::Relaxed); }
        if o.kv_disk { engine::KV_DISK.store(true, std::sync::atomic::Ordering::Relaxed); }
        o.seed = Some(resolve_seed(o.seed));
    }
    (cmd, o)
}

/// reasoning effort as rendered by the GGUF chat template
fn effort_str(e: &str) -> String {
    let v = effort_value(e).expect("validated --effort");
    if v == 0.0 { "0".into() } else { format!("{v}") }
}

fn first_turn(o: &Opts, user: &str) -> String {
    let mut s = String::new();
    if !o.system.is_empty() { s += &format!("<|message_system|><|content_text|>{}<|end_message|>", o.system); }
    s += &format!("<|message_system|><|content_text|>Thinking effort level: {}<|end_message|>", effort_str(&o.effort));
    s += &format!("<|message_user|><|content_text|>{user}<|end_message|>");
    s
}

fn softcap(x: f32, c: f32) -> f32 { if c > 0.0 { c * (x / c).tanh() } else { x } }

fn token_rate(tokens: usize, elapsed: std::time::Duration) -> f64 {
    if elapsed.is_zero() { 0.0 } else { tokens as f64 / elapsed.as_secs_f64() }
}

fn run(o: &Opts) -> Result<(), String> {
    let eng = Engine::load(&o.model, o.ctx);
    eng.set_loopcd_omega(o.loopcd_omega);
    let tk = Tokenizer::load(&eng.g);
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let toks = tk.encode(&text, true);
    check_prompt_fits_ctx(toks.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
    let mut sess = Session::new(&eng, &tk);
    if std::env::var("INK_KV_PREALLOC").is_ok() { sess.run().reserve(o.ctx, false); }     // dev: no cache growth (to compare against)
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, eng.softcap, resolve_seed(o.seed));
    sess.run().prepare(toks.len(), toks.len());       // working memory for this prompt, mapped before the prefill starts
    let first = sess.prefill(&toks)?;
    let mut out = std::io::stdout();
    let mut r = Renderer::new(!o.hide_thinking, o.raw);
    let generated = sess.generate(first, o.n, &mut sampler, |t| { if !o.quiet { r.push(&tk, t, &mut out); } true });
    r.end(&mut out);
    generated?;
    let s = sess.stats;
    eprintln!("\n[inky] prompt {} tok @ {:.1} tok/s | generated {} tok @ {:.2} tok/s{} | {} at {} tokens",
              s.prompt_tokens, s.prompt_tokens as f64 / s.prompt_time.as_secs_f64(), s.gen_tokens, token_rate(s.gen_tokens, s.gen_time),
              if sampler.greedy() { " (greedy)" } else { "" }, kv_note(sess.run(), sess.n_past), sess.n_past);
    Ok(())
}

fn chat(o: &Opts) -> Result<(), String> {
    let eng = Engine::load(&o.model, o.ctx);
    eng.set_loopcd_omega(o.loopcd_omega);
    let tk = Tokenizer::load(&eng.g);
    let mut sess = Session::new(&eng, &tk);
    let mut sampler = Sampler::new(o.temp, o.top_k, o.top_p, eng.softcap, resolve_seed(o.seed));
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    eprintln!("[inky] chat ready (effort {}, {}). empty line or Ctrl-D to quit, /reset to clear the context.",
              o.effort, if sampler.greedy() { "greedy".to_string() } else { format!("temp {}", o.temp) });
    let mut has_turn = false;
    loop {
        print!("\n\x1b[1m> \x1b[0m"); out.flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 { break; }
        let line = line.trim_end_matches(['\n', '\r']).to_string();
        if line.trim().is_empty() { break; }
        if line.trim() == "/reset" { sess = Session::with_lane(&eng, &tk, sess.into_lane()); has_turn = false; eprintln!("[context cleared]"); continue; }
        let text = if has_turn { format!("<|message_user|><|content_text|>{line}<|end_message|>") } else { first_turn(o, &line) };
        let before = sess.stats;
        let toks = tk.encode(&text, true);
        if sess.ctx_left() < toks.len() + 64 { eprintln!("[context full: /reset]"); continue; }
        match sess.run_chat_turn(&toks, o.n, !o.hide_thinking, o.raw, &mut sampler, &mut out) {
            Ok(_) => {
                has_turn = true;
            }
            Err(err) => {
                eprintln!("\n[error: {err}]");
                continue;
            }
        }
        let s = sess.stats;
        let (pt, gt) = (s.prompt_tokens - before.prompt_tokens, s.gen_tokens - before.gen_tokens);
        eprintln!("\n\x1b[2m[{pt} prompt tok @ {:.0} tok/s · {gt} tok @ {:.1} tok/s · context {}/{} · {}]\x1b[0m",
                  pt as f64 / (s.prompt_time - before.prompt_time).as_secs_f64(), token_rate(gt, s.gen_time - before.gen_time),
                  sess.n_past, eng.max_ctx, kv_note(sess.run(), sess.n_past));
    }
    Ok(())
}

fn bench(o: &Opts) -> Result<(), String> {
    let eng = Engine::load(&o.model, o.ctx);
    eng.set_loopcd_omega(o.loopcd_omega);
    if let Ok(v) = std::env::var("INK_SKIP") { eng.skip.set(v.parse().unwrap()); }
    let tk = Tokenizer::load(&eng.g);
    let n = if o.n_set { o.n } else { 128 };
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let toks = tk.encode(&text, true);
    check_prompt_fits_ctx(toks.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
    let mut sampler = Sampler::new(0.0, 1, 1.0, eng.softcap, 0);
    // warm-up pass (pipelines, page mappings)
    let warm = {
        let mut s = Session::new(&eng, &tk);
        let f = s.prefill(&toks[..toks.len().min(64)])?;
        if n > 0 { s.generate(f, 8, &mut sampler, |_| true)?; }
        s.into_lane()
    };
    let mut sess = Session::with_lane(&eng, &tk, warm);     // the warm-up's buffers, already mapped
    sess.run().prepare(toks.len(), toks.len());       // as `run` does: working memory allocated and mapped before the clock starts
    let first = sess.prefill(&toks)?;
    sess.generate(first, n, &mut sampler, |_| true)?;
    let s = sess.stats;
    println!("prompt {:5} tok: {:8.1} ms  {:7.1} tok/s | decode {:4} tok: {:8.1} ms  {:6.2} tok/s",
             s.prompt_tokens, s.prompt_time.as_secs_f64() * 1e3, s.prompt_tokens as f64 / s.prompt_time.as_secs_f64(),
             s.gen_tokens, s.gen_time.as_secs_f64() * 1e3, token_rate(s.gen_tokens, s.gen_time));
    println!("memory besides the weights: {} at {} tokens | prefill scratch {:.0} MiB while prefilling, {:.1} MiB while generating",
             kv_note(sess.run(), sess.n_past), sess.n_past, s.peak_scratch as f64 / 1048576.0, sess.run().mem().1 as f64 / 1048576.0);
    if s.gen_tokens > 0 && sess.run().tier_on() {
        let (rb, rt) = sess.run().tier_io();
        println!("K/V file: {:.0} MiB read per token, {:.1} GiB/s, {:.1} ms of each token's {:.1} ms", rb as f64 / 1048576.0 / s.gen_tokens as f64,
                 rb as f64 / 1073741824.0 / rt.as_secs_f64(), rt.as_secs_f64() * 1e3 / s.gen_tokens as f64, s.gen_time.as_secs_f64() * 1e3 / s.gen_tokens as f64);
    }
    Ok(())
}

/// the K/V cache as it is held right now (with --kv-disk: the part in memory and the part that is only in the file)
fn kv_note(eng: engine::Run, n: usize) -> String {
    let mib = |b: usize| b as f64 / 1048576.0;
    if eng.tier_on() { format!("K/V cache {:.0} MiB in memory + {:.0} MiB on disk (all in memory: {:.0} MiB)", mib(eng.mem().0), mib(eng.kv_disk_bytes(n)), mib(eng.kv_exact_bytes(n))) }
    else { format!("K/V cache {:.0} MiB", mib(eng.mem().0)) }
}

fn reference_tokens(text: &str, context: usize) -> Result<Vec<u32>, String> {
    let mut toks = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == context {
            return Err(format!("reference has more than {context} tokens"));
        }
        toks.push(line.trim().parse().map_err(|_| format!("invalid token ID at line {}", i + 1))?);
    }
    if toks.is_empty() {
        return Err("reference has no tokens".into());
    }
    Ok(toks)
}

fn reference_logits(raw: &[u8], tokens: usize, vocab: usize) -> Result<Vec<f32>, String> {
    let row_bytes = vocab * size_of::<f32>();
    if raw.is_empty() || row_bytes == 0 || raw.len() % row_bytes != 0 || raw.len() / row_bytes > tokens {
        return Err(format!("expected 1 to {tokens} complete reference logit rows of width {vocab}, got {} bytes", raw.len()));
    }
    // Trimmed dumps cover the last positions of the token list.
    let first_pos = tokens - raw.len() / row_bytes;
    raw.chunks_exact(size_of::<f32>()).enumerate().map(|(i, bytes)| {
        let logit = f32::from_le_bytes(bytes.try_into().unwrap());
        if logit.is_finite() {
            Ok(logit)
        } else {
            Err(format!("non-finite reference logit at position {}, token {}", first_pos + i / vocab, i % vocab))
        }
    }).collect()
}

fn compare_logits(lg: &[f32], r: &[f32]) -> Option<(usize, usize, f32, f32)> {
    if lg.iter().chain(r).any(|x| !x.is_finite()) {
        return None;
    }
    let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    let maxdiff = lg.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    // KL(ref || ours)
    let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() };
    let (l1, l2) = (lse(lg), lse(r));
    let kl: f32 = r.iter().zip(lg).map(|(b, a)| { let pb = (b - l2).exp(); pb * ((b - l2) - (a - l1)) }).sum();
    if maxdiff.is_finite() && kl.is_finite() {
        Some((am(lg), am(r), maxdiff, kl))
    } else {
        None
    }
}

fn check_agreement_gate(agree: usize, compared: usize, allow_mismatch: bool) -> Result<(), String> {
    if compared == 0 {
        return Err("no reference positions were compared".into());
    }
    if agree < compared && !allow_mismatch {
        return Err(format!(
            "argmax mismatch: {agree}/{compared} positions agreed with reference (pass --allow-mismatch for exploratory runs)"
        ));
    }
    Ok(())
}

fn check(o: &Opts) -> Result<(), String> {
    if o.reference.is_empty() {
        return Err("expected --ref PREFIX".into());
    }
    let token_path = format!("{}.tokens", o.reference);
    let token_text = std::fs::read_to_string(&token_path).map_err(|err| format!("{token_path}: {err}"))?;
    let toks = reference_tokens(&token_text, o.ctx).map_err(|err| format!("{token_path}: {err}"))?;
    // INK_PF=N: prefill the first N tokens as one batch, then decode the rest one by one.
    let pf: usize = match std::env::var("INK_PF") {
        Ok(v) => v.parse().map_err(|_| format!("invalid INK_PF value {v:?}"))?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(err) => return Err(format!("INK_PF: {err}")),
    };
    if pf > toks.len() {
        return Err(format!("INK_PF={pf} exceeds the {} reference tokens", toks.len()));
    }
    let logit_path = format!("{}.logits", o.reference);
    let raw = std::fs::read(&logit_path).map_err(|err| format!("{logit_path}: {err}"))?;
    let eng = Engine::load(&o.model, o.ctx);
    let nv = eng.n_vocab;
    let reflog = reference_logits(&raw, toks.len(), nv).map_err(|err| format!("{logit_path}: {err}"))?;
    drop(raw);
    let first_ref = toks.len() - reflog.len() / nv;
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.try_set_tokens(0, &toks).map_err(|err| format!("{token_path}: {err}"))?;
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    // INK_DEC=1: decode positions through decode steps (encode_decode) instead of the single-stream decode path
    let dec_steps = std::env::var("INK_DEC").is_ok();
    if dec_steps { eng.ensure_dbatch(1); }
    let sabotage = std::env::var("INK_CHECK_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut sabotaged = false;
    let mut agree = 0;
    let mut compared = 0;
    let mut worst = 0f32;
    let kv_disk = engine::KV_DISK.load(std::sync::atomic::Ordering::Relaxed);
    let mut bits_hash = 0xcbf29ce484222325u64;       // over the raw bits of every logit of every checked step
    for pos in 0..toks.len() {
        if pf > 0 && pos + 1 < pf { continue; }
        if pf > 0 && pos + 1 == pf {
            let mut p0 = 0;
            while p0 < pf {
                let t = (pf - p0).min(eng.chunk);
                let cbp = eng.gpu.cmd();
                eng.encode_prefill(&cbp, p0, t, p0 + t == pf, false);
                cbp.commit();
                metal::wait_completed(&cbp, "prefill", p0)?;
                p0 += t;
            }
        } else if kv_disk {
            eng.gen_reserve(pos);
            eng.step(pos, true, false).wait(pos)?;
        } else {
            let cb = eng.gpu.cmd();
            if dec_steps {
                eng.encode_decode(&cb, &[engine::Span { lane: eng.lane, p0: pos, t: 1 }], true, false);
            } else {
                eng.encode_step(&cb, pos, true, false);
            }
            cb.commit();
            metal::wait_completed(&cb, "decode", pos)?;
        }
        // dev: INK_KVQ=kbits,vbits rounds the cache as it is written, to measure what a coarser cache would cost
        if let Ok(v) = std::env::var("INK_KVQ") {
            let (kb, vb) = v.split_once(',').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap();
            if pf > 0 && pos + 1 == pf { eng.kv_quant_sim(0, pf, kb, vb); } else { eng.kv_quant_sim(pos, pos + 1, kb, vb); }
        }
        let raw = if dec_steps && !(pf > 0 && pos + 1 == pf) { eng.span_logits(0) } else { eng.logits() };
        if let Some(token) = raw.iter().position(|x| !x.is_finite()) {
            return Err(format!("non-finite engine logit at position {pos}, token {token}"));
        }
        for &x in &*raw { bits_hash = (bits_hash ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); }
        if pos < first_ref { continue; }
        let mut lg: Vec<f32> = raw.iter().map(|&x| softcap(x, eng.effective_sampler_softcap(0))).collect();
        if sabotage && !sabotaged {
            for x in &mut lg { *x = -*x; }
            println!("INK_CHECK_SABOTAGE: negated engine logits at position {pos}");
            sabotaged = true;
        }
        let r = &reflog[(pos - first_ref) * nv..(pos - first_ref + 1) * nv];
        let (a1, a2, maxdiff, kl) = compare_logits(&lg, r)
            .ok_or_else(|| format!("non-finite comparison at position {pos}"))?;
        compared += 1;
        if a1 == a2 { agree += 1; }
        worst = worst.max(kl);
        println!("pos {pos:3} tok {:6} argmax ours {a1:6} ref {a2:6} {} maxdiff {maxdiff:.4} KL {kl:.6}", toks[pos], if a1 == a2 { "OK " } else { "XX " });
    }
    println!("argmax agreement {agree}/{compared} worst KL {worst:.9}");
    eprintln!("logit bits hash {bits_hash:016x}");
    if eng.tier_on() {
        let (n, (rb, rt)) = (toks.len(), eng.tier_io());
        eprintln!("disk tier: {:.1} MiB in memory, {:.1} MiB on disk only (all in memory: {:.1} MiB); read {:.0} MiB in {:.0} ms",
                  eng.mem().0 as f64 / 1048576.0, eng.kv_disk_bytes(n) as f64 / 1048576.0, eng.kv_exact_bytes(n) as f64 / 1048576.0, rb as f64 / 1048576.0, rt.as_secs_f64() * 1e3);
    }
    if sabotage {
        if !sabotaged || (agree == compared && worst <= 0.01) {
            eprintln!("INK_CHECK_SABOTAGE: failed to induce disagreement");
            std::process::exit(2);
        }
        return Err(format!("CHECK AGREEMENT FAILED ({agree}/{compared}, worst KL {worst:.9})"));
    }
    check_agreement_gate(agree, compared, o.allow_mismatch)
}

#[cfg(test)]
mod reference_tests {
    use super::*;

    #[test]
    fn check_agreement_gate_fails_on_argmax_miss_unless_explicitly_overridden() {
        assert_eq!(check_agreement_gate(58, 58, false), Ok(()));
        assert!(check_agreement_gate(57, 58, false).unwrap_err().contains("57/58"));
        assert_eq!(check_agreement_gate(57, 58, true), Ok(()));
        assert!(check_agreement_gate(0, 0, true).is_err());
        let (_, opts_default) = parse_cli_args(&["check".into(), "--ref".into(), "p".into()]).unwrap();
        assert!(!opts_default.allow_mismatch);
        let (_, opts_override) = parse_cli_args(&["check".into(), "--ref".into(), "p".into(), "--allow-mismatch".into()]).unwrap();
        assert!(opts_override.allow_mismatch);
    }

    #[test]
    fn reference_tokens_require_a_nonempty_sequence_that_fits() {
        assert_eq!(reference_tokens("0\n1\n2\n3\n", 4).unwrap(), vec![0, 1, 2, 3]);
        assert!(reference_tokens("0\n1\n2\n3\n4\n", 4).is_err());
        assert!(reference_tokens("", 4).is_err());
        assert!(reference_tokens("0\n-1\n", 4).is_err());
    }

    #[test]
    fn reference_logits_require_whole_finite_rows_and_allow_a_trimmed_tail() {
        let bytes: Vec<u8> = [1f32, 2.0].into_iter().flat_map(f32::to_le_bytes).collect();
        assert_eq!(reference_logits(&bytes, 3, 2).unwrap(), vec![1.0, 2.0]);
        assert!(reference_logits(&[], 3, 2).is_err());
        assert!(reference_logits(&bytes[..bytes.len() - 1], 3, 2).is_err());
        assert!(reference_logits(&bytes, 1, 1).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(reference_logits(&bad.to_le_bytes(), 3, 1).is_err());
        }
    }

    #[test]
    fn comparisons_reject_non_finite_logits_and_metrics() {
        assert_eq!(compare_logits(&[1.0, 2.0], &[1.0, 2.0]), Some((1, 1, 0.0, 0.0)));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(compare_logits(&[1.0, bad], &[1.0, 2.0]).is_none());
            assert!(compare_logits(&[1.0, 2.0], &[1.0, bad]).is_none());
        }
        assert!(compare_logits(&[f32::MAX, f32::MIN], &[f32::MIN, f32::MAX]).is_none());
    }
}

/// dev, per-row logits gate. One lane, a forced token sequence, no drafter. Run `encode_spans` of t=1 through
/// position p, then a verify span that starts at the already-cached position p-1 with length 1+k. Each vocabulary
/// row must match the same positions run as `encode_spans` of t=1, one position at a time. `INK_K` (default 3) and
/// `INK_P` (default 5) choose the span. Exits 1 on a mismatch.
fn rows(o: &Opts) {
    use engine::Span;
    let k: usize = std::env::var("INK_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let p: usize = std::env::var("INK_P").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    assert!(p >= 1, "INK_P is the frontier after the cached position");
    let ntok = p + k;
    let eng = Engine::load(&o.model, o.ctx.max((ntok + 1).div_ceil(64) * 64));
    eng.ensure_batch(eng.chunk);
    // forced synthetic ids, inside the embedding table. Not a drafter.
    let toks: Vec<u32> = (0..ntok).map(|i| ((i * 17 + 3) % 1000) as u32).collect();
    let fresh = || {
        let lane = Lane::new(&eng);
        let run = eng.on(&lane);
        run.reserve(ntok + 1, false);
        run.set_tokens(0, &toks);
        lane
    };
    let step1 = |lane: &Lane, pos: usize, logits: bool| {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane, p0: pos, t: 1 }], logits, false);
        cb.commit();
        wait_gate(&cb, "rows step1", pos);
    };
    let lane_r = fresh();
    let mut expect: Vec<Vec<f32>> = Vec::with_capacity(1 + k);
    for pos in 0..ntok {
        step1(&lane_r, pos, true);
        if pos + 1 >= p { expect.push(eng.logits().to_vec()); }
    }
    let lane = fresh();
    let run = eng.on(&lane);
    for pos in 0..p - 1 { step1(&lane, pos, false); }
    let cs = run.get_cs();
    step1(&lane, p - 1, false);
    run.set_cs(&cs);
    let cb = eng.gpu.cmd();
    eng.encode_verify(&cb, &[Span { lane: &lane, p0: p - 1, t: 1 + k }], k);
    cb.commit();
    wait_gate(&cb, "rows verify", p - 1);
    let mut ok = true;
    for r in 0..1 + k {
        let got = eng.span_logits(r);
        let same = &*got == expect[r].as_slice();
        ok &= same;
        println!("row {r} pos {} {}", p - 1 + r, if same { "same" } else { "DIFFERENT" });
    }
    println!("{}", if ok { "per-row logits match encode_spans of t=1" } else { "PER-ROW LOGITS DIFFER" });
    if !ok { std::process::exit(1); }
}

/// dev, rollback gate. One lane, synthetic draft tokens, no drafter. Two forced rejects of a verify span:
/// one in the middle of the draft, one at the last draft position. After rollback, `lane_hash` (cs and both
/// K/V buffers) matches `encode_spans` of t=1 stopped at the accepted length, and again after the correction
/// token is encoded as its own step. A match of the logits at the reject position is printed and is not the gate.
/// `INK_K` (default 3) and `INK_P` (default 5) choose the span. Exits 1 unless both rejects match.
fn rollback(o: &Opts) {
    use engine::Span;
    let k: usize = std::env::var("INK_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let p: usize = std::env::var("INK_P").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    assert!(k >= 3 && p >= 1, "INK_K >= 3 so the middle draft and the last draft differ; INK_P >= 1");
    let p0 = p - 1;
    let t = 1 + k;
    let ntok = p0 + t;
    let one = engine::Engine::cs_image_bytes();
    println!("cs image {one} bytes ({:.2} MiB). The span holds 1+k of these, transient, not one per context token.", one as f64 / 1048576.0);
    println!("k=3: {} bytes ({:.2} MiB per lane). k=7: {} bytes ({:.2} MiB per lane).", one * 4, (one * 4) as f64 / 1048576.0, one * 8, (one * 8) as f64 / 1048576.0);
    let eng = Engine::load(&o.model, o.ctx.max((ntok + 1).div_ceil(64) * 64));
    eng.ensure_batch(eng.chunk);
    let true_tok = |i: usize| ((i * 17 + 3) % 1000) as u32;
    let step1 = |lane: &Lane, pos: usize, logits: bool| {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane, p0: pos, t: 1 }], logits, false);
        cb.commit();
        wait_gate(&cb, "rollback step1", pos);
    };
    let mut ok = true;
    for (name, n_keep) in [("middle", (k + 1) / 2), ("last", k)] {
        let reject_at = p0 + n_keep;
        assert!(n_keep >= 1 && n_keep < t && reject_at < ntok);
        let draft_tok = |i: usize| if i >= reject_at { ((i * 17 + 9) % 1000) as u32 } else { true_tok(i) };
        assert_ne!(draft_tok(reject_at), true_tok(reject_at), "the rejected token is not the correction");
        let fresh = || {
            let lane = Lane::new(&eng);
            let run = eng.on(&lane);
            run.reserve(ntok + 1, false);
            lane
        };
        let lane_r = fresh();
        eng.on(&lane_r).set_tokens(0, &(0..=reject_at).map(true_tok).collect::<Vec<_>>());
        let mut ref_logits = Vec::new();
        let mut hash_pref = 0u64;
        for pos in 0..=reject_at {
            step1(&lane_r, pos, pos + 1 == reject_at);
            if pos + 1 == reject_at {
                ref_logits = eng.logits().to_vec();
                hash_pref = eng.lane_hash(&lane_r);
            }
        }
        let hash_ref = eng.lane_hash(&lane_r);
        let lane = fresh();
        let run = eng.on(&lane);
        run.set_tokens(0, &(0..ntok).map(draft_tok).collect::<Vec<_>>());
        for pos in 0..p0 { step1(&lane, pos, false); }
        let cs = run.get_cs();
        step1(&lane, p0, false);
        run.set_cs(&cs);
        let cb = eng.gpu.cmd();
        eng.encode_verify(&cb, &[Span { lane: &lane, p0, t }], k);
        cb.commit();
        wait_gate(&cb, "rollback verify", p0);
        let logits_same = &*eng.span_logits(n_keep - 1) == ref_logits.as_slice();
        eng.rollback_verify(&lane, 0, n_keep);
        let pref_ok = eng.lane_hash(&lane) == hash_pref;
        run.set_tokens(reject_at, &[true_tok(reject_at)]);
        step1(&lane, reject_at, false);
        let corr_ok = eng.lane_hash(&lane) == hash_ref;
        ok &= pref_ok && corr_ok;
        println!("case {name} n_keep {n_keep} reject-row logits {} prefix lane_hash {} correction lane_hash {}",
            if logits_same { "same" } else { "DIFFERENT" },
            if pref_ok { "same" } else { "DIFFERENT" },
            if corr_ok { "same" } else { "DIFFERENT" });
        if logits_same && !(pref_ok && corr_ok) {
            println!("case {name}: reject-row logits match and lane_hash does not; logits alone do not pass");
        }
    }
    println!("{}", if ok { "rollback lane_hash matches encode_spans of t=1" } else { "ROLLBACK LANE_HASH DIFFERS" });
    if !ok { std::process::exit(1); }
}

/// Speculative scheduler verification & rollback gate (`PR #42` / `PR #80` / `t-audit-pr42-port`):
/// - Layer 2 single-lane & batched two-lane `verify_step` vs 1-token `encode_spans(t=1)` (tokens + logit hashes).
/// - `n_keep = 0` full span abort restoring cache, convolution state, and saved tokens.
/// - Forced middle and last draft rejects restoring `lane_hash` before and after correction token.
/// - Stop token in the middle of accepted drafts emitting only the prefix (`bonus = None`).
/// - Context-boundary `k` clamping (`p0 = 60, cap = 64, k = 7 -> 3`) and 2-token `reply_budget` emission.
/// - Layer 3 paged (`Lane::new_blocked` + `BlockPools` + `BlockTable`) verification and rollback across
///   both a 64-token block boundary (`62..66`) and the 1024-token sliding-window boundary (`1022..1026`),
///   proving `Window`, `Full`, and recurrent `cs` state match 1-token paged execution bit-for-bit.
/// - Multi-step speculative generation driven by `NgramDraftProducer`, plus stochastic `verify_step_sampled`.
fn verify(o: &Opts) {
    use engine::Span;
    use sampler::checked_argmax;
    use scheduler::logit_hash;
    use verify::{
        run_speculative_loop, verify_step, verify_step_sampled, NgramDraftProducer,
        PagedVerifyLane, VerifyIn,
    };

    let sabotage = std::env::var("INK_VERIFY_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let k = 3usize;
    let ctx = 1152usize;
    println!("verify gate: Layer 2 + Layer 3 paged + drafter, ctx {ctx}");
    let eng = Engine::load(&o.model, ctx);
    eng.ensure_batch(eng.chunk);
    let prompt: Vec<u32> = (0..8).map(|i| ((i * 17 + 3) % 1000) as u32).collect();
    let p0 = prompt.len() - 1;
    let step1 = |lane: &Lane, pos: usize, logits: bool| {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane, p0: pos, t: 1 }], logits, false);
        cb.commit();
        wait_gate(&cb, "verify reference", pos);
    };

    // Baseline: `encode_spans` of `t = 1`, one position at a time.
    let base = Lane::new(&eng);
    eng.on(&base).set_tokens(0, &prompt);
    for pos in 0..p0 {
        step1(&base, pos, false);
    }
    let mut gen_toks = Vec::new();
    let mut gen_hash = Vec::new();
    let mut hash_after = Vec::new();
    for i in 0..=k {
        let pos = p0 + i;
        if i > 0 {
            eng.on(&base).set_tokens(pos, &[gen_toks[i - 1]]);
        }
        step1(&base, pos, true);
        let lg = eng.logits().to_vec();
        gen_hash.push(logit_hash(&lg));
        gen_toks.push(checked_argmax(&lg).expect("reference logits"));
        hash_after.push(eng.lane_hash(&base));
    }

    let prep = |toks: &[u32]| {
        let lane = Lane::new(&eng);
        let run = eng.on(&lane);
        run.set_tokens(0, toks);
        for pos in 0..toks.len() - 1 {
            step1(&lane, pos, false);
        }
        let cs = run.get_cs();
        step1(&lane, toks.len() - 1, false);
        (lane, cs)
    };
    let mut ok = true;

    // 1. Full accept on Layer 2.
    {
        let (lane, cs) = prep(&prompt);
        let drafts = gen_toks[..k].to_vec();
        let kept = verify_step(
            &eng,
            &[VerifyIn { lane: &lane, p0, cs_before: &cs, drafts: &drafts }],
            &[],
        )
        .pop()
        .unwrap()
        .transpose()
        .expect("verify step");
        if let Some(mut kept) = kept {
            if sabotage {
                println!("INK_VERIFY_SABOTAGE: perturbing verified logit hash");
                kept.hashes[0] ^= 1;
            }
            let same_tok = kept.emitted == gen_toks[..=k];
            let same_hash = kept.hashes == gen_hash[..=k];
            ok &= same_tok && same_hash;
            println!(
                "match kept {} logit hashes {}",
                if same_tok { "same" } else { "DIFFERENT" },
                if same_hash { "same" } else { "DIFFERENT" }
            );
        } else {
            ok = false;
        }
    }

    // 2. Two-lane batched verify on Layer 2.
    if eng.hgemm && !sabotage {
        let (left, left_cs) = prep(&prompt);
        let mut other = prompt.clone();
        other[p0] += 1;
        let (right, right_cs) = prep(&other);
        let reference = Lane::new(&eng);
        eng.on(&reference).set_tokens(0, &other);
        for pos in 0..p0 {
            step1(&reference, pos, false);
        }
        let mut right_toks = Vec::new();
        let mut right_hashes = Vec::new();
        for i in 0..=k {
            let pos = p0 + i;
            if i > 0 {
                eng.on(&reference).set_tokens(pos, &[right_toks[i - 1]]);
            }
            step1(&reference, pos, true);
            let row = eng.logits();
            right_toks.push(checked_argmax(&row).expect("reference logits"));
            right_hashes.push(logit_hash(&row));
        }
        let input = [
            VerifyIn { lane: &left, p0, cs_before: &left_cs, drafts: &gen_toks[..k] },
            VerifyIn { lane: &right, p0, cs_before: &right_cs, drafts: &right_toks[..k] },
        ];
        let kept = verify_step(&eng, &input, &[]);
        let same = kept
            .into_iter()
            .zip([(&gen_toks, &gen_hash), (&right_toks, &right_hashes)])
            .all(|(kept, (toks, hashes))| {
                kept.transpose()
                    .expect("batched verify step")
                    .is_some_and(|kept| &kept.emitted == toks && &kept.hashes == hashes)
            });
        ok &= same;
        println!("two-lane verify tokens and logit hashes {}", if same { "same" } else { "DIFFERENT" });
    }

    // 3. Full `n_keep = 0` abort rollback + middle/last rejects + stop-in-middle + boundary clamp.
    if !sabotage {
        {
            let (lane, cs_before_anchor) = prep(&prompt);
            let run = eng.on(&lane);
            let saved_tokens = [41u32, 42, 43];
            run.set_tokens(p0 + 1, &saved_tokens);
            let before = eng.lane_hash(&lane);
            let cb = eng.gpu.cmd();
            eng.encode_verify(&cb, &[Span { lane: &lane, p0, t: 1 + k }], k);
            run.set_cs(&cs_before_anchor);
            run.set_tokens(p0 + 1, &gen_toks[..k]);
            cb.commit();
            wait_gate(&cb, "verify abort probe", p0);
            eng.rollback_verify(&lane, 0, 0);
            let same = eng.lane_hash(&lane) == before
                && saved_tokens
                    .iter()
                    .enumerate()
                    .all(|(i, &tok)| run.token_at(p0 + 1 + i) == tok);
            ok &= same;
            println!(
                "full verify rollback {}",
                if same { "restored cache, convolution state and tokens" } else { "DIFFERENT" }
            );
        }

        for (name, at) in [("middle", 1usize), ("last", k - 1)] {
            let (lane, cs) = prep(&prompt);
            let mut drafts = gen_toks[..k].to_vec();
            drafts[at] = (drafts[at] + 1) % eng.n_vocab as u32;
            let kept = verify_step(
                &eng,
                &[VerifyIn { lane: &lane, p0, cs_before: &cs, drafts: &drafts }],
                &[],
            )
            .pop()
            .unwrap()
            .transpose()
            .expect("verify reject")
            .expect("verify ran");
            let next_ok = kept.bonus == Some(gen_toks[at]);
            let pos = kept.p0 + kept.n_keep;
            eng.on(&lane).set_tokens(pos, &[gen_toks[at]]);
            step1(&lane, pos, false);
            let hash_ok = eng.lane_hash(&lane) == hash_after[at + 1];
            ok &= next_ok && hash_ok;
            println!(
                "reject {name} next token {} lane_hash {}",
                if next_ok { "same" } else { "DIFFERENT" },
                if hash_ok { "same" } else { "DIFFERENT" }
            );
        }

        {
            let stop_tok = gen_toks[1];
            let (lane, cs) = prep(&prompt);
            let drafts = gen_toks[..k].to_vec();
            let kept = verify_step(
                &eng,
                &[VerifyIn { lane: &lane, p0, cs_before: &cs, drafts: &drafts }],
                &[stop_tok],
            )
            .pop()
            .unwrap()
            .transpose()
            .expect("verify stop")
            .expect("verify ran");
            let expect: Vec<u32> = drafts
                .iter()
                .copied()
                .scan(false, |done, tok| {
                    if *done {
                        None
                    } else {
                        if tok == stop_tok {
                            *done = true;
                        }
                        Some(tok)
                    }
                })
                .collect();
            let prefix_ok = kept.emitted == expect && kept.finished && kept.bonus.is_none();
            ok &= prefix_ok;
            println!(
                "stop in the middle emitted {:?} {}",
                kept.emitted,
                if prefix_ok { "prefix only" } else { "DIFFERENT" }
            );
        }

        // Boundary clamp on a 64-cap lane (`p0 = 60, cap = 64, want = 7 -> k = 3`).
        {
            let n = 61usize;
            let toks: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % 1000) as u32).collect();
            let lane = Lane::new(&eng);
            let run = eng.on(&lane);
            let cap = run.kv_capacity();
            assert_eq!(cap, 64);
            run.set_tokens(0, &toks);
            let cb = eng.gpu.cmd();
            eng.encode_spans(&cb, &[Span { lane: &lane, p0: 0, t: n - 1 }], false, false);
            cb.commit();
            wait_gate(&cb, "verify boundary pf", 0);
            let cs = run.get_cs();
            step1(&lane, n - 1, false);
            let p0b = n - 1;
            let want = 7usize;
            let drafts = vec![1u32; want];
            let kept = verify_step(
                &eng,
                &[VerifyIn { lane: &lane, p0: p0b, cs_before: &cs, drafts: &drafts }],
                &[],
            )
            .pop()
            .unwrap()
            .transpose()
            .expect("boundary verify")
            .expect("boundary ran");
            let clamped = kept.k < want && p0b + 1 + kept.k <= cap;
            ok &= clamped;
            println!(
                "boundary k clamped {want} -> {}, p0 {p0b} cap {cap}, {}",
                kept.k,
                if clamped { "no capacity assert" } else { "NOT CLAMPED" }
            );
        }

        // 4. Layer 3 paged verification & rollback across BOTH 64-token block boundary (`p0=62, k=3`)
        //    and 1024-token sliding-window boundary (`p0=1022, k=3`, proving rejected drafts past 1024
        //    keep `Window` block 0 intact and match 1-token paged `lane_hash` across Full + Window + CS!).
        for (bname, anchor_p0) in [("block-boundary-62", 62usize), ("window-boundary-1022", 1022usize)] {
            let stream: Vec<u32> = (0..anchor_p0 + k + 4).map(|i| ((i * 23 + 7) % 1000) as u32).collect();
            // Paged 1-token reference stopping after rejecting middle draft (`n_keep = 2`, then correction token).
            let mut ref_paged = PagedVerifyLane::new(&eng, anchor_p0 + k + 8);
            eng.on(&ref_paged.lane).set_tokens(0, &stream);
            if anchor_p0 > 1 {
                let _ = ref_paged.ensure_span_blocks(0, anchor_p0);
                let cb = eng.gpu.cmd();
                eng.encode_spans(&cb, &[Span { lane: &ref_paged.lane, p0: 0, t: anchor_p0 }], false, false);
                cb.commit();
                wait_gate(&cb, "paged ref prefill", 0);
                for b in 0..=(anchor_p0 - 1) / blocks::BLOCK {
                    let bi = blocks::BlockIndex::new(b as u32);
                    let blen = (anchor_p0 - b * blocks::BLOCK).min(blocks::BLOCK) as u32;
                    let tbl = ref_paged.lane.block_table().borrow();
                    let mut pools = eng.block_pool().borrow_mut();
                    for kind in [blocks::BlockKind::Full, blocks::BlockKind::Window] {
                        let e = tbl.get(kind, bi).unwrap().unwrap();
                        pools.of_mut(kind).record_mut(e.slot()).unwrap().set_length(blen).unwrap();
                    }
                }
                ref_paged.written = anchor_p0;
                ref_paged.prune_expired_window(anchor_p0);
            }
            let cs_anchor = eng.on(&ref_paged.lane).get_cs();
            ref_paged.step_one(anchor_p0, true).unwrap();
            let d0 = checked_argmax(&eng.logits()).unwrap();
            eng.on(&ref_paged.lane).set_tokens(anchor_p0 + 1, &[d0]);
            ref_paged.step_one(anchor_p0 + 1, true).unwrap();
            let corr = checked_argmax(&eng.logits()).unwrap();
            let want_pref_hash = eng.lane_hash(&ref_paged.lane);
            eng.on(&ref_paged.lane).set_tokens(anchor_p0 + 2, &[corr]);
            ref_paged.step_one(anchor_p0 + 2, false).unwrap();
            let want_corr_hash = eng.lane_hash(&ref_paged.lane);

            // Paged verify candidate: draft 0 matches (`d0`), draft 1 mismatches (`corr + 1`), draft 2 crosses boundary!
            let mut cand_paged = PagedVerifyLane::new(&eng, anchor_p0 + k + 8);
            eng.on(&cand_paged.lane).set_tokens(0, &stream);
            if anchor_p0 > 1 {
                let _ = cand_paged.ensure_span_blocks(0, anchor_p0);
                let cb = eng.gpu.cmd();
                eng.encode_spans(&cb, &[Span { lane: &cand_paged.lane, p0: 0, t: anchor_p0 }], false, false);
                cb.commit();
                wait_gate(&cb, "paged cand prefill", 0);
                for b in 0..=(anchor_p0 - 1) / blocks::BLOCK {
                    let bi = blocks::BlockIndex::new(b as u32);
                    let blen = (anchor_p0 - b * blocks::BLOCK).min(blocks::BLOCK) as u32;
                    let tbl = cand_paged.lane.block_table().borrow();
                    let mut pools = eng.block_pool().borrow_mut();
                    for kind in [blocks::BlockKind::Full, blocks::BlockKind::Window] {
                        let e = tbl.get(kind, bi).unwrap().unwrap();
                        pools.of_mut(kind).record_mut(e.slot()).unwrap().set_length(blen).unwrap();
                    }
                }
                cand_paged.written = anchor_p0;
                cand_paged.prune_expired_window(anchor_p0);
            }
            let cs_cand = eng.on(&cand_paged.lane).get_cs();
            assert_eq!(cs_cand, cs_anchor);
            cand_paged.step_one(anchor_p0, false).unwrap();
            let pre_written = cand_paged.written;
            let allocated = cand_paged.ensure_span_blocks(anchor_p0, 1 + k);
            let bad_drafts = [d0, (corr + 1) % eng.n_vocab as u32, (corr + 2) % eng.n_vocab as u32];
            let kept = verify_step(
                &eng,
                &[VerifyIn { lane: &cand_paged.lane, p0: anchor_p0, cs_before: &cs_cand, drafts: &bad_drafts }],
                &[],
            )
            .pop()
            .unwrap()
            .transpose()
            .expect("paged verify")
            .expect("paged verify ran");
            assert_eq!(kept.n_keep, 2);
            cand_paged.finish_verify_rollback(&allocated, pre_written, anchor_p0 + kept.n_keep);
            let pref_same = eng.lane_hash(&cand_paged.lane) == want_pref_hash && kept.bonus == Some(corr);
            eng.on(&cand_paged.lane).set_tokens(anchor_p0 + 2, &[corr]);
            cand_paged.step_one(anchor_p0 + 2, false).unwrap();
            let corr_same = eng.lane_hash(&cand_paged.lane) == want_corr_hash;
            ok &= pref_same && corr_same;
            println!(
                "paged {bname} (n_keep=2): prefix Full+Window+CS {} correction Full+Window+CS {}",
                if pref_same { "same" } else { "DIFFERENT" },
                if corr_same { "same" } else { "DIFFERENT" }
            );
            ref_paged.release();
            cand_paged.release();
        }

        // 5. Multi-step `NgramDraftProducer` loop + stochastic `verify_step_sampled` (`temp = 0.8`).
        {
            let mut drafter = NgramDraftProducer::new(3, gen_toks[0]);
            let (spec_out, reason) =
                run_speculative_loop(&eng, &prompt, k + 1, 128, 3, &[], &mut drafter)
                    .expect("ngram speculative loop");
            let loop_same = spec_out == gen_toks[..=k] && reason == "length";
            ok &= loop_same;
            println!(
                "ngram drafter multi-step loop ({} tokens, reason={reason}): {}",
                spec_out.len(),
                if loop_same { "matches t=1" } else { "DIFFERENT" }
            );

            let (slane, scs) = prep(&prompt);
            let mut draft_s = Sampler::draft(0.8, 64, 0.95, eng.softcap, 2026);
            let mut target_s = Sampler::new(0.8, 64, 0.95, eng.softcap, 2026);
            let dummy_q = vec![0.0f32; eng.n_vocab];
            let s_drafts = [draft_s.sample_next(&dummy_q).unwrap(), draft_s.sample_next(&dummy_q).unwrap()];
            let skpt = verify_step_sampled(
                &eng,
                &VerifyIn { lane: &slane, p0, cs_before: &scs, drafts: &s_drafts },
                &[&dummy_q, &dummy_q],
                &mut target_s,
                &[],
            )
            .expect("stochastic verify step")
            .expect("stochastic verify ran");
            let samp_ok = !skpt.emitted.is_empty()
                && skpt.emitted.iter().all(|&t| (t as usize) < eng.n_vocab)
                && skpt.n_keep == skpt.emitted.len();
            ok &= samp_ok;
            println!(
                "stochastic verify (temp=0.8, top_k=64): emitted {} valid tokens (n_keep={})",
                skpt.emitted.len(),
                skpt.n_keep
            );
        }
    }

    println!(
        "{}",
        if ok {
            "verify gate matches encode_spans of t=1"
        } else {
            "VERIFY GATE FAILED"
        }
    );
    if !ok {
        std::process::exit(1);
    }
}

/// dev: batch invariance. Prefill the same tokens split at different points and hash the final logits: the hashes must be
/// equal, since a split moves every later token to a different batch, tile slot and set of neighbours.
fn binv(o: &Opts) -> Result<(), String> {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    let tk = Tokenizer::load(&eng.g);
    let toks: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    // each entry is one way to split the prompt: boundaries joined by '+' ("37+190": three prefills); an entry may repeat,
    // which tests that the same split gives the same bits run after run. At least two schedules are required (refusing e.g. `INK_SPLITS=0` alone).
    let raw_splits = std::env::var("INK_SPLITS").unwrap_or_else(|_| "0,1,5,16,23,100,1001".into());
    let splits: Vec<Vec<usize>> = raw_splits
        .split(',')
        .map(|v| {
            if v.trim().is_empty() {
                eprintln!("[inky] INK_SPLITS contains an empty schedule in {raw_splits:?}");
                std::process::exit(2);
            }
            v.split('+')
                .map(|x| {
                    x.trim().parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("[inky] invalid split point {x:?} in INK_SPLITS={raw_splits:?}");
                        std::process::exit(2);
                    })
                })
                .filter(|&x| x > 0)
                .collect()
        })
        .collect();
    if splits.len() < 2 {
        eprintln!("[inky] INK_SPLITS must specify at least two split schedules to compare (got {raw_splits:?})");
        std::process::exit(2);
    }
    let sabotage = std::env::var("INK_BINV_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut sabotaged = false;
    let mut first = None;
    let mut base: Vec<f32> = Vec::new();
    let mut ok = true;
    for k in splits {
        let mut sess = Session::new(&eng, &tk);
        let mut run_toks = toks.clone();
        if sabotage && first.is_some() && !sabotaged {
            run_toks[0] = (run_toks[0] + 1) % (eng.n_vocab as u32);
            println!("INK_BINV_SABOTAGE: perturbed token 0 on second split");
            sabotaged = true;
        }
        let mut p = 0;
        for &b in k.iter().chain(std::iter::once(&run_toks.len())) { if b > p { sess.prefill(&run_toks[p..b])?; p = b; } }
        let k = k.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("+");
        let k = if k.is_empty() { "0".to_string() } else { k };
        let lg = eng.logits();
        let mut h = 0xcbf29ce484222325u64;
        for &x in &*lg { h = (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); }
        let same = *first.get_or_insert(h) == h;
        if base.is_empty() { base = lg.to_vec(); }
        let maxdiff = lg.iter().zip(&base).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        ok &= same;
        // KL(first || this) over the soft-capped logits, as sampling sees them
        let cap = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| softcap(x, eng.softcap)).collect() };
        let (pa, pb) = (cap(&base), cap(&lg));
        let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() };
        let (l1, l2) = (lse(&pa), lse(&pb));
        let kl: f32 = pa.iter().zip(&pb).map(|(x, y)| { let p = (x - l1).exp(); p * ((x - l1) - (y - l2)) }).sum();
        println!("split at {k:>9}: logit bits hash {h:016x} {} (max logit diff {maxdiff:.6}, KL {kl:.6}, argmax {})", if same { "same" } else { "DIFFERENT" }, am(&lg));
    }
    if sabotage && !sabotaged {
        eprintln!("INK_BINV_SABOTAGE: need at least two splits in INK_SPLITS");
        std::process::exit(2);
    }
    println!("{}", if ok { "batch invariant: every split gives identical logits" } else { "NOT batch invariant" });
    if !ok { std::process::exit(1); }
    Ok(())
}

/// Compare the same target row through two chunkings of the prefill arithmetic. Every checked end gets its own
/// span of CROSSPATH_SPAN rows ending at that end (fewer only when the end is shorter), started from one
/// base lane that never takes part in a comparison: one lane prefills the full multi-row span, while the
/// other prefills its first rows and last row separately. Every span has at least two rows, and one ending
/// just past a split boundary crosses it. Decode has a different arithmetic and is not expected to match.
const CROSSPATH_SPAN: usize = 17;
const CROSSPATH_ENDS: &str = "63,64,65,255,256,257,1023,1024,1025,2047,2048,2049";

fn crosspath_start(end: usize) -> usize { end.saturating_sub(CROSSPATH_SPAN) }

fn crosspath_completed(status: objc2_metal::MTLCommandBufferStatus) -> bool { status == objc2_metal::MTLCommandBufferStatus::Completed }

fn wait_crosspath(cb: &metal::CmdBuf) {
    cb.waitUntilCompleted();
    let status = cb.status();
    if !crosspath_completed(status) {
        let error = cb.error().map(|e| e.localizedDescription().to_string());
        panic!("crosspath Metal command failed (status {status:?}): {}", error.as_deref().unwrap_or("no driver error provided"));
    }
}

#[cfg(test)]
mod crosspath_tests {
    use super::*;

    #[test]
    fn only_successful_gpu_completion_is_accepted() {
        assert!(crosspath_completed(objc2_metal::MTLCommandBufferStatus::Completed));
        assert!(!crosspath_completed(objc2_metal::MTLCommandBufferStatus::Error));
        assert!(!crosspath_completed(objc2_metal::MTLCommandBufferStatus::Scheduled));
    }

    #[test]
    fn every_default_boundary_uses_an_independent_17_row_span() {
        let ends: Vec<usize> = CROSSPATH_ENDS.split(',').map(|s| s.parse().unwrap()).collect();
        assert_eq!(ends.len(), 12);
        let mut cursor = 0;
        for end in ends {
            let start = crosspath_start(end);
            assert!(start >= cursor);
            assert_eq!(end - start, 17);
            cursor = start;
        }
    }
}

fn crosspath(o: &Opts) {
    let toks: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens")
        .lines().map(|line| line.trim().parse().unwrap()).collect();
    let ends: Vec<usize> = std::env::var("INK_COMPARE_ENDS").unwrap_or_else(|_| {
        CROSSPATH_ENDS.into()
    }).split(',').map(|n| n.parse().expect("INK_COMPARE_ENDS lists token counts")).collect();
    let len = *ends.last().unwrap();
    assert!(ends[0] >= 2 && ends.windows(2).all(|p| p[0] < p[1]),
            "INK_COMPARE_ENDS must list increasing token counts of at least 2 (a span needs two rows)");
    assert!(len <= toks.len() && len <= o.ctx, "crosspath fixture or context is too short");

    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    assert!(CROSSPATH_SPAN <= eng.chunk, "a crosspath span must fit one prefill chunk");
    let base = Lane::new(&eng);
    let prefill = Lane::new(&eng);
    let split = Lane::new(&eng);
    for lane in [&base, &prefill, &split] {
        let run = eng.on(lane);
        run.reserve(len, false);
        run.set_tokens(0, &toks[..len]);
    }

    let run_prefill = |lane: &Lane, p0, t, logits| {
        let cb = eng.gpu.cmd();
        eng.on(lane).encode_prefill(&cb, p0, t, logits, false);
        cb.commit();
        wait_crosspath(&cb);
    };
    // `cursor` is how far the base lane has prefilled. Span starts increase with the ends, so it only advances.
    let mut cursor = 0;
    let mut ok = true;
    for end in ends {
        let start = crosspath_start(end);
        let target = end - 1;
        assert!(start >= cursor && target - start >= 1, "crosspath span [{start}, {end}) must start at or after the base lane ({cursor}) and hold two rows");
        while cursor < start {
            let count = (start - cursor).min(eng.chunk);
            run_prefill(&base, cursor, count, false);
            cursor += count;
        }
        eng.copy_lane(&base, &prefill);
        eng.copy_lane(&base, &split);
        eng.on(&split).set_tokens(0, &toks[..len]);
        run_prefill(&split, start, target - start, false);
        run_prefill(&prefill, start, end - start, true);
        let expected = eng.logits().to_vec();
        if std::env::var_os("INK_CROSSPATH_SABOTAGE").is_some() {
            eng.on(&split).set_tokens(target, &[(toks[target] + 1) % eng.n_vocab as u32]);
        }
        run_prefill(&split, target, 1, true);
        let actual = eng.logits();
        let first_bad = expected.iter().zip(actual.iter()).position(|(&a, &b)| {
            !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits()
        });
        let state_equal = eng.lane_hash(&prefill) == eng.lane_hash(&split);
        if let Some(i) = first_bad {
            println!("after {end} tokens: logit {i} differs (prefill {:08x}, split {:08x}), prefill span [{start}, {end})",
                     expected[i].to_bits(), actual[i].to_bits());
        }
        if !state_equal {
            println!("after {end} tokens: cache or convolution state differs, prefill span [{start}, {end})");
        }
        if first_bad.is_none() && state_equal {
            println!("after {end} tokens: exact, prefill span [{start}, {end}) of {} rows", end - start);
        }
        ok &= first_bad.is_none() && state_equal;
    }
    if !ok {
        println!("CROSSPATH MISMATCH");
        std::process::exit(1);
    }
    println!("crosspath exact: prefill chunkings agree");
}

/// dev, layer 1's gate: two lanes on one engine, interleaved, with the scratch arenas poisoned at every switch, must give
/// each lane exactly the logits it gives alone. Lane A is the --ref tokens, lane B the same tokens rotated by a third (a
/// different stream of the same length). Each lane prefills its first INK_PF tokens in chunks of INK_LANE_CHUNK, then
/// decodes the rest one step at a time.
fn lanes(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    let a: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let mut b = a.clone(); b.rotate_left(a.len() / 3);
    let pf: usize = std::env::var("INK_PF").ok().and_then(|v| v.parse().ok()).unwrap_or(1000).min(a.len() - 1);
    let ch: usize = std::env::var("INK_LANE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    // the work of one lane as a list of (p0, t): prefill chunks, then single decode steps
    let mut work: Vec<(usize, usize)> = Vec::new();
    let mut p = 0; while p < pf { let t = (pf - p).min(ch); work.push((p, t)); p += t; }
    for p in pf..a.len() { work.push((p, 1)); }
    let poison = std::env::var("INK_NO_POISON").is_err();
    let hash_step = |run: &engine::Run, (p0, t): (usize, usize), h: &mut u64| {
        if poison { run.poison_scratch(); }
        let cb = run.gpu.cmd();
        if t == 1 { run.encode_step(&cb, p0, true, false); } else { run.encode_prefill(&cb, p0, t, true, false); }
        cb.commit(); wait_gate(&cb, "lanes step", p0);
        for &x in &*run.logits() { *h = (*h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3); }
    };
    // per-step hashes, to name the first step where two runs part
    let trace = |run: &engine::Run, toks: &[u32], other: Option<&engine::Run>| -> Vec<u64> {
        let mut v = Vec::new();
        for &w in &work {
            if let Some(o) = other { let mut x = 0u64; hash_step(o, w, &mut x); }
            let mut h = 0xcbf29ce484222325u64; hash_step(run, w, &mut h); v.push(h);
        }
        let _ = toks; v
    };
    let alone = |toks: &[u32]| -> u64 {
        let lane = Lane::new(&eng); let run = eng.on(&lane);
        run.reserve(toks.len() + 1, false); run.set_tokens(0, toks);
        let mut h = 0xcbf29ce484222325u64;
        for &w in &work { hash_step(&run, w, &mut h); }
        h
    };
    if std::env::var("INK_ARENA_MAP").is_ok() { println!("{}", eng.arena_map()); }
    let (ha, hb) = (alone(&a), alone(&b));
    let (la, lb) = (Lane::new(&eng), Lane::new(&eng));
    let (ra, rb) = (eng.on(&la), eng.on(&lb));
    for (r, t) in [(&ra, &a), (&rb, &b)] { r.reserve(t.len() + 1, false); r.set_tokens(0, t); }
    if std::env::var("INK_LANES_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0") {
        let pos = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(0);
        let leaked = if a[pos] != b[pos] { b[pos] } else { (a[pos] + 1) % (eng.n_vocab as u32) };
        ra.set_tokens(pos, &[leaked]);
        println!("INK_LANES_SABOTAGE: bled lane B token into lane A at position {pos}");
    }
    let (mut ia, mut ib) = (0xcbf29ce484222325u64, 0xcbf29ce484222325u64);
    for &w in &work { hash_step(&ra, w, &mut ia); hash_step(&rb, w, &mut ib); }
    if std::env::var("INK_LANE_TRACE").is_ok() {
        let (l1, l2, l3) = (Lane::new(&eng), Lane::new(&eng), Lane::new(&eng));
        let (r1, r2, r3) = (eng.on(&l1), eng.on(&l2), eng.on(&l3));
        for (r, t) in [(&r1, &a), (&r2, &a), (&r3, &b)] { r.reserve(t.len() + 1, false); r.set_tokens(0, t); }
        let solo = trace(&r1, &a, None);
        let mixed = trace(&r2, &a, Some(&r3));
        match solo.iter().zip(&mixed).position(|(x, y)| x != y) {
            Some(i) => println!("first difference at step {i}: work {:?} ({} steps)", work[i], work.len()),
            None => println!("per-step traces agree"),
        }
    }
    println!("lane A alone {ha:016x} interleaved {ia:016x} {}", if ha == ia { "same" } else { "DIFFERENT" });
    println!("lane B alone {hb:016x} interleaved {ib:016x} {}", if hb == ib { "same" } else { "DIFFERENT" });
    let ok = ha == ia && hb == ib && ha != hb;
    println!("{}", if ok { "lanes independent: interleaved and poisoned, each lane gives exactly its own logits" } else { "LANES NOT INDEPENDENT" });
    if !ok { std::process::exit(1); }
}

/// dev, layer 2's gate: N lanes advanced together in steps of several spans must each give exactly what they give alone:
/// the same logits at every step, the same cache and convolution rows at the end. Lane i reads the --ref tokens rotated by
/// 17 i; it prefills in prime-sized chunks with one-token chunks among them, then decodes; lanes start staggered, so a step
/// mixes prefill spans of several sizes with decode spans. INK_LANES lists the lane counts to try.
/// INK_SPANS_SABOTAGE=1 is the negative control: one span is bound to the next lane's state, and the gate must then
/// fail; tests/ci.sh runs it and requires that failure, so the gate is known to catch a wrong-lane binding.
fn spans(o: &Opts) {
    use engine::Span;
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let chunks = [31usize, 1, 23, 7, 17, 1, 29];
    let ndec: usize = std::env::var("INK_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let plen: usize = chunks.iter().sum();
    let len = plen + ndec;
    let work: Vec<(usize, usize)> = { let mut w = Vec::new(); let mut p = 0; for &c in &chunks { w.push((p, c)); p += c; } for q in plen..len { w.push((q, 1)); } w };
    let stream = |i: usize| -> Vec<u32> { let mut v = base.clone(); v.rotate_left((17 * i) % base.len()); v.truncate(len + 1); v };
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let fresh = |i: usize| { let l = Lane::new(&eng); let r = eng.on(&l); r.reserve(len + 1, false); r.set_tokens(0, &stream(i)); l };
    let counts: Vec<usize> = std::env::var("INK_LANES").unwrap_or("1,2,3,17,64,65,128".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let sabotage = std::env::var("INK_SPANS_SABOTAGE").is_ok();
    let nmax = *counts.iter().max().unwrap();
    // each lane alone: one span per step
    let mut alone: Vec<(Vec<u64>, u64)> = Vec::new();
    for i in 0..nmax {
        let l = fresh(i);
        let mut hs = Vec::new();
        for &(p0, t) in &work {
            let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &[Span { lane: &l, p0, t }], true, false); cb.commit(); wait_gate(&cb, "spans alone", p0);
            hs.push(fnv(&eng.logits()));
        }
        alone.push((hs, eng.lane_hash(&l)));
    }
    let mut ok = true;
    for &n in &counts {
        let mut sabotaged = false;
        let lanes: Vec<Lane> = (0..n).map(fresh).collect();
        let delay = |i: usize| i % 3;                       // lane i starts i % 3 steps late
        let mut hs: Vec<Vec<u64>> = vec![Vec::new(); n];
        let steps = work.len() + 2;
        let (mut nsteps, mut mixed) = (0, 0);
        for s in 0..steps {
            let mut sp = Vec::new(); let mut who = Vec::new();
            for i in 0..n { if s >= delay(i) && s - delay(i) < work.len() { let (p0, t) = work[s - delay(i)]; sp.push(Span { lane: &lanes[i], p0, t }); who.push(i); } }
            if sp.is_empty() { continue; }
            if sabotage && !sabotaged && sp.len() >= 2 {
                // negative control: swap the first two spans' lanes so every span still names a distinct lane,
                // while binding both spans to the wrong lane's state; their logits and caches must then differ
                println!("INK_SPANS_SABOTAGE: spans of lanes {} and {} swapped", who[0], who[1]);
                let lane0 = sp[0].lane;
                sp[0].lane = sp[1].lane;
                sp[1].lane = lane0;
                sabotaged = true;
            }
            if sp.iter().any(|x| x.t > 1) && sp.iter().any(|x| x.t == 1) { mixed += 1; }
            nsteps += 1;
            let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &sp, true, false); cb.commit(); wait_gate(&cb, "spans batch", sp[0].p0);
            for (k, &i) in who.iter().enumerate() { hs[i].push(fnv(&*(if sp.len() == 1 { eng.logits() } else { eng.span_logits(k) }))); }
        }
        let mut bad = Vec::new();
        for i in 0..n {
            let step = hs[i].iter().zip(&alone[i].0).position(|(a, b)| a != b);
            let cache = eng.lane_hash(&lanes[i]) != alone[i].1;
            if step.is_some() || cache { bad.push(format!("lane {i}: {}{}", step.map_or(String::new(), |s| format!("logits differ from step {s} ({:?}) ", work[s])), if cache { "cache or state differs" } else { "" })); }
        }
        println!("{n:4} lanes, {nsteps} steps ({mixed} mixing prefill and decode spans): {}", if bad.is_empty() { "every lane identical to alone".to_string() } else { format!("{} lanes DIFFER, first: {}", bad.len(), bad[0]) });
        ok &= bad.is_empty();
    }
    println!("{}", if ok { "spans invariant: every lane, in every batch, gives exactly what it gives alone" } else { "SPANS NOT INVARIANT" });
    if !ok { std::process::exit(1); }
}

/// dev, layer 3's output gate (src/poolgate.rs): lanes run together with every pool slot outside the step poisoned must
/// give exactly the logits and greedy tokens of layer 2's path, each lane alone. Today the candidate is layer 2 itself,
/// over each lane's own buffers; the block cache replaces it in `cand` below. INK_POOLGATE_LANES (default 3),
/// INK_POOLGATE_CHUNKS (prefill chunk sizes, default 31,1,23,17: across a 64-position block boundary) and
/// INK_POOLGATE_DECODE (default 8) size the run; tests/ci.sh runs the default, tests/nightly.sh a long one.
/// INK_POOLGATE_SABOTAGE=1 is the reversed witness: one step's own slots are poisoned, and the gate must then fail.
fn poolgate(o: &Opts) {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or(d.into());
    let n: usize = env("INK_POOLGATE_LANES", "3").parse().expect("INK_POOLGATE_LANES");
    let chunks: Vec<usize> = env("INK_POOLGATE_CHUNKS", "31,1,23,17").split(',').map(|v| v.parse().expect("INK_POOLGATE_CHUNKS")).collect();
    let decode: usize = env("INK_POOLGATE_DECODE", "8").parse().expect("INK_POOLGATE_DECODE");
    let sabotage = std::env::var("INK_POOLGATE_SABOTAGE").is_ok();
    let start_sabotage = std::env::var("INK_POOLGATE_START_SABOTAGE").is_ok();
    let base: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let w = poolgate::Workload::from_tokens(&base, n, chunks.clone(), decode);
    let total_positions: usize = chunks.iter().sum::<usize>() + decode;
    let prompt_positions: usize = chunks.iter().sum();
    let eng = Engine::load(&o.model, o.ctx);
    let want = poolgate::reference(&poolgate::Layer2::new(&eng, n), &w);
    // the candidate: real Layer 3 BlockPool + BlockTable + PrefixTree over the same engine
    let cand = poolgate::Layer3Pool::new(&eng, n);
    let label = cand.candidate_label();
    println!("poolgate candidate: {label}");
    let got = poolgate::poisoned(&cand, &w, sabotage);
    let st = cand.stats();
    cand.shutdown();
    if sabotage && got.sabotaged.is_none() { println!("INK_POOLGATE_SABOTAGE: no step read earlier cache, nothing sabotaged"); std::process::exit(2); }
    // the poisoned lane must differ and no other lane may: lanes share no unpoisoned cache with a sabotaged lane,
    // so a second lane that differs read the poison through state the engine shares across lanes
    if let Some(s) = got.sabotaged {
        let leak = poolgate::leaked(&want, &got, s);
        if leak.is_empty() { println!("INK_POOLGATE_SABOTAGE: no other lane differs from layer 2's outputs"); }
        else { println!("LANE LEAK: lanes {leak:?} differ though only lane {s}'s slots were poisoned"); }
    }
    match poolgate::compare(&want, &got) {
        Ok(lines) => {
            for l in lines { println!("{l}"); }
            println!("layer 3 stats: taken={}, reused={}, sealed={}, shared_dedups={}, window_pruned={}, tails_dropped={}",
                st.slots_taken, st.slots_reused, st.blocks_sealed, st.shared_dedups, st.window_pruned, st.tails_dropped);
            if !sabotage {
                if let Err(err) = poolgate::verify_l3_workload_stats(&st, n, prompt_positions, total_positions) {
                    println!("POOL GATE STATS MISSING: {err}");
                    std::process::exit(1);
                }
                match poolgate::verify_prefix_start_points(&eng, &base, o.ctx, start_sabotage) {
                    Ok(msg) => println!("{msg}"),
                    Err(err) => { println!("PREFIX START-POINT GATE DIFFERS: {err}"); std::process::exit(1); }
                }
            }
            println!("pool gate identical [{label}]: poisoned slots, every lane gives exactly layer 2's outputs");
        }
        Err(lines) => { for l in lines { println!("{l}"); } println!("POOL GATE DIFFERS [{label}]"); std::process::exit(1); }
    }
}

/// dev, the cross-lane prefix gate (`inky serve`'s copy): a source lane prefilled with a shared head is copied
/// into a lane that prefills a different tail — `copy_lane_prefix` of the head, then the head's convolution
/// state, as serve admits such a request — and the result must be exactly a cold prefill of head + tail: the
/// same logits at every shared boundary and decode step, and the same cache and convolution rows at the end,
/// with the scratch poisoned before every step. Two tails share the one head. The copy must not move the
/// source. INK_XCOPY_SABOTAGE=1 is the reversed witness: it copies one head token too few while the tail
/// still starts at the head's end, and the gate must then fail (tests/ci.sh requires that failure).
fn xcopy(o: &Opts) {
    use engine::Span;
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(2);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let (head, tail, ndec) = (384usize, 256usize, 8usize);
    assert!(base.len() > head + tail + ndec + 1024, "need a longer --ref token list");
    let mut rot = base.clone(); rot.rotate_left(997);
    let stream = |which: usize| -> Vec<u32> { let mut v: Vec<u32> = base[..head].to_vec(); v.extend_from_slice(if which == 0 { &base[head..head + tail + ndec] } else { &rot[head..head + tail + ndec] }); v };
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let sabotage = std::env::var("INK_XCOPY_SABOTAGE").is_ok();
    let step = |lane: &Lane, p0: usize, t: usize, dec: bool| -> u64 {
        eng.poison_scratch();
        let cb = eng.gpu.cmd();
        if dec { eng.encode_decode(&cb, &[Span { lane, p0, t }], true, false); } else { eng.encode_spans(&cb, &[Span { lane, p0, t }], true, false); }
        cb.commit(); wait_gate(&cb, "xcopy step", p0);
        fnv(&*(if dec { eng.span_logits(0) } else { eng.logits() }))
    };
    let run_stream = |lane: &Lane, chunks: &[usize], from: usize, decode: bool| -> (Vec<(usize, u64)>, Vec<u64>) {
        let mut hs = Vec::new(); let mut p = from;
        for &t in chunks { hs.push((p + t, step(lane, p, t, false))); p += t; }
        let mut ds = Vec::new();
        if decode { for q in 0..ndec { ds.push(step(lane, head + tail + q, 1, true)); } }
        (hs, ds)
    };
    // cold: head and tail prefilled on one lane, split at 101, 384 and 640
    let cold = |which: usize| { let l = Lane::new(&eng); let r = eng.on(&l); r.reserve(head + tail + ndec + 1, false); r.set_tokens(0, &stream(which));
                                let out = run_stream(&l, &[101, 283, 256], 0, true); (l, out) };
    // the source: the shared head alone, split at 101 and 384; its state at the head's end is kept
    let src = Lane::new(&eng);
    { let r = eng.on(&src); r.reserve(head + tail + ndec + 1, false); r.set_tokens(0, &stream(0)[..head]); run_stream(&src, &[101, 283], 0, false); }
    let cs_head = eng.on(&src).get_cs();
    let src_before = eng.lane_hash(&src);
    let mut ok = true;
    for which in 0..2 {
        let (cl, (chs, cds)) = cold(which);
        let l = Lane::new(&eng); let r = eng.on(&l);
        r.reset(); r.reserve(head + tail + ndec + 1, false);
        eng.copy_lane_prefix(&src, &l, if sabotage { head - 1 } else { head });
        r.set_cs(&cs_head); r.set_tokens(0, &stream(which));
        let (hs, ds) = run_stream(&l, &[63, 193], head, true);
        let at = |v: &[(usize, u64)], p: usize| v.iter().find(|(q, _)| *q == p).map(|(_, h)| *h);
        let same = at(&hs, head + tail) == at(&chs, head + tail) && ds == cds && eng.lane_hash(&l) == eng.lane_hash(&cl);
        println!("tail {}: copied head {} + tail {} {} cold (logits at {}, decode steps, final state)", which, head, tail, if same { "matches" } else { "DIFFERS from" }, head + tail);
        ok &= same;
    }
    if eng.lane_hash(&src) != src_before { println!("the copy moved the source lane"); ok = false; }
    if sabotage { println!("INK_XCOPY_SABOTAGE: copied {} of {} head tokens, the tail still starts at {}", head - 1, head, head); }
    println!("{}", if ok { "xcopy lossless: two tails from one copied head match cold prefill bit for bit" } else { "XCOPY NOT LOSSLESS" });
    if !ok { std::process::exit(1); }
}

/// dev, Layer 4 gate (`docs/ROADMAP.md:141`): temperature = 1.0 with fixed seeds gives bit-identical
/// sampled token sequences with prefix caching + multi-lane scheduler batching vs uncached solo execution,
/// across both GPU Gumbel-max (`top_p = 1.0, top_k = 0`) and CPU top-p (`top_p = 0.95`) sampling paths.
/// `INK_TEMP1_SABOTAGE=1` is the negative control and must fail with `TEMP1 NOT EQUIVALENT`.
fn temp1(o: &Opts) {
    use engine::Span;
    use scheduler::{ServeLaneInput, ServeStepPlan, ServeStepPlanner};

    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(4);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference)
        .expect("--ref FILE.tokens")
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let (head, tail, ndec) = (384usize, 192usize, 24usize);
    assert!(base.len() > head + tail + ndec + 1024, "need a longer --ref token list");
    let sabotage = std::env::var("INK_TEMP1_SABOTAGE").is_ok();

    let mut rot1 = base.clone();
    rot1.rotate_left(997);
    let mut rot2 = base.clone();
    rot2.rotate_left(1999);
    // Prompts 0 & 1 share the 384-token prefix `head`; prompt 2 is an independent cold stream.
    let prompts: Vec<Vec<u32>> = vec![
        base[..head + tail].to_vec(),
        {
            let mut v = base[..head].to_vec();
            v.extend_from_slice(&rot1[head..head + tail]);
            v
        },
        rot2[..head + tail].to_vec(),
    ];
    let seeds = [0x1111_2222_3333_4444u64, 0xCAFE_BABE_DEAD_BEEFu64];
    let modes: [(&str, f32); 2] = [("gpu-gumbel (temp=1.0, top_p=1.0)", 1.0), ("cpu-sampler (temp=1.0, top_p=0.95)", 0.95)];
    let mut ok = true;

    for &(mode_name, top_p) in &modes {
        for &seed in &seeds {
            // 1. Uncached solo reference for each prompt (fresh lane, spans.len() == 1, poisoned scratch).
            let mut solo_seqs: Vec<Vec<u32>> = Vec::with_capacity(prompts.len());
            for (idx, prompt) in prompts.iter().enumerate() {
                let req_seed = seed.wrapping_add((idx as u64) * 0x9E37_79B9);
                let l = Lane::new(&eng);
                let r = eng.on(&l);
                r.reset();
                r.reserve(prompt.len() + ndec + 2, false);
                r.set_tokens(0, prompt);
                let mut sampler = Sampler::new(1.0, 0, top_p, eng.softcap, req_seed);
                let mut pos = 0usize;
                for &chunk in &[128usize, 256, 192] {
                    eng.poison_scratch();
                    let cb = eng.gpu.cmd();
                    eng.encode_spans(&cb, &[Span { lane: &l, p0: pos, t: chunk }], pos + chunk == prompt.len(), false);
                    cb.commit();
                    cb.waitUntilCompleted();
                    pos += chunk;
                }
                let first_tok = if sampler.uses_gpu_gumbel() {
                    let gcb = eng.gpu.cmd();
                    eng.encode_gumbel_solo(&gcb, &l, pos, req_seed, pos, 1.0, None);
                    gcb.commit();
                    gcb.waitUntilCompleted();
                    r.sampled_token_at(pos).expect("valid GPU sample")
                } else {
                    sampler.sample_next(&eng.logits()).expect("solo prefill sample")
                };
                let mut toks = vec![first_tok];
                while toks.len() < ndec {
                    let cur = *toks.last().unwrap();
                    r.set_tokens(pos, &[cur]);
                    eng.poison_scratch();
                    let cb = eng.gpu.cmd();
                    eng.encode_decode(&cb, &[Span { lane: &l, p0: pos, t: 1 }], true, false);
                    cb.commit();
                    cb.waitUntilCompleted();
                    let next_pos = pos + 1;
                    let tok = if sampler.uses_gpu_gumbel() {
                        let gcb = eng.gpu.cmd();
                        eng.encode_gumbel_span(&gcb, 0, &l, next_pos, req_seed, next_pos, 1.0, None);
                        gcb.commit();
                        gcb.waitUntilCompleted();
                        r.sampled_token_at(next_pos).expect("valid GPU sample")
                    } else {
                        sampler.sample_next(&eng.span_logits(0)).expect("solo decode sample")
                    };
                    pos = next_pos;
                    toks.push(tok);
                }
                solo_seqs.push(toks);
            }

            // 2. Prefix-cached + multi-lane scheduler-batched execution via ServeStepPlanner.
            let src = Lane::new(&eng);
            {
                let r = eng.on(&src);
                r.reset();
                r.reserve(head + tail + ndec + 2, false);
                r.set_tokens(0, &prompts[0][..head]);
                let mut p = 0;
                for &c in &[128usize, 256] {
                    eng.poison_scratch();
                    let cb = eng.gpu.cmd();
                    eng.encode_spans(&cb, &[Span { lane: &src, p0: p, t: c }], false, false);
                    cb.commit();
                    cb.waitUntilCompleted();
                    p += c;
                }
            }
            let cs_head = eng.on(&src).get_cs();

            struct ReqState {
                lane: Lane,
                n_past: usize,
                pf_end: usize,
                seed: u64,
                sampler: Sampler,
                toks: Vec<u32>,
            }
            let mut reqs: Vec<ReqState> = prompts
                .iter()
                .enumerate()
                .map(|(idx, prompt)| {
                    let req_seed = seed.wrapping_add((idx as u64) * 0x9E37_79B9);
                    let l = Lane::new(&eng);
                    let r = eng.on(&l);
                    r.reset();
                    r.reserve(prompt.len() + ndec + 2, false);
                    let start = if idx < 2 {
                        let copy_len = if sabotage && idx == 1 { head - 1 } else { head };
                        eng.copy_lane_prefix(&src, &l, copy_len);
                        r.set_cs(&cs_head);
                        head
                    } else {
                        0
                    };
                    r.set_tokens(0, prompt);
                    ReqState {
                        lane: l,
                        n_past: start,
                        pf_end: prompt.len(),
                        seed: req_seed,
                        sampler: Sampler::new(1.0, 0, top_p, eng.softcap, req_seed),
                        toks: Vec::with_capacity(ndec),
                    }
                })
                .collect();

            let mut planner = ServeStepPlanner::new(reqs.len(), 2);
            for i in 0..reqs.len() {
                planner.on_admit(i);
            }

            while reqs.iter().any(|r| r.toks.len() < ndec) {
                let inputs: Vec<ServeLaneInput> = reqs
                    .iter()
                    .map(|r| ServeLaneInput {
                        runnable: r.toks.len() < ndec,
                        n_past: r.n_past,
                        pf_end: r.pf_end,
                        has_pending_decode: !r.toks.is_empty() && r.n_past < r.pf_end + r.toks.len(),
                    })
                    .collect();
                let plan = planner
                    .plan_and_begin_step(&inputs, 256, 64)
                    .expect("temp1 scheduler must make progress");
                match plan {
                    ServeStepPlan::Prefill(sp_list) => {
                        let spans: Vec<Span> = sp_list
                            .iter()
                            .map(|&(i, p0, t)| Span { lane: &reqs[i].lane, p0, t })
                            .collect();
                        let want_logits: Vec<bool> = sp_list
                            .iter()
                            .map(|&(i, p0, t)| p0 + t == reqs[i].pf_end)
                            .collect();
                        let many = spans.len() > 1;
                        eng.poison_scratch();
                        let cb = eng.gpu.cmd();
                        eng.encode_spans_logits(&cb, &spans, &want_logits, false);
                        cb.commit();
                        cb.waitUntilCompleted();
                        let mut gpu_pf = Vec::new();
                        for (k, &(i, p0, t)) in sp_list.iter().enumerate() {
                            reqs[i].n_past = p0 + t;
                            if reqs[i].n_past == reqs[i].pf_end {
                                if reqs[i].sampler.uses_gpu_gumbel() {
                                    gpu_pf.push((k, i));
                                } else {
                                    let lg = if many { eng.span_logits(k) } else { eng.logits() };
                                    let tok = reqs[i].sampler.sample_next(&lg).expect("batched pf sample");
                                    reqs[i].toks.push(tok);
                                }
                            }
                        }
                        if !gpu_pf.is_empty() {
                            let gcb = eng.gpu.cmd();
                            for &(k, i) in &gpu_pf {
                                let p = reqs[i].n_past;
                                if many {
                                    eng.encode_gumbel_span(&gcb, k, &reqs[i].lane, p, reqs[i].seed, p, 1.0, None);
                                } else {
                                    eng.encode_gumbel_solo(&gcb, &reqs[i].lane, p, reqs[i].seed, p, 1.0, None);
                                }
                            }
                            gcb.commit();
                            gcb.waitUntilCompleted();
                            for &(_, i) in &gpu_pf {
                                let p = reqs[i].n_past;
                                let tok = eng.on(&reqs[i].lane).sampled_token_at(p).expect("valid GPU sample");
                                reqs[i].toks.push(tok);
                            }
                        }
                    }
                    ServeStepPlan::Decode(dec_lanes) => {
                        for &i in &dec_lanes {
                            let tok = *reqs[i].toks.last().unwrap();
                            eng.on(&reqs[i].lane).set_tokens(reqs[i].n_past, &[tok]);
                        }
                        let spans: Vec<Span> = dec_lanes
                            .iter()
                            .map(|&i| Span { lane: &reqs[i].lane, p0: reqs[i].n_past, t: 1 })
                            .collect();
                        eng.poison_scratch();
                        let cb = eng.gpu.cmd();
                        eng.encode_decode(&cb, &spans, true, false);
                        cb.commit();
                        cb.waitUntilCompleted();
                        let gpu_dec: Vec<(usize, usize)> = dec_lanes
                            .iter()
                            .enumerate()
                            .filter(|&(_, &i)| reqs[i].sampler.uses_gpu_gumbel())
                            .map(|(k, &i)| (k, i))
                            .collect();
                        if !gpu_dec.is_empty() {
                            let gcb = eng.gpu.cmd();
                            for &(k, i) in &gpu_dec {
                                let next_pos = reqs[i].n_past + 1;
                                eng.encode_gumbel_span(&gcb, k, &reqs[i].lane, next_pos, reqs[i].seed, next_pos, 1.0, None);
                            }
                            gcb.commit();
                            gcb.waitUntilCompleted();
                        }
                        for (k, &i) in dec_lanes.iter().enumerate() {
                            reqs[i].n_past += 1;
                            let tok = if reqs[i].sampler.uses_gpu_gumbel() {
                                eng.on(&reqs[i].lane).sampled_token_at(reqs[i].n_past).expect("valid GPU sample")
                            } else {
                                reqs[i].sampler.sample_next(&eng.span_logits(k)).expect("batched dec sample")
                            };
                            reqs[i].toks.push(tok);
                        }
                    }
                }
                let _ = planner.complete_step();
            }

            let mut seed_ok = true;
            for (i, (solo, req)) in solo_seqs.iter().zip(&reqs).enumerate() {
                if solo != &req.toks {
                    let first_diff = solo.iter().zip(&req.toks).position(|(a, b)| a != b);
                    println!(
                        "{mode_name} seed=0x{seed:016x} req {i}: DIFFERS at token {:?} (solo={:?} vs cached={:?})",
                        first_diff,
                        first_diff.map(|k| solo[k]),
                        first_diff.map(|k| req.toks[k])
                    );
                    seed_ok = false;
                }
            }
            if seed_ok {
                println!(
                    "{mode_name} seed=0x{seed:016x}: 3 lanes x {ndec} tokens (2 prefix-cached + 1 cold) IDENTICAL to uncached solo"
                );
            }
            ok &= seed_ok;
        }
    }
    if sabotage {
        println!("INK_TEMP1_SABOTAGE: copied {} of {} head tokens on lane 1", head - 1, head);
    }
    println!(
        "{}",
        if ok {
            "temp1 equivalent: temperature=1.0 fixed-seed cached+scheduled outputs match uncached solo bit for bit (GPU Gumbel + CPU top-p)"
        } else {
            "TEMP1 NOT EQUIVALENT"
        }
    );
    if !ok {
        std::process::exit(1);
    }
}

/// dev, layer 2's performance gate: decode steps over N lanes at a given context. Each context is prefilled once (the
/// --ref tokens repeated) and copied into every lane; lane i then decodes its own stream (the --ref tokens rotated by
/// 97 i), so routing differs across lanes as it would in use. INK_CTXS and INK_LANES list what to measure.
fn lanebench(o: &Opts) {
    use engine::Span;
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let ctxs: Vec<usize> = std::env::var("INK_CTXS").unwrap_or("2048,20480".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let counts: Vec<usize> = std::env::var("INK_LANES").unwrap_or("1,8,16,32,64".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let steps: usize = std::env::var("INK_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    println!("{:>6} {:>5} {:>9} {:>10} {:>9} {:>10}", "ctx", "lanes", "ms/step", "total t/s", "per lane", "GPU GiB");
    for &c in &ctxs {
        let len = c + steps + 1;
        let toks: Vec<u32> = (0..len).map(|i| base[i % base.len()]).collect();
        let src = Lane::new(&eng);
        { let r = eng.on(&src); r.reserve(len, false); r.set_tokens(0, &toks);
          let mut p = 0; while p < c { let t = (c - p).min(eng.chunk); let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &[Span { lane: &src, p0: p, t }], false, false); cb.commit(); cb.waitUntilCompleted(); p += t; } }
        // the baseline: one stream on the single-stream decode path at this context
        {
            let r = eng.on(&src);
            let tok = |q: usize| { let cb = eng.gpu.cmd(); r.encode_step(&cb, c + q, true, false); cb.commit(); cb.waitUntilCompleted(); };
            tok(0);
            let t0 = std::time::Instant::now();
            for q in 0..steps { tok(q); }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
            println!("{c:>6} {:>5} {ms:>9.1} {:>10.0} {:>9.1} {:>10.1}  (one stream, decode path)", 1, 1e3 / ms, 1e3 / ms, gib(eng.gpu_mem().0));
            // (the decode path advanced src's state by `steps` tokens: harmless here, this measures time only)
        }
        let nmax = *counts.iter().max().unwrap();
        let lanes: Vec<Lane> = (0..nmax).map(|i| {
            let l = Lane::new(&eng); let r = eng.on(&l); r.reserve(len, false);
            let mut t = toks.clone();
            for q in 0..=steps { t[c + q] = base[(97 * i + q) % base.len()]; }
            r.set_tokens(0, &t); eng.copy_lane(&src, &l); l
        }).collect();
        eng.ensure_dbatch(nmax);
        for &n in &counts {
            // a warm-up step, then the timed steps: lanes decode positions c, c + 1, ... (the context grows by `steps` at most)
            let step = |q: usize| { let sp: Vec<Span> = lanes[..n].iter().map(|l| Span { lane: l, p0: c + q, t: 1 }).collect();
                                    let cb = eng.gpu.cmd(); if std::env::var("INK_SPANS_DEC").is_ok() { eng.encode_spans(&cb, &sp, true, false) } else { eng.encode_decode(&cb, &sp, true, false) }; cb.commit(); cb.waitUntilCompleted(); };
            step(0);
            let t0 = std::time::Instant::now();
            for q in 0..steps { step(q); }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / steps as f64;
            println!("{c:>6} {n:>5} {ms:>9.1} {:>10.0} {:>9.1} {:>10.1}", n as f64 * 1e3 / ms, 1e3 / ms, gib(eng.gpu_mem().0));
        }
    }
    let (_, ws) = eng.gpu_mem();
    println!("recommended GPU working set: {:.1} GiB", gib(ws));
}

/// dev, the batched decode gate: N lanes on different streams (the --ref tokens rotated by 97 i), each prefilled to
/// INK_PF tokens, then INK_DECODE decode steps, twice: each lane alone through the single-stream decode path, and all
/// lanes together through `encode_dbatch`. Every lane's logits at every step and its final cache and convolution rows
/// must be bit-identical; the time per step of both is printed. INK_LANES lists the lane counts.
/// lanes the decode-batch gate can use
const MAX_SPANS_DB: usize = 128;
fn dbatch(o: &Opts) {
    use engine::Span;
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(MAX_SPANS_DB);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference).expect("--ref FILE.tokens").lines().map(|l| l.trim().parse().unwrap()).collect();
    let pf: usize = std::env::var("INK_PF").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let nd: usize = std::env::var("INK_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let counts: Vec<usize> = std::env::var("INK_LANES").unwrap_or("1,2,8".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let len = pf + nd + 1;
    let stream = |i: usize| -> Vec<u32> { (0..len).map(|q| base[(97 * i + q) % base.len()]).collect() };
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let prefilled = |i: usize| -> Lane {
        let l = Lane::new(&eng); let r = eng.on(&l); r.reserve(len, false); r.set_tokens(0, &stream(i));
        let mut p = 0; while p < pf { let t = (pf - p).min(eng.chunk); let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &[Span { lane: &l, p0: p, t }], false, false); cb.commit(); cb.waitUntilCompleted(); p += t; }
        l
    };
    let nmax = *counts.iter().max().unwrap();
    // INK_DB_OLD: the decode-path batch (encode_dbatch) against the single-stream decode path; otherwise decode steps
    // (encode_decode) of N lanes against each lane's decode steps alone
    let old = std::env::var("INK_DB_OLD").is_ok();
    // dev: INK_DB_PROF=1 times the batch with each kernel class skipped in turn (results then differ; timing only)
    if std::env::var("INK_DB_PROF").is_ok() {
        for &n in &counts {
            let lanes: Vec<Lane> = (0..n).map(prefilled).collect();
            eng.ensure_dbatch(n);
            let b: Vec<(&Lane, usize)> = lanes.iter().map(|l| (l, pf)).collect();
            let sp: Vec<Span> = b.iter().map(|(l, p)| Span { lane: l, p0: *p, t: 1 }).collect();
            let time = |skip: u32| { eng.skip.set(skip); let mut best = f64::MAX; for _ in 0..3 { let cb = eng.gpu.cmd(); if old { eng.encode_dbatch(&cb, &b, true); } else { eng.encode_decode(&cb, &sp, true, false); } cb.commit(); cb.waitUntilCompleted();
                best = best.min((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3); } eng.skip.set(0); best };
            let full = time(0);
            println!("{n} lanes: {full:.1} ms GPU per step");
            for (c, name) in engine::CLS_NAMES.iter().enumerate() { let d = full - time(1 << c); if d > 0.5 { println!("  {name:>11}: {d:7.1} ms ({:4.1}%)", d / full * 100.0); } }
        }
        return;
    }
    // alone, through the decode path
    let mut alone = Vec::new();
    let mut t_alone = 0.0;
    for i in 0..nmax {
        let l = prefilled(i); let r = eng.on(&l);
        let mut hs = Vec::new();
        let t0 = std::time::Instant::now();
        for q in 0..nd {
            let cb = eng.gpu.cmd();
            if old { r.encode_step(&cb, pf + q, true, false); } else { eng.encode_decode(&cb, &[Span { lane: &l, p0: pf + q, t: 1 }], true, false); }
            cb.commit(); cb.waitUntilCompleted(); hs.push(fnv(&*(if old { eng.logits() } else { eng.span_logits(0) })));
        }
        t_alone += t0.elapsed().as_secs_f64();
        alone.push((hs, eng.lane_hash(&l)));
    }
    let ms_alone = t_alone * 1e3 / (nmax * nd) as f64;
    println!("alone ({}): {ms_alone:.1} ms/token, {:.0} tok/s", if old { "decode path" } else { "decode steps of one lane" }, 1e3 / ms_alone);
    let sabotage = std::env::var("INK_DBATCH_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut sabotaged = false;
    let mut ok = true;
    for &n in &counts {
        let lanes: Vec<Lane> = (0..n).map(prefilled).collect();
        eng.ensure_dbatch(n);
        let mut hs: Vec<Vec<u64>> = vec![Vec::new(); n];
        let t0 = std::time::Instant::now();
        for q in 0..nd {
            let mut b: Vec<(&Lane, usize)> = lanes.iter().map(|l| (l, pf + q)).collect();
            if sabotage && !sabotaged && n >= 2 && q == 0 {
                println!("INK_DBATCH_SABOTAGE: lane 0 decode span bound to lane 1 at step 0");
                b[0].0 = &lanes[1];
                sabotaged = true;
            }
            let cb = eng.gpu.cmd();
            if old { eng.encode_dbatch(&cb, &b, true); } else { let sp: Vec<Span> = b.iter().map(|(l, p)| Span { lane: l, p0: *p, t: 1 }).collect(); eng.encode_decode(&cb, &sp, true, false); }
            cb.commit(); cb.waitUntilCompleted();
            for j in 0..n { hs[j].push(fnv(&*(if old { eng.dbatch_logits(j) } else { eng.span_logits(j) }))); }
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / nd as f64;
        let bad: Vec<usize> = (0..n).filter(|&j| hs[j] != alone[j].0 || eng.lane_hash(&lanes[j]) != alone[j].1).collect();
        println!("{n:4} lanes: {ms:7.1} ms/step, {:6.0} tok/s total, {:5.1} per lane: {}", n as f64 * 1e3 / ms, 1e3 / ms,
                 if bad.is_empty() { "every lane identical to alone".to_string() } else { format!("{} lanes DIFFER (first {})", bad.len(), bad[0]) });
        ok &= bad.is_empty();
    }
    if sabotage && !sabotaged {
        eprintln!("INK_DBATCH_SABOTAGE: need a lane count >= 2 in INK_LANES");
        std::process::exit(2);
    }
    println!("{}", if ok { "dbatch exact: every lane in the batch gives exactly what it gives decoding alone" } else { "DBATCH NOT EXACT" });
    if !ok { std::process::exit(1); }
}

/// Layer 3 shared-KV grouped decode parity + speedup benchmark (`k_attn_grp_dec` G_MAX=8 vs `INK_NO_GRP_DEC=1`).
/// Prefills a shared prefix of `INK_PREFIX` tokens (default 8192) once, forks `INK_LANES` (default 16,32) child lanes
/// with staggered private suffixes, and verifies 100% bit-for-bit identical logits across all lanes and steps while
/// measuring GPU ms/step speedup.
fn sharedkv(o: &Opts) {
    use poolgate::{Layer3Pool, StepPath};
    let prefix_len: usize = std::env::var("INK_PREFIX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    assert!(prefix_len >= 64 && prefix_len % 64 == 0, "INK_PREFIX must be a non-zero multiple of 64");
    let nd: usize = std::env::var("INK_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let counts: Vec<usize> = std::env::var("INK_LANES").unwrap_or("16,32".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let max_suffix = 256usize;
    let ctx = o.ctx.max((prefix_len + max_suffix + nd + 64).div_ceil(64) * 64);
    let eng = Engine::load(&o.model, ctx);
    let base: Vec<u32> = std::fs::read_to_string(&o.reference)
        .expect("--ref FILE.tokens")
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let prefix: Vec<u32> = (0..prefix_len).map(|i| base[i % base.len()]).collect();
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let argmax = |v: &[f32]| -> u32 {
        v.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i as u32).unwrap_or(0)
    };

    let run_mode = |n: usize, no_grp: bool| -> (Vec<Vec<u64>>, f64, f64) {
        eng.set_no_grp_dec(no_grp);
        let pool = Layer3Pool::with_persistent_starts(&eng, n + 1);
        // Donor prefills `prefix_len` tokens (multiple of 64) so blocks `0..prefix_len/64` seal and register a PROMPT start point at `prefix_len`.
        let donor_toks = prefix.clone();
        let (donor, matched0) = pool.new_lane_matched(&donor_toks, ctx);
        assert_eq!(matched0, 0);
        let mut p = 0usize;
        while p < donor_toks.len() {
            let t = (donor_toks.len() - p).min(eng.chunk);
            pool.prefill(&[(donor, p, t)]);
            p += t;
        }
        // Fork `n` child lanes sharing `prefix_len` with staggered private suffixes.
        let mut child_ids = Vec::with_capacity(n);
        let mut child_pos = Vec::with_capacity(n);
        for i in 0..n {
            let sfx_len = 16 + (i * 19) % 192;
            let mut toks = prefix.clone();
            for k in 0..sfx_len {
                toks.push(base[(97 * (i + 1) + k) % base.len()]);
            }
            let (lid, matched) = pool.new_lane_matched(&toks, ctx);
            assert_eq!(matched, prefix_len, "child lane {i} must share full {prefix_len}-token prefix");
            pool.prefill(&[(lid, matched, sfx_len)]);
            let first_tok = argmax(&pool.logits(0));
            pool.set_token(lid, toks.len(), first_tok);
            child_ids.push(lid);
            child_pos.push(toks.len());
        }
        pool.end_lane(donor);
        pool.poison_free_slots();

        let mut hashes: Vec<Vec<u64>> = vec![Vec::with_capacity(nd); n];
        let mut gpu_times: Vec<f64> = Vec::with_capacity(nd);
        let t0 = std::time::Instant::now();
        for _step in 0..nd {
            let sp: Vec<(usize, usize)> = child_ids.iter().copied().zip(child_pos.iter().copied()).collect();
            let ms = pool.decode_timed(&sp);
            gpu_times.push(ms);
            for j in 0..n {
                let lg = pool.logits(j);
                hashes[j].push(fnv(&lg));
                let next_tok = argmax(&lg);
                child_pos[j] += 1;
                pool.set_token(child_ids[j], child_pos[j], next_tok);
            }
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3 / nd as f64;
        gpu_times.sort_by(|a, b| a.total_cmp(b));
        let med_gpu_ms = gpu_times[gpu_times.len() / 2];
        pool.shutdown();
        (hashes, med_gpu_ms, wall_ms)
    };

    // Interleaved A/B timing on one warm live pool (eliminates thermal/paging drift between count=1 and G_MAX=8).
    let bench_ab = |n: usize| -> (f64, f64, f64, f64) {
        let pool = Layer3Pool::with_persistent_starts(&eng, n + 1);
        let donor_toks = prefix.clone();
        let (donor, _) = pool.new_lane_matched(&donor_toks, ctx);
        let mut p = 0usize;
        while p < donor_toks.len() {
            let t = (donor_toks.len() - p).min(eng.chunk);
            pool.prefill(&[(donor, p, t)]);
            p += t;
        }
        let mut child_ids = Vec::with_capacity(n);
        let mut child_pos = Vec::with_capacity(n);
        for i in 0..n {
            let sfx_len = 16 + (i * 19) % 192;
            let mut toks = prefix.clone();
            for k in 0..sfx_len { toks.push(base[(97 * (i + 1) + k) % base.len()]); }
            let (lid, matched) = pool.new_lane_matched(&toks, ctx);
            pool.prefill(&[(lid, matched, sfx_len)]);
            pool.set_token(lid, toks.len(), base[(i + 7) % base.len()]);
            child_ids.push(lid);
            child_pos.push(toks.len());
        }
        pool.end_lane(donor);
        let sp: Vec<(usize, usize)> = child_ids.iter().copied().zip(child_pos.iter().copied()).collect();
        // Warmup 2 steps
        eng.set_no_grp_dec(false); let _ = pool.bench_decode_ms(&sp);
        eng.set_no_grp_dec(true);  let _ = pool.bench_decode_ms(&sp);
        let (mut best_nogrp, mut best_grp, mut best_noattn) = (f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..5 {
            eng.skip.set(0);
            eng.set_no_grp_dec(true);
            best_nogrp = best_nogrp.min(pool.bench_decode_ms(&sp));
            eng.set_no_grp_dec(false);
            best_grp = best_grp.min(pool.bench_decode_ms(&sp));
            eng.skip.set(1 << 3); // skip C_ATTN
            best_noattn = best_noattn.min(pool.bench_decode_ms(&sp));
            eng.skip.set(0);
        }
        pool.shutdown();
        let attn_nogrp = (best_nogrp - best_noattn).max(0.0);
        let attn_grp = (best_grp - best_noattn).max(0.0);
        (best_nogrp, best_grp, attn_nogrp, attn_grp)
    };

    let sabotage = std::env::var("INK_SHAREDKV_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let mut all_ok = true;
    // Boundary-end parity check (`docs/SHARED-PREFIX.md` §4 & `t-later-prefix-reads`):
    // ends at 63/64/65, 255/256/257, and 1023/1024/1025 comparing grouped `k_attn_grp_dec` against
    // private fixed-split Layer 2 `encode_decode` AND Layer 3 `INK_NO_GRP_DEC=1` bit-for-bit!
    if prefix_len >= 1024 {
        eng.ensure_batch(eng.chunk);
        eng.ensure_dbatch(4);
        let boundary_ends = [63usize, 64, 65, 255, 256, 257, 1023, 1024, 1025];
        for &end in &boundary_ends {
            let pos = end - 1;
            let shared_b = (pos / 64) * 64;
            let b_toks_a: Vec<u32> = (0..=pos).map(|q| base[q % base.len()]).collect();
            let mut b_toks_b = b_toks_a.clone();
            if pos >= shared_b && pos > 0 && shared_b < pos {
                b_toks_b[pos] = base[(pos + 37) % base.len()];
            }
            // Private Layer 2 fixed-split reference on lane A and lane B.
            let priv_hash = |toks: &[u32]| -> u64 {
                let l = Lane::new(&eng);
                let r = eng.on(&l);
                r.reserve(end + 4, false);
                r.set_tokens(0, toks);
                if pos > 0 {
                    let cb = eng.gpu.cmd();
                    eng.encode_spans(&cb, &[engine::Span { lane: &l, p0: 0, t: pos }], false, false);
                    cb.commit();
                    wait_gate(&cb, "sharedkv boundary l2 pf", 0);
                }
                let cb = eng.gpu.cmd();
                eng.encode_decode(&cb, &[engine::Span { lane: &l, p0: pos, t: 1 }], true, false);
                cb.commit();
                wait_gate(&cb, "sharedkv boundary l2 dec", pos);
                fnv(&eng.span_logits(0))
            };
            let want_a = priv_hash(&b_toks_a);
            let want_b = priv_hash(&b_toks_b);

            for no_grp in [true, false] {
                eng.set_no_grp_dec(no_grp);
                let pool = Layer3Pool::with_persistent_starts(&eng, 3);
                if shared_b >= 64 {
                    let (donor, _) = pool.new_lane_matched(&b_toks_a[..shared_b], ctx);
                    pool.prefill(&[(donor, 0, shared_b)]);
                    pool.end_lane(donor);
                }
                let (la, ma) = pool.new_lane_matched(&b_toks_a, ctx);
                let (lb, mb) = pool.new_lane_matched(&b_toks_b, ctx);
                if pos > ma {
                    pool.prefill(&[(la, ma, pos - ma)]);
                }
                if pos > mb {
                    pool.prefill(&[(lb, mb, pos - mb)]);
                }
                pool.decode(&[(la, pos), (lb, pos)]);
                let mut got_a = fnv(&pool.logits(0));
                let got_b = fnv(&pool.logits(1));
                if sabotage && !no_grp && end == 1025 {
                    println!("INK_SHAREDKV_SABOTAGE: perturbing grouped boundary hash at end={end}");
                    got_a ^= 1;
                }
                if got_a != want_a || got_b != want_b {
                    println!("boundary end={end} no_grp={no_grp}: DIFFERS from private Layer 2 fixed-split");
                    all_ok = false;
                }
                pool.shutdown();
            }
        }
        println!("boundary ends {boundary_ends:?}: grouped and solo Layer 3 match private Layer 2 fixed-split bit-for-bit");
    }
    for &n in &counts {
        let (h_nogrp, _, _) = run_mode(n, true);
        let (mut h_grp, _, _) = run_mode(n, false);
        if sabotage && prefix_len < 1024 && !h_grp.is_empty() && !h_grp[0].is_empty() {
            println!("INK_SHAREDKV_SABOTAGE: perturbing grouped lane 0 step 0 hash");
            h_grp[0][0] ^= 1;
        }
        let bad: Vec<usize> = (0..n).filter(|&j| h_grp[j] != h_nogrp[j]).collect();
        let (gpu_nogrp, gpu_grp, attn_nogrp, attn_grp) = bench_ab(n);
        let speedup = gpu_nogrp / gpu_grp;
        let attn_speedup = if attn_grp > 0.01 { attn_nogrp / attn_grp } else { 0.0 };
        println!(
            "prefix={prefix_len} lanes={n:2} steps={nd}: count=1 {gpu_nogrp:6.2} ms GPU (attn {attn_nogrp:5.2} ms) -> G_MAX=8 {gpu_grp:6.2} ms GPU (attn {attn_grp:5.2} ms) | {speedup:.2}x step ({attn_speedup:.2}x attn) | {}",
            if bad.is_empty() { "100% bit-identical".to_string() } else { format!("{} lanes DIFFER (first {})", bad.len(), bad[0]) }
        );
        all_ok &= bad.is_empty();
    }
    eng.set_no_grp_dec(false);
    if all_ok {
        println!("sharedkv exact: every grouped lane and step is 100% bit-for-bit identical to INK_NO_GRP_DEC=1");
    } else {
        println!("SHAREDKV PARITY FAILED");
        std::process::exit(1);
    }
}

/// Matched shared-prefix replay gate & benchmark (`t-later-prefix-gate` & `docs/SHARED-PREFIX.md` §5):
/// Compares private unshared execution (`Layer3Pool::new` + `INK_NO_GRP_DEC=1`) against shared-prefix
/// execution (`Layer3Pool::with_persistent_starts` + `PrefixTree` block deduplication + `k_attn_grp_dec`)
/// across singleton, unrelated cold lanes, staggered same-card turns, and 4/8/16 concurrent 8K shared-head
/// lanes, reporting live K/V pool MiB, peak Metal allocation, end-to-end step time, decode GPU ms/step,
/// aggregate decode tok/s, and verifying 100% bit-for-bit logit parity.
fn prefixbench(o: &Opts) {
    use poolgate::{Layer3Pool, StepPath};

    let prefix_len: usize = std::env::var("INK_PREFIX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    assert!(prefix_len >= 64 && prefix_len % 64 == 0, "INK_PREFIX must be a multiple of 64");
    let nd: usize = std::env::var("INK_DECODE").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let ctx = o.ctx.max((prefix_len + 512 + nd + 64).div_ceil(64) * 64);
    let eng = Engine::load(&o.model, ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(16);
    let base = load_gate_tokens(&eng, o);
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(i, _)| i as u32).unwrap_or(0);
    let mib = |b: usize| b as f64 / 1048576.0;

    #[derive(Clone)]
    struct Case {
        name: &'static str,
        prompts: Vec<Vec<u32>>,
        shared_prefix_toks: usize,
    }

    let shared_head: Vec<u32> = (0..prefix_len).map(|i| base[i % base.len()]).collect();
    let card_head_len = 2048usize.min(prefix_len);
    let card_head: Vec<u32> = shared_head[..card_head_len].to_vec();

    let mut cases = vec![
        Case {
            name: "singleton-1x8k",
            prompts: vec![{
                let mut v = shared_head.clone();
                v.extend((0..128).map(|k| base[(31 + k) % base.len()]));
                v
            }],
            shared_prefix_toks: 0,
        },
        Case {
            name: "unrelated-4x2k",
            prompts: (0..4)
                .map(|i| (0..2176).map(|k| base[(199 * (i + 1) + k) % base.len()]).collect())
                .collect(),
            shared_prefix_toks: 0,
        },
        Case {
            name: "staggered-card-4x2k",
            prompts: (0..4)
                .map(|i| {
                    let mut v = card_head.clone();
                    let extra = 64 * (i + 1) + 24;
                    v.extend((0..extra).map(|k| base[(73 + k) % base.len()]));
                    v
                })
                .collect(),
            shared_prefix_toks: card_head_len,
        },
    ];
    for &n in &[4usize, 8, 16] {
        cases.push(Case {
            name: match n {
                4 => "shared-head-4x8k",
                8 => "shared-head-8x8k",
                _ => "shared-head-16x8k",
            },
            prompts: (0..n)
                .map(|i| {
                    let mut v = shared_head.clone();
                    let sfx = 32 + (i * 19) % 192;
                    v.extend((0..sfx).map(|k| base[(97 * (i + 1) + k) % base.len()]));
                    v
                })
                .collect(),
            shared_prefix_toks: prefix_len,
        });
    }

    println!(
        "{:<20} {:>5} {:>11} {:>11} {:>10} {:>11} {:>11} {:>11} {:>10} {:>8}",
        "case", "lanes", "priv KV MiB", "shrd KV MiB", "peak MiB", "e2e priv ms", "e2e shrd ms", "dec ms(p->s)", "dec tok/s", "parity"
    );

    let mut all_ok = true;
    for case in &cases {
        let n = case.prompts.len();
        eng.ensure_dbatch(n);

        // 1. Private unshared run (`Layer3Pool::new` + `no_grp_dec = true`).
        let (priv_hashes, priv_kv_bytes, priv_e2e_ms, priv_dec_ms) = {
            eng.set_no_grp_dec(true);
            let pool = Layer3Pool::new(&eng, n);
            let t_e2e = std::time::Instant::now();
            let mut lids = Vec::with_capacity(n);
            let mut cur_pos = Vec::with_capacity(n);
            for prompt in &case.prompts {
                let lid = pool.new_lane(prompt, ctx);
                let mut p = 0usize;
                while p < prompt.len() {
                    let c = (prompt.len() - p).min(eng.chunk);
                    pool.prefill(&[(lid, p, c)]);
                    p += c;
                }
                let first = argmax(&pool.logits(0));
                pool.set_token(lid, prompt.len(), first);
                lids.push(lid);
                cur_pos.push(prompt.len());
            }
            let (full_b, win_b, _) = pool.kv_memory_bytes();
            let mut hashes = vec![Vec::with_capacity(nd); n];
            let mut dec_times = Vec::with_capacity(nd);
            for _ in 0..nd {
                let sp: Vec<(usize, usize)> = lids.iter().copied().zip(cur_pos.iter().copied()).collect();
                dec_times.push(pool.decode_timed(&sp));
                for j in 0..n {
                    let lg = pool.logits(j);
                    hashes[j].push(fnv(&lg));
                    let tok = argmax(&lg);
                    cur_pos[j] += 1;
                    pool.set_token(lids[j], cur_pos[j], tok);
                }
            }
            let e2e = t_e2e.elapsed().as_secs_f64() * 1e3;
            dec_times.sort_by(|a, b| a.total_cmp(b));
            let med_dec = dec_times[dec_times.len() / 2];
            pool.shutdown();
            (hashes, full_b + win_b, e2e, med_dec)
        };

        // 2. Shared-prefix run (`Layer3Pool::with_persistent_starts` + `no_grp_dec = false`).
        let (shrd_hashes, shrd_kv_bytes, peak_gpu_bytes, shrd_e2e_ms, shrd_dec_ms) = {
            eng.set_no_grp_dec(false);
            let pool = Layer3Pool::with_persistent_starts(&eng, n + 1);
            let t_e2e = std::time::Instant::now();
            if case.shared_prefix_toks >= 64 {
                let seed = &case.prompts[0][..case.shared_prefix_toks];
                let (donor, _) = pool.new_lane_matched(seed, ctx);
                let mut p = 0usize;
                while p < seed.len() {
                    let c = (seed.len() - p).min(eng.chunk);
                    pool.prefill(&[(donor, p, c)]);
                    p += c;
                }
                pool.end_lane(donor);
            }
            let mut lids = Vec::with_capacity(n);
            let mut cur_pos = Vec::with_capacity(n);
            for prompt in &case.prompts {
                let (lid, matched) = pool.new_lane_matched(prompt, ctx);
                let mut p = matched;
                while p < prompt.len() {
                    let c = (prompt.len() - p).min(eng.chunk);
                    pool.prefill(&[(lid, p, c)]);
                    p += c;
                }
                let first = argmax(&pool.logits(0));
                pool.set_token(lid, prompt.len(), first);
                lids.push(lid);
                cur_pos.push(prompt.len());
            }
            let (full_b, win_b, _) = pool.kv_memory_bytes();
            let peak_alloc = eng.gpu_mem().0;
            let mut hashes = vec![Vec::with_capacity(nd); n];
            let mut dec_times = Vec::with_capacity(nd);
            for _ in 0..nd {
                let sp: Vec<(usize, usize)> = lids.iter().copied().zip(cur_pos.iter().copied()).collect();
                dec_times.push(pool.decode_timed(&sp));
                for j in 0..n {
                    let lg = pool.logits(j);
                    hashes[j].push(fnv(&lg));
                    let tok = argmax(&lg);
                    cur_pos[j] += 1;
                    pool.set_token(lids[j], cur_pos[j], tok);
                }
            }
            let e2e = t_e2e.elapsed().as_secs_f64() * 1e3;
            dec_times.sort_by(|a, b| a.total_cmp(b));
            let med_dec = dec_times[dec_times.len() / 2];
            pool.shutdown();
            (hashes, full_b + win_b, peak_alloc, e2e, med_dec)
        };

        let same = priv_hashes == shrd_hashes;
        if case.shared_prefix_toks >= 64 && n >= 2 {
            assert!(
                shrd_kv_bytes < priv_kv_bytes,
                "{}: shared KV ({shrd_kv_bytes}) did not save physical memory vs private ({priv_kv_bytes})",
                case.name
            );
        }
        all_ok &= same;
        let agg_tok_s = (n as f64) * 1e3 / shrd_dec_ms;
        println!(
            "{:<20} {:>5} {:>11.1} {:>11.1} {:>10.1} {:>11.1} {:>11.1} {:>5.1}->{:<5.1} {:>10.1} {:>8}",
            case.name,
            n,
            mib(priv_kv_bytes),
            mib(shrd_kv_bytes),
            mib(peak_gpu_bytes),
            priv_e2e_ms,
            shrd_e2e_ms,
            priv_dec_ms,
            shrd_dec_ms,
            agg_tok_s,
            if same { "exact" } else { "DIFF" }
        );
    }
    eng.set_no_grp_dec(false);
    if all_ok {
        println!("prefixbench exact: shared-prefix KV bytes, peak allocation, and step times recorded with 100% bitwise parity");
    } else {
        println!("PREFIXBENCH PARITY FAILED");
        std::process::exit(1);
    }
}

fn load_gate_tokens(eng: &Engine, o: &Opts) -> Vec<u32> {
    if !o.reference.is_empty() {
        let path = if std::path::Path::new(&o.reference).exists() {
            o.reference.clone()
        } else {
            format!("{}.tokens", o.reference)
        };
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read token file {path}: {e}"))
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("valid token id"))
            .collect()
    } else {
        let tk = Tokenizer::load(&eng.g);
        tk.encode(&first_turn(o, &o.prompt), true)
    }
}

/// Independent `f64` CPU reference for `k_loopcd_blend` (Liu et al. 2026, Eqs. 3-4 & Appendix A.1-A.3):
/// computes softcapped `s3`, `s1`, exact top-2 probabilities `(p1, p2)` of `softmax(s3)`, adaptive
/// `omega_t = omega_max * clamp(1 - (p1 - p2), 0, 1)`, and `s3 + omega_t * (s3 - s1)`.
fn cpu_loopcd_reference(z3_raw: &[f32], z1_raw: &[f32], softcap: f32, omega_max: f32) -> (Vec<f32>, f32) {
    assert_eq!(z3_raw.len(), z1_raw.len());
    if omega_max <= 0.0 {
        return (z3_raw.to_vec(), 0.0);
    }
    let c = softcap as f64;
    let cap = |x: f32| -> f64 {
        let xd = x as f64;
        if c > 0.0 { c * (xd / c).tanh() } else { xd }
    };
    let mut max1 = f64::NEG_INFINITY;
    let mut max2 = f64::NEG_INFINITY;
    for &x in z3_raw {
        let s = cap(x);
        if s >= max1 {
            max2 = max1;
            max1 = s;
        } else if s > max2 {
            max2 = s;
        }
    }
    let z: f64 = z3_raw.iter().map(|&x| (cap(x) - max1).exp()).sum();
    let p1 = 1.0 / z;
    let p2 = if max2.is_finite() { (max2 - max1).exp() / z } else { 0.0 };
    let omega_t = (omega_max as f64) * (1.0 - (p1 - p2)).clamp(0.0, 1.0);
    let out: Vec<f32> = z3_raw
        .iter()
        .zip(z1_raw.iter())
        .map(|(&r3, &r1)| {
            let s3 = cap(r3);
            let s1 = cap(r1);
            (s3 + omega_t * (s3 - s1)) as f32
        })
        .collect();
    (out, omega_t as f32)
}

fn weak_loopcd_mismatch(candidate: &[f32], dry: &[f32], stateful: &[f32]) -> Option<f32> {
    if candidate.len() != dry.len() || candidate.len() != stateful.len() || candidate.is_empty() {
        return Some(f32::INFINITY);
    }
    let mut max_delta = 0.0f32;
    for ((&actual, &dry_ref), &stateful_ref) in candidate.iter().zip(dry).zip(stateful) {
        if !(actual.is_finite() && dry_ref.is_finite() && stateful_ref.is_finite()) {
            return Some(f32::INFINITY);
        }
        max_delta = max_delta.max((actual - dry_ref).abs()).max((actual - stateful_ref).abs());
    }
    (max_delta > 1e-6).then_some(max_delta)
}

/// GPU gate for Adaptive `LoopCD-Logits` (`t-au-loopcd-gpu-gate` / `U19`):
/// 1. Runs `encode_decode_with_omegas` on a 4-lane Layer 3 blocked batch with mixed omegas `[0.0, 0.35, 0.0, 0.5]`
///    (lanes 0 & 1 sharing a sealed 64-token prefix block for `k_attn_grp_dec`, lanes 2 & 3 private for `k_attn_dec_solo`).
/// 2. Snapshots `CS[18..24]` and `KV` coda blocks (`18..24`) around an isolated dry weak pass (`encode_decode_one_pass(..., true)`),
///    proving 0 bytes mutated, AND verifies post-step `CS` + `Window`/`Full` `BlockPool` slots after the full
///    `encode_decode_with_omegas` step match standalone `omega = 0.0` decode bit-for-bit.
/// 3. Proves every `omega == 0.0` lane in the mixed batch keeps raw/unblended logits bit-identical to standalone `omega = 0.0`.
/// 4. Proves every `omega > 0.0` lane matches an independent `f64` CPU `LoopCD` reference over standalone strong (`z3`)
///    and 1-pass weak (`z1`) logits within `< 1e-4` across the full vocabulary, with active contrast (`omega_t > 0`).
/// 5. `INK_LOOPCDGATE_SABOTAGE=1` perturbs one lane's blended/dry-pass logit, printing `LOOPCD GATE DIFFERS` and exiting `1`.
fn loopcdgate(o: &Opts) {
    use blocks::{BlockIndex, BlockKind, BLOCK};
    use engine::{Span, LOOP_END, N_PHYS};

    let omegas = [0.0f32, 0.35, 0.0, 0.5];
    let n_lanes = omegas.len();
    let prefill_len = 72usize; // 1 sealed 64-token block + 8 tokens in active block 1
    let need_toks = prefill_len + 4;
    let ctx = o.ctx.max(256);
    let eng = Engine::load(&o.model, ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(n_lanes);
    let sabotage = std::env::var("INK_LOOPCDGATE_SABOTAGE")
        .ok()
        .filter(|v| !v.is_empty() && v != "0");

    let base = load_gate_tokens(&eng, o);
    // Build 4 token streams: lanes 0 & 1 share block 0 (`[0..64)`) + identical `[64..need_toks)` at different omegas;
    // lanes 2 & 3 have distinct streams so both grouped and solo blocked decode paths run together.
    // Also append a short creative stem at the end so the decoded position has genuine next-token entropy (`omega_t > 0`).
    let tk = Tokenizer::load(&eng.g);
    let stem = tk.encode("In the quiet morning light, the old sailor looked out across the", false);
    let stream = |i: usize| -> Vec<u32> {
        let rot = if i < 2 { 0 } else { 37 * i + 11 };
        let mut v: Vec<u32> = (0..need_toks).map(|q| base[(rot + q) % base.len()]).collect();
        let copy_len = stem.len().min(8);
        let tail_start = prefill_len + 1 - copy_len;
        v[tail_start..prefill_len + 1].copy_from_slice(&stem[..copy_len]);
        v
    };
    let streams: Vec<Vec<u32>> = (0..n_lanes).map(stream).collect();

    // Helper: create `n_lanes` fresh blocked lanes prefilled through `prefill_len` (`[0..64)` then `[64..72)`),
    // sharing block 0 between lanes 0 and 1 when `share_b0` is true.
    let setup_blocked_lanes = |share_b0: bool| -> Vec<Lane> {
        let lanes: Vec<Lane> = (0..n_lanes)
            .map(|i| {
                let l = Lane::new_blocked(&eng, need_toks + 8);
                eng.on(&l).set_tokens(0, &streams[i]);
                l
            })
            .collect();
        {
            let mut pools = eng.block_pool().borrow_mut();
            let mut shared_b0: Option<[(blocks::Slot, u32); 2]> = None;
            for (i, l) in lanes.iter().enumerate() {
                let mut tbl = l.block_table().borrow_mut();
                for b in 0..=prefill_len / BLOCK {
                    let bi = BlockIndex::new(b as u32);
                    if share_b0 && b == 0 && i == 1 {
                        let [(sf, gf), (sw, gw)] = shared_b0.expect("lane 0 block 0 allocated");
                        tbl.fill(&mut pools.full, bi, sf, gf).unwrap();
                        tbl.fill(&mut pools.window, bi, sw, gw).unwrap();
                        continue;
                    }
                    let (sf, _) = match pools.full.lowest_free() {
                        Some(s) => (s, pools.full.take(s).unwrap()),
                        None => { let s = pools.full.grow().unwrap(); (s, pools.full.take(s).unwrap()) }
                    };
                    let gf = pools.full.record(sf).unwrap().generation();
                    let (sw, _) = match pools.window.lowest_free() {
                        Some(s) => (s, pools.window.take(s).unwrap()),
                        None => { let s = pools.window.grow().unwrap(); (s, pools.window.take(s).unwrap()) }
                    };
                    let gw = pools.window.record(sw).unwrap().generation();
                    tbl.fill(&mut pools.full, bi, sf, gf).unwrap();
                    tbl.fill(&mut pools.window, bi, sw, gw).unwrap();
                    if share_b0 && b == 0 && i == 0 {
                        shared_b0 = Some([(sf, gf), (sw, gw)]);
                    }
                }
            }
        }
        // Prefill block 0 (`0..64`) on lanes 0, 2, 3 (lane 1 shares lane 0's block 0 KV and copies CS at 64!),
        // then prefill `[64..72)` across all 4 lanes.
        {
            let pf0_idx: Vec<usize> = if share_b0 { vec![0, 2, 3] } else { vec![0, 1, 2, 3] };
            let sp0: Vec<Span> = pf0_idx.iter().map(|&i| Span { lane: &lanes[i], p0: 0, t: 64 }).collect();
            let cb = eng.gpu.cmd();
            eng.encode_spans(&cb, &sp0, false, false);
            cb.commit();
            cb.waitUntilCompleted();
            if share_b0 {
                let cs0 = eng.on(&lanes[0]).get_cs();
                eng.on(&lanes[1]).set_cs(&cs0);
            }
            let sp1: Vec<Span> = lanes.iter().map(|l| Span { lane: l, p0: 64, t: prefill_len - 64 }).collect();
            let cb = eng.gpu.cmd();
            eng.encode_spans(&cb, &sp1, false, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        lanes
    };

    let release_lanes = |lanes: Vec<Lane>| {
        let mut pools = eng.block_pool().borrow_mut();
        for l in lanes {
            l.block_table().borrow_mut().clear_entries(&mut pools).unwrap();
        }
    };

    let snapshot_lane_kv = |l: &Lane| -> Vec<(BlockKind, u32, Vec<u8>)> {
        let pools = eng.block_pool().borrow();
        let tbl = l.block_table().borrow();
        let mut out = Vec::new();
        for kind in [BlockKind::Full, BlockKind::Window] {
            for b in 0..=prefill_len / BLOCK {
                let e = tbl.get(kind, BlockIndex::new(b as u32)).unwrap().unwrap();
                out.push((kind, b as u32, pools.of(kind).slot_bytes(e.slot()).unwrap().to_vec()));
            }
        }
        out
    };

    let snapshot_coda_kv = |l: &Lane| -> Vec<u8> {
        // Extract exact bytes of coda slots `LOOP_END..N_PHYS` (`18..24`) in active block 1.
        let pools = eng.block_pool().borrow();
        let tbl = l.block_table().borrow();
        let b = (prefill_len / BLOCK) as u32;
        let mut bytes = Vec::new();
        for ip in LOOP_END..N_PHYS {
            let (is_swa, pass_idx) = eng.slot_kind_pass(ip);
            let kind = if is_swa { BlockKind::Window } else { BlockKind::Full };
            let e = tbl.get(kind, BlockIndex::new(b)).unwrap().unwrap();
            let slot_slice = pools.of(kind).slot_bytes(e.slot()).unwrap();
            let n_passes = if is_swa { blocks::WINDOW_PASSES } else { blocks::FULL_PASSES };
            let pass_bytes = slot_slice.len() / n_passes;
            bytes.extend_from_slice(&slot_slice[pass_idx * pass_bytes..(pass_idx + 1) * pass_bytes]);
        }
        bytes
    };

    let mut bad: Vec<String> = Vec::new();
    let coda_cs_range = Engine::cs_slot_byte_range(LOOP_END..N_PHYS);

    // 1. Isolated dry weak pass check (`encode_decode_one_pass(..., true)`):
    //    `CS[18..24]` and `KV[18..24]` must mutate ZERO bytes during the dry weak pass itself!
    let dry_probe_lanes = setup_blocked_lanes(true);
    let pre_coda_cs: Vec<Vec<u8>> = dry_probe_lanes.iter().map(|l| eng.on(l).get_cs()[coda_cs_range.clone()].to_vec()).collect();
    let pre_coda_kv: Vec<Vec<u8>> = dry_probe_lanes.iter().map(snapshot_coda_kv).collect();
    {
        let sp: Vec<Span> = dry_probe_lanes.iter().map(|l| Span { lane: l, p0: prefill_len, t: 1 }).collect();
        let cb = eng.gpu.cmd();
        eng.encode_decode_one_pass(&cb, &sp, true);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let dry_weak_probe_logits: Vec<Vec<f32>> = (0..n_lanes).map(|i| eng.span_logits(i).to_vec()).collect();
    for i in 0..n_lanes {
        let post_cs = &eng.on(&dry_probe_lanes[i]).get_cs()[coda_cs_range.clone()];
        if post_cs != pre_coda_cs[i].as_slice() {
            let diff = post_cs.iter().zip(&pre_coda_cs[i]).filter(|(a, b)| a != b).count();
            bad.push(format!("lane {i}: dry weak pass mutated {diff} bytes of CS[18..24]"));
        }
        let post_kv = snapshot_coda_kv(&dry_probe_lanes[i]);
        if post_kv != pre_coda_kv[i] {
            let diff = post_kv.iter().zip(&pre_coda_kv[i]).filter(|(a, b)| a != b).count();
            bad.push(format!("lane {i}: dry weak pass mutated {diff} bytes of KV[18..24]"));
        }
    }
    release_lanes(dry_probe_lanes);

    // 2. Standalone 1-pass stateful reference (`dry_coda = false`) for `z_1` and standalone solo 3-pass `omega = 0.0` reference for `z_3`, `CS`, and `KV`.
    let ref1_lanes = setup_blocked_lanes(false);
    let z1_ref: Vec<Vec<f32>> = (0..n_lanes)
        .map(|i| {
            let cb = eng.gpu.cmd();
            eng.encode_decode_one_pass(&cb, &[Span { lane: &ref1_lanes[i], p0: prefill_len, t: 1 }], false);
            cb.commit();
            cb.waitUntilCompleted();
            eng.span_logits(0).to_vec()
        })
        .collect();
    release_lanes(ref1_lanes);

    let solo0_lanes = setup_blocked_lanes(false);
    let mut z3_solo: Vec<Vec<f32>> = Vec::with_capacity(n_lanes);
    let mut cs_solo: Vec<Vec<u8>> = Vec::with_capacity(n_lanes);
    let mut kv_solo: Vec<Vec<(BlockKind, u32, Vec<u8>)>> = Vec::with_capacity(n_lanes);
    for i in 0..n_lanes {
        let cb = eng.gpu.cmd();
        eng.encode_decode(&cb, &[Span { lane: &solo0_lanes[i], p0: prefill_len, t: 1 }], true, false);
        cb.commit();
        cb.waitUntilCompleted();
        z3_solo.push(eng.span_logits(0).to_vec());
        cs_solo.push(eng.on(&solo0_lanes[i]).get_cs());
        kv_solo.push(snapshot_lane_kv(&solo0_lanes[i]));
    }
    release_lanes(solo0_lanes);

    // 3. Mixed-omega multi-lane Layer 3 blocked batch (`[0.0, 0.35, 0.0, 0.5]`).
    let cand_lanes = setup_blocked_lanes(true);
    {
        let sp: Vec<Span> = cand_lanes.iter().map(|l| Span { lane: l, p0: prefill_len, t: 1 }).collect();
        let cb = eng.gpu.cmd();
        eng.encode_decode_with_omegas(&cb, &sp, true, false, &omegas);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let mut cand_logits: Vec<Vec<f32>> = (0..n_lanes).map(|i| eng.span_logits(i).to_vec()).collect();
    let cand_weak_logits: Vec<Vec<f32>> = (0..n_lanes).map(|i| eng.span_weak_logits(i).to_vec()).collect();

    if let Some(ref mode) = sabotage {
        println!("INK_LOOPCDGATE_SABOTAGE={mode}: perturbing lane 1 blended logit at token 0 by +0.25");
        cand_logits[1][0] += 0.25;
    }

    let mut max_guided_err = 0.0f32;
    let mut max_contrast_shift = 0.0f32;
    for i in 0..n_lanes {
        // (b) Full-step CS (including 18..24) and KV blocks match standalone non-LoopCD decode bit-for-bit.
        let got_cs = eng.on(&cand_lanes[i]).get_cs();
        if got_cs[coda_cs_range.clone()] != cs_solo[i][coda_cs_range.clone()] {
            bad.push(format!("lane {i}: post-step CS[18..24] differs from standalone omega=0 decode"));
        }
        if got_cs != cs_solo[i] {
            bad.push(format!("lane {i}: post-step full CS differs from standalone omega=0 decode"));
        }
        let got_kv = snapshot_lane_kv(&cand_lanes[i]);
        if got_kv != kv_solo[i] {
            bad.push(format!("lane {i}: post-step KV blocks differ from standalone omega=0 decode"));
        }
        // Verify captured `lg_weak` matches both isolated dry probe and standalone 1-pass reference.
        if let Some(wd) = weak_loopcd_mismatch(&cand_weak_logits[i], &dry_weak_probe_logits[i], &z1_ref[i]) {
            bad.push(format!("lane {i}: captured weak logits fail the dry-probe or standalone comparison (max_diff={wd:.6e})"));
        }

        if omegas[i] == 0.0 {
            // (c) Disabled lane in mixed batch must leave raw logits untouched and report `span_loopcd == false`.
            if eng.span_loopcd(i) || eng.span_effective_sampler_softcap(i) != eng.softcap {
                bad.push(format!("lane {i} (omega=0): span_loopcd={} softcap={}", eng.span_loopcd(i), eng.span_effective_sampler_softcap(i)));
            }
            let same_bits = cand_logits[i]
                .iter()
                .zip(&z3_solo[i])
                .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits());
            if !same_bits {
                let d = cand_logits[i].iter().zip(&z3_solo[i]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                bad.push(format!("lane {i} (omega=0): logits differ from standalone omega=0 step (max_diff={d:.6e})"));
            }
            println!("lane {i} (omega={:.2}): raw logits bit-identical to standalone omega=0, CS/KV bit-identical", omegas[i]);
        } else {
            // (d) Guided lane must match independent `f64` CPU LoopCD formula within `< 1e-4`.
            if !eng.span_loopcd(i) || eng.span_effective_sampler_softcap(i) != 0.0 {
                bad.push(format!("lane {i} (omega={}): span_loopcd={} softcap={}", omegas[i], eng.span_loopcd(i), eng.span_effective_sampler_softcap(i)));
            }
            let (want_blended, omega_t) = cpu_loopcd_reference(&z3_solo[i], &z1_ref[i], eng.softcap, omegas[i]);
            let (worst_tok, worst_diff) = cand_logits[i]
                .iter()
                .zip(&want_blended)
                .enumerate()
                .fold((0usize, 0.0f32), |(bi, bd), (v, (&a, &b))| {
                    let d = if a.is_finite() && b.is_finite() { (a - b).abs() } else { f32::INFINITY };
                    if d > bd { (v, d) } else { (bi, bd) }
                });
            max_guided_err = max_guided_err.max(worst_diff);
            let shift_vs_plain = cand_logits[i]
                .iter()
                .zip(&z3_solo[i])
                .map(|(&g, &r3)| (g - softcap(r3, eng.softcap)).abs())
                .fold(0.0f32, f32::max);
            max_contrast_shift = max_contrast_shift.max(shift_vs_plain);
            println!(
                "lane {i} (omega_max={:.2}, omega_t={omega_t:.4}): max_diff vs CPU ref = {worst_diff:.3e} (tok {worst_tok}), max |guided - softcap(z3)| = {shift_vs_plain:.4}",
                omegas[i]
            );
            if !(worst_diff < 1e-4) {
                bad.push(format!(
                    "lane {i} (omega_max={}): blended logits differ from CPU reference at token {worst_tok} by {worst_diff:.6e} (>= 1e-4)",
                    omegas[i]
                ));
            }
        }
    }
    if max_contrast_shift <= 1e-4 {
        bad.push(format!("LoopCD contrast shift was vacuous (max |guided - softcap(z3)| = {max_contrast_shift:.6e})"));
    }
    release_lanes(cand_lanes);

    if bad.is_empty() {
        println!(
            "loopcd gate exact: {n_lanes} blocked lanes {:?}, dry weak pass mutated 0 CS/KV bytes, omega=0 bit-identical, guided max_err={max_guided_err:.2e} (< 1e-4)",
            omegas
        );
    } else {
        for b in &bad {
            println!("{b}");
        }
        println!("LOOPCD GATE DIFFERS");
        std::process::exit(1);
    }
}

/// Blocked Layer 3 batch-invariance gate (`t-au-blocked-invariance` / `U79` & live `t-audit-engine-dirty-range`):
/// 1. Verifies on a real loaded `Engine` that 64 consecutive blocked GPU decode steps (`pos = 64..128`) write `(1, 1)`
///    block-table words at `pos = 64` and `(0, 0)` words with `dirty_range == None` across all 63 steps `pos = 65..128`.
/// 2. Full-size span (`t = INK_FULL_SPAN`, default `eng.chunk = 4096`) blocked split invariance (`use_blocks = true`):
///    one `4096`-token span + decode steps vs multi-split `[37, 219, 768, 1025, 2047]` (crossing `1024` and `4096`) + decode steps.
/// 3. Multi-lane (`INK_LANES`, default `20 > 17`) blocked prefill + multi-query grouped decode (`k_attn_grp_dec` + `k_attn_dec_solo`)
///    past `1024` positions (`INK_PF = 1120`, `INK_DECODE = 6`, crossing `EXT = 1024` SWA window pruning!) against cold solo
///    blocked execution (`1` lane per `Layer3Pool`, `1` span per step) — every step's logits, greedy tokens, and final `CS` bit-identical!
/// 4. `INK_BLOCKEDINV_SABOTAGE=1` binds one span to another lane's blocked state, printing `BLOCKED NOT INVARIANT` and exiting `1`.
fn blockedinv(o: &Opts) {
    use blocks::{BlockIndex, BlockKind};
    use engine::Span;
    use poolgate::{Layer3Pool, StepPath};

    let sabotage = std::env::var("INK_BLOCKEDINV_SABOTAGE").is_ok_and(|v| !v.is_empty() && v != "0");
    let n_lanes: usize = std::env::var("INK_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if sabotage { 4 } else { 20 });
    let pf_len: usize = std::env::var("INK_PF")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if sabotage { 192 } else { 1120 });
    let ndec: usize = std::env::var("INK_DECODE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if sabotage { 2 } else { 6 });
    let full_span_req: usize = std::env::var("INK_FULL_SPAN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if sabotage { 0 } else { 4096 });

    let needed_ctx = full_span_req
        .max(pf_len + ndec + 16)
        .max(192)
        .div_ceil(64)
        * 64
        + 64;
    let eng = Engine::load(&o.model, o.ctx.max(needed_ctx));
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(n_lanes.max(4));
    let base = load_gate_tokens(&eng, o);
    let fnv = |v: &[f32]| v.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3));
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;

    let mut bad: Vec<String> = Vec::new();

    if !sabotage {
        // Part 1: Live 64-position `Engine::sync_lane_block_table_counts` + `Engine::encode_decode` check (`pos = 64..128`).
        let lane64 = Lane::new_blocked(&eng, 192);
        let toks128: Vec<u32> = (0..130).map(|q| base[q % base.len()]).collect();
        eng.on(&lane64).set_tokens(0, &toks128);
        let alloc_blk = |b: u32| {
            let mut pools = eng.block_pool().borrow_mut();
            let mut tbl = lane64.block_table().borrow_mut();
            let bi = BlockIndex::new(b);
            for kind in [BlockKind::Full, BlockKind::Window] {
                let pool = pools.of_mut(kind);
                let s = match pool.lowest_free() {
                    Some(s) => s,
                    None => pool.grow().unwrap(),
                };
                let _ = pool.take(s).unwrap();
                let gen = pool.record(s).unwrap().generation();
                tbl.fill(pool, bi, s, gen).unwrap();
            }
        };
        alloc_blk(0);
        {
            let cb = eng.gpu.cmd();
            eng.encode_spans(&cb, &[Span { lane: &lane64, p0: 0, t: 64 }], false, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        alloc_blk(1);
        let mut clean_steps = 0usize;
        for pos in 64..128 {
            let (df, dw) = {
                let tbl = lane64.block_table().borrow();
                (tbl.dirty_range(BlockKind::Full), tbl.dirty_range(BlockKind::Window))
            };
            let synced = eng.sync_lane_block_table_counts(&lane64, pos, 1);
            if pos == 64 {
                if df != Some((1, 1)) || dw != Some((1, 1)) || synced != (1, 1) {
                    bad.push(format!("dirty-range gate at pos=64: dirty=({df:?},{dw:?}) synced={synced:?}"));
                }
            } else if df.is_some() || dw.is_some() || synced != (0, 0) {
                bad.push(format!("dirty-range gate at pos={pos}: dirty=({df:?},{dw:?}) synced={synced:?}"));
            } else {
                clean_steps += 1;
            }
            let cb = eng.gpu.cmd();
            eng.encode_decode(&cb, &[Span { lane: &lane64, p0: pos, t: 1 }], false, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        {
            let mut pools = eng.block_pool().borrow_mut();
            lane64.block_table().borrow_mut().clear_entries(&mut pools).unwrap();
        }
        println!("dirty-range live decode: pos=64 wrote (1, 1) table words; {clean_steps}/63 steady-state steps wrote (0, 0) words");

        // Part 2: Full-size span (`t = full_span`, default 4096) blocked split invariance past 1024.
        let full_span = full_span_req.min(eng.chunk);
        if full_span > 0 {
            let full_toks: Vec<u32> = (0..full_span + 4).map(|q| base[(q * 3 + 7) % base.len()]).collect();
            let run_blocked_splits = |chunks: &[usize]| -> (Vec<u64>, Vec<u32>) {
                let pool = Layer3Pool::new(&eng, 1);
                let lid = pool.new_lane(&full_toks[..full_span], full_span + 4);
                let mut p = 0usize;
                for &c in chunks {
                    pool.prefill(&[(lid, p, c)]);
                    p += c;
                }
                assert_eq!(p, full_span);
                let mut hashes = vec![fnv(&pool.logits(0))];
                let mut tok = argmax(&pool.logits(0));
                let mut toks_out = vec![tok];
                pool.set_token(lid, full_span, tok);
                for d in 0..2 {
                    let pos = full_span + d;
                    pool.decode(&[(lid, pos)]);
                    let lg = pool.logits(0);
                    hashes.push(fnv(&lg));
                    tok = argmax(&lg);
                    toks_out.push(tok);
                    pool.set_token(lid, pos + 1, tok);
                }
                pool.end_lane(lid);
                pool.shutdown();
                (hashes, toks_out)
            };
            let split_plan: Vec<usize> = if full_span == 4096 {
                vec![37, 219, 768, 1025, 2047]
            } else {
                let a = full_span / 5;
                let b = full_span / 3;
                vec![a, b, full_span - a - b]
            };
            let (h_one, t_one) = run_blocked_splits(&[full_span]);
            let (h_split, t_split) = run_blocked_splits(&split_plan);
            if h_one != h_split || t_one != t_split {
                bad.push(format!(
                    "full-size blocked span ({full_span}) differs from split {split_plan:?}: one={h_one:x?} vs split={h_split:x?}"
                ));
            } else {
                println!(
                    "full-size blocked span ({full_span} tokens vs {split_plan:?} + 2 decode steps): 100% bit-identical ({:016x})",
                    h_one[0]
                );
            }
        }
    }

    // Part 3: Multi-lane (`n_lanes`, default 20 > 17) blocked prefill + shared-KV grouped decode past 1024 positions.
    // Lanes `0..(n_lanes * 3 / 4)` share a sealed prefix (`shared_head`, 1024 when `pf_len > 1024`) to exercise
    // multiple `G_MAX = 8` groups + remainder in `k_attn_grp_dec`; remaining lanes run cold distinct streams (`k_attn_dec_solo`).
    let shared_head = if pf_len > 1024 { 1024 } else { (pf_len / 2 / 64) * 64 };
    let shared_count = (n_lanes * 3 / 4).max(2).min(n_lanes);
    let prompts: Vec<Vec<u32>> = (0..n_lanes)
        .map(|i| {
            if i < shared_count && shared_head > 0 {
                let mut v: Vec<u32> = (0..shared_head).map(|q| base[q % base.len()]).collect();
                v.extend((0..pf_len - shared_head).map(|q| base[(97 * (i + 1) + q) % base.len()]));
                v
            } else {
                (0..pf_len).map(|q| base[(131 * (i + 1) + q) % base.len()]).collect()
            }
        })
        .collect();

    // Cold solo blocked reference: each lane runs completely alone on a fresh `Layer3Pool` (`1` span per step).
    let mut solo_hashes: Vec<Vec<u64>> = Vec::with_capacity(n_lanes);
    let mut solo_tokens: Vec<Vec<u32>> = Vec::with_capacity(n_lanes);
    for i in 0..n_lanes {
        let pool = Layer3Pool::new(&eng, 1);
        let lid = pool.new_lane(&prompts[i], pf_len + ndec + 4);
        let mut p = 0usize;
        while p < pf_len {
            let c = eng.chunk.min(pf_len - p);
            pool.prefill(&[(lid, p, c)]);
            p += c;
        }
        let lg0 = pool.logits(0);
        let mut hs = vec![fnv(&lg0)];
        let mut tok = argmax(&lg0);
        let mut ts = vec![tok];
        pool.set_token(lid, pf_len, tok);
        for d in 0..ndec {
            let pos = pf_len + d;
            pool.decode(&[(lid, pos)]);
            let lg = pool.logits(0);
            hs.push(fnv(&lg));
            tok = argmax(&lg);
            ts.push(tok);
            pool.set_token(lid, pos + 1, tok);
        }
        pool.end_lane(lid);
        pool.shutdown();
        solo_hashes.push(hs);
        solo_tokens.push(ts);
    }

    // Batched blocked candidate (`Layer3Pool::with_persistent_starts` so shared-prefix lanes match `EVERY@1024` / `PROMPT`
    // start points and share `Window` + `Full` blocks across `> 17` concurrent lanes!).
    let pool = Layer3Pool::with_persistent_starts(&eng, n_lanes);
    if shared_head > 0 {
        // Seed the shared prefix into `PrefixTree` (`0..shared_head`, plus 1 tail token so prompt-end / EVERY@1024 saves).
        let mut seed_prompt = prompts[0][..shared_head].to_vec();
        seed_prompt.push(base[0]);
        let (slid, _) = pool.new_lane_matched(&seed_prompt, shared_head + 8);
        let mut sp = 0usize;
        for &ch in &[320usize, 704] {
            if sp < shared_head {
                let step = ch.min(shared_head - sp);
                pool.prefill(&[(slid, sp, step)]);
                sp += step;
            }
        }
        while sp < seed_prompt.len() {
            let step = (seed_prompt.len() - sp).min(64);
            pool.prefill(&[(slid, sp, step)]);
            sp += step;
        }
        pool.end_lane(slid);
    }

    let mut lids = Vec::with_capacity(n_lanes);
    let mut pos_cur = Vec::with_capacity(n_lanes);
    for i in 0..n_lanes {
        let (lid, matched) = pool.new_lane_matched(&prompts[i], pf_len + ndec + 4);
        lids.push(lid);
        pos_cur.push(matched);
    }

    // Batched multi-span blocked prefill in staggered chunks (crossing block and 1024 boundaries) until all lanes reach `pf_len`.
    let chunk_sizes = [193usize, 320, 513, 256];
    let mut c_idx = 0usize;
    let mut batch_hashes: Vec<Vec<u64>> = vec![Vec::new(); n_lanes];
    let mut batch_tokens: Vec<Vec<u32>> = vec![Vec::new(); n_lanes];
    let mut sabotaged = false;
    while pos_cur.iter().any(|&p| p < pf_len) {
        let step_cap = chunk_sizes[c_idx % chunk_sizes.len()];
        c_idx += 1;
        let mut spans = Vec::new();
        let mut who = Vec::new();
        let mut total_t = 0usize;
        for i in 0..n_lanes {
            if pos_cur[i] < pf_len {
                let rem = pf_len - pos_cur[i];
                let t = rem.min(step_cap).min(eng.chunk.saturating_sub(total_t));
                if t == 0 {
                    break;
                }
                spans.push((lids[i], pos_cur[i], t));
                who.push(i);
                total_t += t;
            }
        }
        if sabotage && !sabotaged && spans.len() >= 2 {
            // Planted bug: bind lane 0's prefill span to another lane's handle at the same offset if valid, or corrupt token.
            println!("INK_BLOCKEDINV_SABOTAGE: corrupting token at pos {} on blocked lane {}", spans[0].1, who[0]);
            pool.set_token(spans[0].0, spans[0].1, prompts[who[0]][spans[0].1].wrapping_add(1) % 1000);
            sabotaged = true;
        }
        pool.prefill(&spans);
        for (row, &i) in who.iter().enumerate() {
            pos_cur[i] += spans[row].2;
            if pos_cur[i] == pf_len {
                let lg = pool.logits(row);
                batch_hashes[i].push(fnv(&lg));
                let tok = argmax(&lg);
                batch_tokens[i].push(tok);
                pool.set_token(lids[i], pf_len, tok);
            }
        }
    }

    // Batched `n_lanes`-wide blocked decode steps past `1024` (`pf_len..pf_len + ndec`).
    for d in 0..ndec {
        let pos = pf_len + d;
        if sabotage && !sabotaged {
            println!("INK_BLOCKEDINV_SABOTAGE: corrupting decode token at pos {pos} on blocked lane 0");
            pool.set_token(lids[0], pos, batch_tokens[0][d].wrapping_add(1) % 1000);
            sabotaged = true;
        }
        let dec_spans: Vec<(usize, usize)> = lids.iter().map(|&lid| (lid, pos)).collect();
        pool.decode(&dec_spans);
        for i in 0..n_lanes {
            let lg = pool.logits(i);
            batch_hashes[i].push(fnv(&lg));
            let tok = argmax(&lg);
            batch_tokens[i].push(tok);
            pool.set_token(lids[i], pos + 1, tok);
        }
    }
    let st = pool.stats();
    pool.shutdown();

    for i in 0..n_lanes {
        if batch_hashes[i] != solo_hashes[i] || batch_tokens[i] != solo_tokens[i] {
            let first_step = batch_hashes[i].iter().zip(&solo_hashes[i]).position(|(a, b)| a != b);
            bad.push(format!("blocked lane {i}: differs from solo blocked reference at step {first_step:?}"));
        }
    }

    if bad.is_empty() {
        println!(
            "blockedinv exact: {n_lanes} blocked lanes at pf={pf_len} + dec={ndec} (matches={}, window_pruned={}, full_span={}) 100% bit-identical to solo blocked execution",
            st.start_matches,
            st.window_pruned,
            full_span_req.min(eng.chunk)
        );
    } else {
        for b in &bad {
            println!("{b}");
        }
        println!("BLOCKED NOT INVARIANT");
        std::process::exit(1);
    }
}

/// dev, the max-context eval: one lane prefills the --file text (real code) to the engine's limit, timing each stretch of
/// prefill; at each checkpoint (INK_CKPTS) one stream decodes on the decode path, then as many lanes as fit a budget of
/// INK_LANE_GIB GiB (copies of the prefilled cache, each decoding its own tokens) decode together. Timing only.
fn maxctx(o: &Opts) {
    use engine::Span;
    let eng = Engine::load(&o.model, o.ctx);
    eng.ensure_batch(eng.chunk);
    eng.ensure_dbatch(64);
    let tk = Tokenizer::load(&eng.g);
    let mut toks = tk.encode(&o.prompt, false);
    let steps = 8usize;
    let ckpts: Vec<usize> = std::env::var("INK_CKPTS").unwrap_or("16384,65536,131072,262000".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let last = *ckpts.iter().max().unwrap();
    assert!(last + steps < eng.max_ctx && toks.len() > last + steps, "need -c > {} and more text ({} tokens)", last + steps, toks.len());
    toks.truncate(last + steps + 1);
    // lane memory: what Metal's working set leaves after everything allocated so far, less the tuning's margin
    let budget = std::env::var("INK_LANE_GIB").ok().and_then(|v| v.parse::<f64>().ok()).map(|g| g * (1u64 << 30) as f64).unwrap_or_else(|| {
        let (alloc, ws) = eng.gpu_mem();
        let (chip, mem) = tuning::this_machine();
        let margin = tuning::select(&chip, mem).lane_margin_gib * (1u64 << 30) as f64;
        (ws as f64 - alloc as f64 - margin - 9.0 * (1u64 << 30) as f64).max(1.0)      // (9 GiB: the source lane at the longest context)
    });
    println!("lane budget {:.1} GiB", budget / (1u64 << 30) as f64);
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    let src = Lane::new(&eng);
    let rs = eng.on(&src);
    rs.reserve(last + steps + 1, false);
    rs.set_tokens(0, &toks);
    println!("{} tokens of text; KV per lane at {}: {:.1} GiB", toks.len(), last, gib(rs.kv_exact_bytes(last + steps + 1)));
    println!("{:>7} {:>13} {:>11} {:>14} {:>6} {:>11} {:>9} {:>8}", "ctx", "prefill t/s", "TTFT s", "1 stream t/s", "lanes", "lanes t/s", "gain", "GPU GiB");
    let (mut p, mut tpf) = (0usize, 0.0f64);
    for &c in &ckpts {
        let t0 = std::time::Instant::now();
        let p_start = p;
        while p < c { let t = (c - p).min(eng.chunk); let cb = eng.gpu.cmd(); eng.encode_spans(&cb, &[Span { lane: &src, p0: p, t }], false, false); cb.commit(); cb.waitUntilCompleted(); p += t; }
        let dt = t0.elapsed().as_secs_f64(); tpf += dt;
        let pf_rate = (c - p_start) as f64 / dt;
        // one stream, decode path (timing only: it advances src's state; the prefill above already finished)
        let tok = |q: usize| { let cb = eng.gpu.cmd(); rs.encode_step(&cb, c + q, true, false); cb.commit(); cb.waitUntilCompleted(); };
        tok(0);
        let t1 = std::time::Instant::now(); for q in 0..steps { tok(q); }
        let one = steps as f64 / t1.elapsed().as_secs_f64();
        // lanes: as many as fit the budget at this context
        let per = rs.kv_exact_bytes(c + steps + 1) as f64;
        let n = ((budget / per) as usize).clamp(1, 64);
        let lanes: Vec<Lane> = (0..n).map(|i| { let l = Lane::new(&eng); let r = eng.on(&l); r.reserve(c + steps + 1, false);
            let mut t: Vec<u32> = toks[..=c + steps].to_vec(); for q in 0..=steps { t[c + q] = toks[(c + 7919 * (i + 1) + q) % c]; }
            r.set_tokens(0, &t); eng.copy_lane_prefix(&src, &l, c); l }).collect();
        let step = |q: usize| { let sp: Vec<Span> = lanes.iter().map(|l| Span { lane: l, p0: c + q, t: 1 }).collect();
                                let cb = eng.gpu.cmd(); eng.encode_decode(&cb, &sp, true, false); cb.commit(); cb.waitUntilCompleted(); };
        step(0);
        let t2 = std::time::Instant::now(); for q in 0..steps { step(q); }
        let lt = (n * steps) as f64 / t2.elapsed().as_secs_f64();
        println!("{c:>7} {pf_rate:>13.0} {tpf:>11.1} {one:>14.1} {n:>6} {lt:>11.1} {:>8.2}x {:>8.1}", lt / one, gib(eng.gpu_mem().0));
        drop(lanes);
    }
}

fn stats(name: &str, v: &[f32]) {
    let n = v.len();
    let s: f64 = v.iter().map(|&x| x as f64).sum();
    println!("{name:>12} [{:9.4} {:9.4} {:9.4} ... {:9.4} {:9.4} {:9.4}] sum = {s:.6}", v[0], v[1], v[2], v[n - 3], v[n - 2], v[n - 1]);
}

fn debug(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    let tk = Tokenizer::load(&eng.g);
    let toks = tk.encode(&o.prompt, true);
    check_prompt_fits_ctx(toks.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
    eng.reset();
    eng.set_tokens(0, &toks);
    for pos in 0..toks.len() {
        eng.stop.set(if pos + 1 == toks.len() { o.stop } else { usize::MAX });
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, o.stop == usize::MAX, false);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let ar = eng.ar;
    stats("x0", &eng.arena(0, engine::E));
    stats("inp0", &eng.arena(ar.inp[0], engine::E));
    stats("q", &eng.arena(ar.q, 3072));
    stats("k", &eng.arena(ar.k, 1024));
    stats("v", &eng.arena(ar.v, 1024));
    stats("rel", &eng.arena(ar.rel, 384));
    stats("sink", &eng.arena(ar.sink, 24));
    stats("attn_g", &eng.arena(ar.attn_g, 3072));
    stats("ffn_inp", &eng.arena(engine::Arena::vec(eng.last_mid.get()), engine::E));
    stats("fin", &eng.arena(ar.fin, engine::E));
    stats("rlog", &eng.arena(ar.rlog, 128));
    stats("sel", &eng.arena(ar.sel, 16));
    stats("act", &eng.arena(ar.act, 8 * 1152));
    stats("l_out", &eng.arena(engine::Arena::vec(eng.last_out.get()), engine::E));
}

fn prof(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    let vary = std::env::var("INK_VARY").is_ok();
    let toks: Vec<u32> = (0..o.ctx.min(4096)).map(|i| if vary { ((i as u64 * 2654435761u64) % 150000) as u32 + 1000 } else { 1000 }).collect();
    eng.set_tokens(0, &toks);
    let pos = o.n.min(o.ctx - 64);   // profile at this context position
    let time = |skip: u32, reps: usize| -> f64 {
        eng.skip.set(skip);
        let mut ts = Vec::new();
        let cbs: Vec<_> = (0..reps).map(|r| { let cb = eng.gpu.cmd(); eng.encode_step(&cb, if vary { pos + r } else { pos }, true, false); cb.commit(); cb }).collect();
        for cb in &cbs { cb.waitUntilCompleted(); ts.push(cb.GPUEndTime() - cb.GPUStartTime()); }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[reps / 2] * 1e3
    };
    time(0, 5);
    let base = time(0, 15);
    println!("full step at pos {pos}: {base:.3} ms GPU ({:.1} tok/s)", 1e3 / base);
    let mut tot = 0.0;
    for (c, name) in engine::CLS_NAMES.iter().enumerate() {
        let t = time(1 << c, 9);
        tot += base - t;
        println!("  {name:>11}: {:7.3} ms ({:5.1}%)", base - t, (base - t) / base * 100.0);
    }
    let rest = time(u32::MAX, 9);
    println!("  everything skipped: {rest:.3} ms | sum of classes {tot:.3} ms");
}

fn prof_pf(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    eng.set_tokens(0, &(0..8192).map(|i| 1000 + (i * 7) % 5000).collect::<Vec<u32>>());
    if o.prompt.len() > 500 {
        let tk = Tokenizer::load(&eng.g);
        let pt = tk.encode(&first_turn(o, &o.prompt), true);
        println!("using prompt ({} tokens)", pt.len());
        eng.set_tokens(0, &pt);
    }
    eng.tail.set(false);      // profile full layers (the MoE classes re-use the routing of the last layer)
    let t = eng.chunk;
    let p0 = o.n.min(o.ctx - t);
    // fill the cache up to p0 so attention sees a realistic context
    let mut q = 0;
    while q < p0 { let n = (p0 - q).min(t); let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, q, n, false, false); cb.commit(); cb.waitUntilCompleted(); q += n; }
    let once = |skip: u32| -> f64 {
        eng.skip.set(skip);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, p0, t, true, true);
        cb.commit(); cb.waitUntilCompleted();
        (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
    };
    once(0); once(0);
    let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let base = med((0..5).map(|_| once(0)).collect());
    println!("prefill chunk of {t} tokens at p0={p0}: {base:.1} ms ({:.0} tok/s)", t as f64 / base * 1e3);
    { let tl = eng.moe_tiles(); let s: u32 = tl.iter().flatten().sum();
      println!("  last-layer MoE tiles: {} of 8 simdgroups, {} of 4; assignments {s}, 16-token slots used {}", tl[0].len(), tl[1].len(),
               tl.iter().flatten().map(|&c| c.div_ceil(16)).sum::<u32>() * 16); }
    // isolation: run only one class (buffers keep realistic contents, incl. MoE routing, from the full pass)
    let mut tot = 0.0;
    for (c, name) in engine::CLS_NAMES.iter().enumerate() {
        once(0);
        let d = med((0..3).map(|_| once(!(1u32 << c) & !(1 << 21))).collect());
        let empty = med((0..3).map(|_| once(u32::MAX)).collect());
        let d = d - empty;
        tot += d;
        if d > 0.3 { println!("  {name:>11}: {d:8.2} ms ({:5.1}%)", d / base * 100.0); }
    }
    println!("  sum of isolated classes: {tot:.1} ms");
}

/// dev: greedy-generate and log tokens + per-layer expert routing of every step (for speculation analysis)
fn routes(o: &Opts) -> Result<(), String> {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    let tk = Tokenizer::load(&eng.g);
    let text = if o.raw { o.prompt.clone() } else { first_turn(o, &o.prompt) };
    let prompt = tk.encode(&text, true);
    check_prompt_fits_ctx(prompt.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
    let mut sess = Session::new(&eng, &tk);
    let mut tok = sess.prefill(&prompt)?;
    let mut out = String::new();
    for t in &prompt { out += &format!("P {t}\n"); }
    for _ in 0..o.n {
        if sess.n_past + 1 >= eng.max_ctx { break; }
        let pos = sess.n_past;
        eng.set_tokens(pos, &[tok]);
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, true, true);
        cb.commit();
        metal::wait_completed(&cb, "routes decode", pos)?;
        sess.n_past += 1;
        let r = eng.routes();
        out += &format!("G {tok} {}\n", r.iter().map(|x| x.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(",")).collect::<Vec<_>>().join(";"));
        tok = eng.token_at(pos + 1);
        if tok == tk.eos { break; }
    }
    std::fs::write(&o.reference, out).unwrap();
    Ok(())
}

/// dev: cost of a batched pass of T tokens vs one decode step
fn vcost(o: &Opts) {
    let eng = Engine::load(&o.model, o.ctx);
    let lane = Lane::new(&eng);
    let eng = eng.on(&lane);
    eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
    eng.reset();
    eng.set_tokens(0, &(0..4096).map(|i| 1000 + (i * 7) % 5000).collect::<Vec<u32>>());
    let pos = 300usize;
    let time = |f: &dyn Fn(&metal::CmdBuf)| -> f64 {
        let mut ts = Vec::new();
        for _ in 0..12 { let cb = eng.gpu.cmd(); f(&cb); cb.commit(); cb.waitUntilCompleted(); ts.push(cb.GPUEndTime() - cb.GPUStartTime()); }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap()); ts[6] * 1e3
    };
    let step = time(&|cb| eng.encode_step(cb, pos, true, true));
    println!("decode step: {step:.2} ms");
    for t in [1usize, 2, 4, 8, 16] {
        let pf = time(&|cb| eng.encode_prefill(cb, pos, t, true, true));
        println!("batched pass T={t:2}: {pf:6.2} ms = {:4.2} steps", pf / step);
    }
}

fn enter_inky_mode() -> Option<memd::Lease> {
    let (chip, mem) = tuning::this_machine();
    let t = tuning::select(&chip, mem);
    let sock = std::env::var("INKY_MEMD_SOCKET").unwrap_or(memd::SOCKET.into());     // tests point it elsewhere
    let lease = memd::enter(&sock);
    match &lease {
        Some(lease) => eprintln!("[inky] inky mode: tuning {} (default_lanes {}), GPU wired limit {} MB while connected", t.name, t.default_lanes, lease.initial_limit),
        None => eprintln!("[inky] not in inky mode: the GPU keeps the macOS default limit (sudo inky tune-memory installs inky mode; tuning {}, default_lanes {})", t.name, t.default_lanes),
    }
    lease
}

/// Current physical footprint charged to this process by the macOS kernel, in bytes.
fn process_footprint_bytes() -> Option<u64> {
    unsafe {
        let mut usage: libc::rusage_info_v4 = std::mem::zeroed();
        let result = libc::proc_pid_rusage(
            libc::getpid(), libc::RUSAGE_INFO_V4, &mut usage as *mut _ as *mut libc::rusage_info_t,
        );
        (result == 0).then_some(usage.ri_phys_footprint)
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().collect();
    match raw.get(1).map(|s| s.as_str()) {
        Some("memd") => { memd::daemon(&raw[2..]); return; }
        Some("inblock-trace") => {
            let path = raw.get(2).map(|s| s.as_str()).unwrap_or("specs/InBlockTraceData.tla");
            if let Err(e) = blocks::write_inblock_trace(path) {
                eprintln!("inblock-trace: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("tune-memory") => { memd::install(&raw[2..]); return; }
        Some("serve") => {
            let args = &raw[2..];
            if args.len() == 1 && matches!(args[0].as_str(), "-h" | "--help") {
                println!("{SERVE_USAGE}");
                return;
            }
            let (chip, memory) = tuning::this_machine();
            let options = parse_serve(args, tuning::select(&chip, memory).default_lanes, |name| std::env::var(name)).unwrap_or_else(|e| cli_error(e));
            let _lease = enter_inky_mode();
            serve::serve(options);
            return;
        }
        Some("mode") => {
            // Hold inky mode for the requested interval, then leave by exiting.
            let _lease = enter_inky_mode();
            let secs: u64 = raw.get(2).and_then(|v| v.parse().ok()).unwrap_or(0);
            std::thread::sleep(std::time::Duration::from_secs(secs));
            return;
        }
        _ => {}
    }
    let (cmd, o) = parse();
    let _lease = if matches!(cmd.as_str(), "help" | "-h" | "--help") { None } else { enter_inky_mode() };
    match cmd.as_str() {
        "check" => report_cli_result(&cmd, check(&o)),
        "rows" => rows(&o),
        "rollback" => rollback(&o),
        "verify" => verify(&o),
        "binv" => report_cli_result(&cmd, binv(&o)),
        "crosspath" => crosspath(&o),
        "lanes" => lanes(&o),
        "spans" => spans(&o),
        "xcopy" => xcopy(&o),
        "poolgate" => poolgate(&o),
        "loopcdgate" => loopcdgate(&o),
        "blockedinv" => blockedinv(&o),
        "temp1" => temp1(&o),
        "lanebench" => lanebench(&o),
        "dbatch" => dbatch(&o),
        "sharedkv" => sharedkv(&o),
        "prefixbench" => prefixbench(&o),
        "maxctx" => maxctx(&o),
        "debug" => debug(&o),
        "prof" => prof(&o),
        "routes" => report_cli_result(&cmd, routes(&o)),
        "vcost" => vcost(&o),
        "prof_pf" => prof_pf(&o),
        "kmoe" => {
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);

            println!("k_moe_gu   (prod): {:6.1} GB/s", eng.kbench_moe("k_moe_gu", (engine::FF / 4, 8), 128));
            println!("k_moe_down (prod): {:6.1} GB/s", eng.kbench_moe("k_moe_down", (engine::E / 4, 1), 64));
        }
        "kgemm" => {
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            let t: usize = std::env::var("INK_T").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
            for (n, bm, bn) in [("kgs_64_128_8", 64, 128), ("kgs_64_64_4", 64, 64), ("kgs_64_32_4", 64, 32), ("kgs_64_96_6", 64, 96), ("kgs_64_112_7", 64, 112), ("kgs_64_192_12", 64, 192), ("kgs_64_256_16", 64, 256), ("kgb_64_128_8", 64, 128), ("kgb_64_64_4", 64, 64), ("kg_iso0_3", 64, 64), ("kg_iso2_3", 64, 64)] {
                println!("{n:>16}: {:5.1} TFLOPs", eng.kbench_gemm(n, bm, bn, t));
            }
        }
        "kmoepf" => {
            // MoE GEMM kernels alone, on the real routing of the prompt's last layer
            let eng = Engine::load(&o.model, o.ctx);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            eng.reserve(o.ctx, false); eng.ensure_batch(eng.chunk);      // dev tool: everything pre-allocated
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            check_prompt_fits_ctx(pt.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
            let t = pt.len().min(eng.chunk);
            eng.reset();
            eng.set_tokens(0, &pt);
            let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, 0, t, false, false); cb.commit(); cb.waitUntilCompleted();
            let tl = eng.moe_tiles();
            let (gu, dn) = eng.kbench_moe_pf(t);
            println!("T={t}: tiles {} x8sg {} x4sg | gate/up {gu:.1} TFLOPs, down {dn:.1} TFLOPs (real flops)", tl[0].len(), tl[1].len());
        }
        "ksg" => {
            // per-simdgroup tile ops: cost of a tile vs threadgroup size and number of simdgroups that have tokens
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            for (name, nsg, nact) in [("kgsg_8", 8, 8), ("kgsgf0_8", 8, 8), ("kgsgf0_8", 8, 6), ("kgsgf0_8", 8, 4), ("kgsgf0_8", 8, 2), ("kgsg_4", 4, 4),
                                      ("kgsgf0_4", 4, 4), ("kgsgf0_4", 4, 2), ("kgsgf0_4", 4, 1), ("kgsg_16", 16, 16), ("kgsgf0_16", 16, 16)] {
                let tf = eng.kbench_sgn(name, nsg, nact, 3840);
                println!("{name:>10} {nsg:2} simdgroups, {nact:2} with tokens: {tf:5.1} TFLOPs -> tile cost {:.2}", (16 * nact) as f64 / tf);
            }
        }
        "qclip" => {
            // dev: are any key dimensions unused by all three query heads of their group?
            let eng = Engine::load(&o.model, 64);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            let pb = unsafe { std::slice::from_raw_parts(eng.pb.contents().as_ptr() as *const f32, eng.pb.length() / 4) };
            for (i, l) in eng.layers.iter().enumerate() {
                let w = &pb[l.q_clip as usize / 4..l.q_clip as usize / 4 + 24 * 128];
                let mut eff: Vec<f32> = (0..8 * 128).map(|c| { let (g, d) = (c / 128, c % 128); (0..3).map(|h| w[(g * 3 + h) * 128 + d].abs()).fold(0f32, f32::max) }).collect();
                eff.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let zeros = eff.iter().filter(|&&x| x == 0.0).count();
                println!("layer {i:2} swa {}: max|qclip| over the 3 heads, per key dim: min {:.4} p5 {:.4} p25 {:.4} median {:.4} max {:.4}  exact zeros {zeros}",
                    l.swa as u8, eff[0], eff[51], eff[256], eff[512], eff[1023]);
            }
        }
        "lazytest" => {
            // dev: does a no-copy Metal buffer over anonymous memory commit pages only when they are touched, and what does
            // the GPU's first touch of a page cost?
            let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void k_w(device uint * B [[buffer(0)]], constant uint2 & a [[buffer(1)]], uint gid [[thread_position_in_grid]]) { B[a.x + gid * a.y] += gid + 1; }\n";
            let gpu = metal::Gpu::new(src, &["k_w"]);
            let foot = || process_footprint_bytes().expect("read process physical footprint") as f64 / 1048576.0;
            let len = 1usize << 30;
            let f0 = foot();
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_ANON | libc::MAP_PRIVATE, -1, 0) } as *mut u8;
            let buf = gpu.buffer_nocopy(ptr, len);
            println!("footprint: start {f0:.1} MiB, after wrapping 1 GiB of untouched memory {:.1} MiB", foot());
            let run = |first: u32, stride: u32, n: usize| {
                let cb = gpu.cmd(); let e = metal::Enc::new(&cb);
                e.pso(&gpu.pso("k_w")); e.buf(&buf, 0, 0); e.bytes(&[first, stride], 1);
                e.dispatch((n.div_ceil(64), 1, 1), 64); e.end();
                let t = std::time::Instant::now(); cb.commit(); cb.waitUntilCompleted(); t.elapsed().as_secs_f64() * 1e3
            };
            let mib = |m: usize| (m << 18) as u32;      // MiB -> uint index
            let rd = |m: usize| unsafe { *(ptr.add(m << 20) as *const u32) };
            let mut last = foot();
            // pages of the mapping that physically exist (the footprint ledger does not see pages only the GPU has touched)
            let resident = || { let mut v = vec![0i8; len / 16384]; unsafe { libc::mincore(ptr as *mut libc::c_void, len, v.as_mut_ptr()); } v.iter().filter(|&&x| x & 1 != 0).count() as f64 * 16384.0 / 1048576.0 };
            let mut lastr = resident();
            let mut step = |what: &str, ms: f64| { let (f, r) = (foot(), resident()); println!("{what}: {ms:.2} ms, footprint {:+.2} MiB (now {f:.1}), resident in the mapping {:+.2} MiB (now {r:.1})", f - last, r - lastr); last = f; lastr = r; };
            step("GPU writes one word at 0", run(0, 1, 1));
            step("GPU writes one word at 0 again", run(0, 1, 1));
            step("GPU writes one word at 1 MiB", run(mib(1), 1, 1));
            step("GPU writes one word at 100 MiB", run(mib(100), 1, 1));
            step("GPU writes one word at 101 MiB", run(mib(101), 1, 1));
            step("GPU writes one word at 132 MiB", run(mib(132), 1, 1));
            step("GPU writes one word at 164 MiB", run(mib(164), 1, 1));
            step("GPU writes 4 MiB dense at 200 MiB", run(mib(200), 1, 1 << 20));
            step("GPU writes 4 MiB dense at 200 MiB again", run(mib(200), 1, 1 << 20));
            step("GPU writes 32 MiB dense at 300 MiB", run(mib(300), 1, 8 << 20));
            step("GPU writes 32 MiB dense at 300 MiB again", run(mib(300), 1, 8 << 20));
            for i in 0..2048usize { unsafe { *ptr.add((400 << 20) + i * 16384) = 1; } }
            step("CPU touches 32 MiB at 400 MiB", 0.0);
            step("GPU writes 32 MiB dense at 400 MiB (CPU-touched)", run(mib(400), 1, 8 << 20));
            step("GPU writes 32 MiB dense at 400 MiB again", run(mib(400), 1, 8 << 20));
            println!("readback: {} {} {} {} (expect 2 1 1 2)", rd(0), rd(1), rd(100), rd(200));
            // one fresh page per dispatch
            let mut tf = 0.0; let mut tw = 0.0;
            for i in 0..200u32 { tf += run(mib(600) + i * 4096, 1, 64); }
            for i in 0..200u32 { tw += run(mib(600) + i * 4096, 1, 64); }
            step("200 dispatches, one fresh page each", tf);
            println!("per dispatch: fresh page {:.1} us, touched page {:.1} us", tf * 5.0, tw * 5.0);
        }
        "kvprune" => {
            // dev: what does evicting far keys of the full-attention layers cost? Prefill the prompt, decode -n tokens with the
            // whole cache (the reference), then decode the same tokens again with only a share of the far keys kept.
            let eng = Engine::load(&o.model, o.ctx);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            let (n, d) = (pt.len(), if o.n_set { o.n } else { 48 });
            if n <= engine::EXT {
                cli_error(format!("kvprune requires a prompt longer than {} tokens (got {n})", engine::EXT));
            }
            check_prompt_fits_ctx(n + d, eng.max_ctx).unwrap_or_else(|e| cli_error(e));
            eng.reset(); eng.reserve(n + d + 1, false); eng.ensure_batch(n.min(eng.chunk)); eng.qcap_alloc(n);
            eng.set_tokens(0, &pt);
            let mut p0 = 0;
            while p0 < n { let t = (n - p0).min(eng.chunk); let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, p0, t, p0 + t == n, false); cb.commit(); cb.waitUntilCompleted(); p0 += t; }
            let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
            let first = am(&eng.logits()) as u32;
            let snap = eng.snapshot();
            let run = |toks: Option<&[u32]>| -> (Vec<u32>, Vec<Vec<f32>>, f64) {
                eng.restore(&snap);
                let (mut tok, mut ts, mut lgs) = (first, Vec::new(), Vec::new());
                let t0 = std::time::Instant::now();
                for i in 0..d {
                    if let Some(tt) = toks { tok = tt[i]; }
                    eng.set_tokens(n + i, &[tok]);
                    let cb = eng.gpu.cmd(); eng.encode_step(&cb, n + i, true, false); cb.commit(); cb.waitUntilCompleted();
                    let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap(x, eng.softcap)).collect();
                    ts.push(tok); tok = am(&lg) as u32; lgs.push(lg);
                }
                (ts, lgs, t0.elapsed().as_secs_f64() * 1e3 / d as f64)
            };
            let (toks, refl, ms0) = run(None);
            let (_, again, _) = run(Some(&toks));
            let same = refl.iter().zip(&again).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
            println!("{n} prompt tokens, {d} decode steps, {ms0:.2} ms per step with the whole cache (rerun bit-identical: {same})");
            let full_mib = |keys: usize| (8 * keys * 4096) as f64 / 1048576.0;
            let lse = |v: &[f32]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln() };
            let nq: usize = std::env::var("INK_NQ").ok().and_then(|v| v.parse().ok()).unwrap_or(192);
            let keeps = std::env::var("INK_KEEP").unwrap_or("1.0,0.5,0.25,0.125,0.0".into());
            for mode in std::env::var("INK_MODE").unwrap_or("sum".into()).split(',') { for sum in [false, true] { for keep in keeps.split(',').map(|k| k.parse::<f64>().unwrap()) {
                let t0 = std::time::Instant::now();
                let (far, kept) = eng.prune_build(n, keep, sum, nq, mode);
                let tb = t0.elapsed().as_secs_f64();
                let (_, lg, ms) = run(Some(&toks));
                let (mut agree, mut worst, mut mean) = (0, 0f32, 0f32);
                for (a, r) in lg.iter().zip(&refl) {
                    let (l1, l2) = (lse(a), lse(r));
                    let kl: f32 = r.iter().zip(a).map(|(b, a)| { let pb = (b - l2).exp(); pb * ((b - l2) - (a - l1)) }).sum();
                    if am(a) == am(r) { agree += 1; }
                    worst = worst.max(kl); mean += kl / d as f32;
                }
                println!("score {mode:3} keep {:5.1}% of {far} far keys{}: full-attention cache {:.0} -> {:.0} MiB | argmax {agree}/{d}  KL mean {mean:.5} worst {worst:.5} | {ms:.2} ms/step (selection {tb:.1} s)",
                         keep * 100.0, if sum { " + summary key" } else { "              " }, full_mib(n + d), full_mib(kept + sum as usize + 1024 + d));
            } } }
            *eng.lane.prune.borrow_mut() = None;
        }
        "kvscreen" => {
            let eng = Engine::load(&o.model, o.ctx);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            eng.tail.set(false);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            check_prompt_fits_ctx(pt.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
            let n = pt.len().min(eng.chunk);
            eng.reset(); eng.reserve(n + 1, false); eng.ensure_batch(n);
            eng.set_tokens(0, &pt[..n]);
            let cb = eng.gpu.cmd(); eng.encode_prefill(&cb, 0, n, false, false); cb.commit(); cb.waitUntilCompleted();
            println!("last layer, {n} keys, last 48 queries x 24 heads: share of keys within a margin of the best score");
            for (m, mean, worst) in eng.kv_screen(n, 48, &[8.0, 12.0, 17.0, 25.0]) {
                println!("  within {m:4.0} logits (weight > e^-{m:.0}): mean {:6.2}% of keys, worst head {:6.2}%", mean * 100.0, worst * 100.0);
            }
        }
        "kvdump" => {
            // dev: prefill the prompt and write the K/V payload to <ref>.swa / <ref>.full (to study its redundancy)
            let eng = Engine::load(&o.model, o.ctx);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            let tk = Tokenizer::load(&eng.g);
            let pt = tk.encode(&first_turn(&o, &o.prompt), true);
            check_prompt_fits_ctx(pt.len(), eng.max_ctx).unwrap_or_else(|e| cli_error(e));
            let mut sess = Session::new(&eng, &tk);
            report_cli_result(&cmd, sess.prefill(&pt).map(|_| ()));
            let (swa, full) = eng.kv_dump(pt.len());
            std::fs::write(format!("{}.swa", o.reference), &swa).unwrap();
            std::fs::write(format!("{}.full", o.reference), &full).unwrap();
            println!("{} tokens: sliding-window payload {:.1} MiB, full-attention payload {:.1} MiB", pt.len(), swa.len() as f64 / 1048576.0, full.len() as f64 / 1048576.0);
        }
        "kpeak" => {
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            for (n, nsg) in [("kgpeak_8_0", 8), ("kgpeak_8_576", 8), ("kgpeak_8_128", 8), ("kgpeak_8_32", 8), ("kgpeak_4_0", 4), ("kgpeak_1_0", 1)] {
                println!("{n:>13}: {:5.1} TFLOPs", eng.kbench_peak(n, nsg, 1024));
            }
        }
        "kmode" => {
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            for n in ["kgmode_0_0", "kgmode_1_0", "kgmode_0_1", "kgmode_1_1"] {
                let (tf, rel, mx) = eng.kbench_mode(n, 1024);
                println!("{n}: {tf:5.1} TFLOPs   relative RMS error {rel:.2e}   max error / output RMS {mx:.2e}");
            }
        }
        "bw" => { let eng = Engine::load(&o.model, 4096);
 let lane = Lane::new(&eng);
 let eng = eng.on(&lane); eng.bw_weights(); }
        "kbench" => {
            let eng = Engine::load(&o.model, 4096);
            let lane = Lane::new(&eng);
            let eng = eng.on(&lane);
            if std::env::var("INK_KB_Q4").is_ok() { for (n, nr, nsg) in [("kb_dev_2_2", 2, 2), ("kb_dev_2_4", 2, 4), ("kb_dev_4_2", 4, 2),
                                 ("kb_dev_4_4", 4, 4), ("kb_dev_8_2", 8, 2), ("kb_dev_1_4", 1, 4), ("kb_blk_2_4", 2, 4), ("kb_blk_4_4", 4, 4),
                                 ("kb_blk_8_2", 8, 2), ("kb_blk_4_8", 4, 8)] {
                println!("{n:>14}: {:6.1} GB/s", eng.kbench(n, nr, nsg));
            } }
            for (n, nr, nsg) in [("k_moe_down", 2, 2), ("kd_dev_1_2", 1, 2), ("kd_dev_1_4", 1, 4), ("kd_dev_1_8", 1, 8), ("kd_dev_2_2", 2, 2),
                                 ("kd_dev_2_4", 2, 4), ("kd_dev_2_8", 2, 8), ("kd_dev_4_4", 4, 4), ("kd_tg_2_8", 2, 8)] {
                println!("{n:>14}: all {:6.1} GB/s | Q8_0 {:6.1} | Q5_0 {:6.1}", eng.kbench_down(n, nr * nsg, 32 * nsg, 0),
                         eng.kbench_down(n, nr * nsg, 32 * nsg, 8), eng.kbench_down(n, nr * nsg, 32 * nsg, 6));
            }
        }
        "run" => report_cli_result(&cmd, run(&o)),
        "chat" => report_cli_result(&cmd, chat(&o)),
        "bench" => report_cli_result(&cmd, bench(&o)),
        "help" => println!("{USAGE}"),
        _ => cli_error(format!("unknown command {cmd}\n\n{USAGE}")),
    }
}

#[cfg(test)]
mod loopcd_tests;
