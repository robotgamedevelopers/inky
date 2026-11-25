//! Test driver for the real scheduler. The workload advances only at tick boundaries;
//! receiver drops exercise the same output-channel failure as a disconnected client.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};

const LANES: usize = 2;
const QUEUE: usize = 1;
const CHUNK: usize = 2;
const CONTEXT: usize = 32;
const REQUESTS: u64 = 32;
/// Version 2: `admit` carries `source` and `from`; `refuse_ctx` ends a request too long for a lane.
const SCHEMA_VERSION: u64 = 2;

/// The scheduler still samples, updates positions, and decides when each lane is done.
/// Only the GPU work is substituted. Token 1 always wins; the two stop tokens are 2, 3.
struct FakeEngine;
impl backend::SchedulerEngine for FakeEngine {
    type Lane = ();
    fn new_lane(&self) -> Self::Lane {}
    fn max_ctx(&self) -> usize { CONTEXT }
    fn chunk(&self) -> usize { CHUNK }
    fn softcap(&self) -> f32 { 0.0 }
    fn tier_on(&self, _: &()) -> bool { false }
    fn prepare(&self, _: &[LaneS<()>], _: Admission, _: &Job) -> Duration { Duration::ZERO }
    fn prefill(&self, _: &[LaneS<()>], _: &[(usize, usize, usize)]) -> Duration { Duration::ZERO }
    fn decode(&self, _: &[LaneS<()>], _: &[usize]) -> Duration { Duration::ZERO }
    fn save_state(&self, _: &()) -> Vec<u8> { vec![] }
    fn result_logits(&self, _: Option<usize>) -> Vec<f32> { vec![0.0, 1.0] }
}

struct Workload<'a> {
    random: u64,
    tick: usize,
    submitted: u64,
    /// Requests sent to the scheduler whose trace has not yet ended (finish, cancel or refuse_ctx).
    outstanding: BTreeSet<u64>,
    /// The last prompt that fits a lane; a later prompt may extend it, so lanes reuse and copy prefixes.
    last: Vec<u32>,
    /// One request per walk is too long for a lane's context, so the context refusal is always exercised.
    oversized: bool,
    jobs: Option<Sender<Job>>,
    clients: BTreeMap<u64, (Receiver<Ev>, InFlight<'a>)>,
    inflight: &'a AtomicUsize,
    metrics: &'a Mutex<Counters>,
    events: Vec<Value>,
}

impl Workload<'_> {
    fn pick(&mut self, choices: usize) -> usize {
        self.random = self.random.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.random >> 32) as usize % choices
    }

    fn arrive(&mut self) {
        self.submitted += 1;
        let job = self.submitted;
        let slot = try_slot(self.inflight, LANES + QUEUE);
        let max = 2 + self.pick(4);
        let toks = if slot.is_some() && !self.oversized && job > (LANES + QUEUE + 1) as u64 {
            // prompt + max + 2 > CONTEXT: refused at admission, after it reaches a free lane
            self.oversized = true;
            vec![job as u32; CONTEXT - 1]
        } else {
            let extra = 1 + self.pick(7);
            let extend = !self.last.is_empty() && self.pick(2) == 0 && self.last.len() + extra + max + 2 <= CONTEXT;
            // A fresh prompt leads with its own unique token, so it shares nothing with any lane.
            let mut toks = if extend { self.last.clone() } else { vec![] };
            toks.extend(vec![job as u32; extra]);
            self.last = toks.clone();
            toks
        };
        self.events.push(json!({"event":"arrive","job":job,"prompt":toks.len(),"max":max}));
        let Some(slot) = slot else {
            self.events.push(json!({"event":"refuse429","job":job}));
            return;
        };
        let (out, rx) = channel();
        self.clients.insert(job, (rx, slot));
        self.outstanding.insert(job);
        self.metrics.lock().unwrap().queued += 1;
        self.jobs.as_ref().unwrap().send(Job {
            seq: job, queued_at: Instant::now(), toks,
            max, temp: 0.0, top_p: 1.0, seed: self.random, out,
        }).unwrap();
    }
}

impl Trace for Workload<'_> {
    fn boundary(&mut self, idle: bool) {
        self.tick += 1;
        // Requests reach the scheduler only from this thread, so at a boundary none is in flight on the
        // channel: an idle scheduler holds nothing, and every request sent must already have ended its trace.
        // Checked here, before the scheduler would block waiting for work, so a lost terminal event fails
        // the test instead of hanging it.
        assert!(!idle || self.outstanding.is_empty(),
                "tick {}: the scheduler is idle but requests {:?} never reached a terminal trace event", self.tick, self.outstanding);
        assert!(idle || !self.outstanding.is_empty(), "tick {}: the scheduler holds work after every request ended", self.tick);
        assert!(self.tick < 512, "scheduler did not drain the seeded workload");
        self.clients.retain(|_, (rx, _)| !rx.try_iter().any(|ev| matches!(ev, Ev::Done { .. } | Ev::Err(_))));
        if self.submitted == REQUESTS && self.outstanding.is_empty() {
            self.jobs.take();
            return;
        }
        if !self.clients.is_empty() && (self.tick == 2 || self.pick(8) == 0) {
            let index = self.pick(self.clients.len());
            let job = *self.clients.keys().nth(index).unwrap();
            self.clients.remove(&job);
            self.events.push(json!({"event":"disconnect","job":job}));
        }
        let arrivals = if self.tick == 1 { LANES + QUEUE + 1 } else if self.outstanding.is_empty() {
            // An idle scheduler blocks until a real request arrives.
            1
        } else {
            self.pick(3)
        };
        for _ in 0..arrivals.min((REQUESTS - self.submitted) as usize) { self.arrive(); }
    }

    fn admit(&mut self, job: u64, adm: Admission) {
        let source = if adm.prefix == 0 { "cold" } else if adm.src == adm.dst { "own" } else { "copy" };
        self.events.push(json!({"event":"admit","job":job,"lane":adm.dst,"cached":adm.prefix,"source":source,"from":adm.src}));
    }
    fn refuse_ctx(&mut self, job: u64) {
        assert!(self.outstanding.remove(&job), "request {job} ended twice or was never sent");
        self.events.push(json!({"event":"refuse_ctx","job":job}));
    }
    fn step(&mut self, mode: &str, spans: impl Iterator<Item = (u64,usize,usize,usize)>) {
        let spans: Vec<_> = spans.map(|(job,lane,pos,tokens)| json!({
            "job":job, "lane":lane, "pos":pos, "tokens":tokens,
        })).collect();
        self.events.push(json!({"event":"build_step","mode":mode,"spans":spans}));
    }
    fn step_finished(&mut self) { self.events.push(json!({"event":"finish_step"})); }
    fn terminal(&mut self, job: u64, lane: usize, reason: &str, generated: usize) {
        assert!(self.outstanding.remove(&job), "request {job} ended twice or was never sent");
        if reason == "cancelled" {
            self.events.push(json!({"event":"cancel","job":job,"lane":lane,"generated":generated}));
        } else {
            self.events.push(json!({"event":"finish","job":job,"lane":lane,"reason":reason,"generated":generated}));
        }
    }
}

fn walk(seed: u64) -> Vec<Value> {
    let inflight = AtomicUsize::new(0);
    let metrics = Mutex::new(Counters::new(Instant::now()));
    let (sender, receiver) = channel();
    let mut workload = Workload {
        random:seed, tick:0, submitted:0, outstanding:BTreeSet::new(), last:vec![], oversized:false, jobs:Some(sender),
        clients:BTreeMap::new(), inflight:&inflight, metrics:&metrics,
        events:vec![json!({"event":"config","version":SCHEMA_VERSION,"seed":seed,"lanes":LANES,"queue":QUEUE,"chunk":CHUNK,"context":CONTEXT})],
    };
    run_scheduler(&FakeEngine, [2, 3], LANES, receiver, &metrics, &mut workload);
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.lock().unwrap().queued, 0);
    workload.events
}

#[test]
fn seeded_decisions_are_repeatable_and_drain() {
    use std::collections::{BTreeSet, HashMap};
    let mut sources = HashMap::new();
    for seed in (0..32).chain([u64::MAX]) {
        let trace = walk(seed);
        assert_eq!(trace, walk(seed));
        assert_eq!(trace[0]["version"], SCHEMA_VERSION);
        let mut pending = BTreeSet::new();
        let mut prompts = HashMap::new();
        let mut maxes = HashMap::new();
        let mut lanes = [None; LANES];
        // the prompt length whose state each lane keeps: set when a prefill reaches the prompt's end
        let mut kept = [None; LANES];
        let mut phases = BTreeSet::new();
        let mut cases = HashMap::new();
        let mut executing = false;
        for ev in &trace[1..] {
            let kind = ev["event"].as_str().unwrap();
            *cases.entry(kind).or_insert(0) += 1;
            match kind {
                "arrive" => {
                    let job = ev["job"].as_u64().unwrap();
                    assert!(pending.insert(job));
                    prompts.insert(job, ev["prompt"].as_u64().unwrap());
                    maxes.insert(job, ev["max"].as_u64().unwrap());
                }
                "refuse429" => assert!(pending.remove(&ev["job"].as_u64().unwrap())),
                "refuse_ctx" => {
                    let job = ev["job"].as_u64().unwrap();
                    assert!(pending.remove(&job));
                    assert!((prompts[&job] + maxes[&job]) as usize + 2 > CONTEXT);
                }
                "admit" => {
                    let job = ev["job"].as_u64().unwrap();
                    assert!(pending.remove(&job));
                    let lane = ev["lane"].as_u64().unwrap() as usize;
                    let from = ev["from"].as_u64().unwrap() as usize;
                    let cached = ev["cached"].as_u64().unwrap();
                    let source = ev["source"].as_str().unwrap();
                    *sources.entry(source.to_string()).or_insert(0) += 1;
                    assert!(cached < prompts[&job], "the suffix to prefill is never empty");
                    match source {
                        "cold" => assert_eq!((cached, from), (0, lane)),
                        "own" => assert!(from == lane && kept[lane] == Some(cached)),
                        "copy" => {
                            assert!(from != lane && kept[from] == Some(cached));
                            // the source is busy, its prefill done and no decode step written yet
                            let (src_job, pos) = lanes[from].unwrap();
                            assert_eq!((pos, prompts[&src_job]), (cached, cached));
                        }
                        other => panic!("unexpected admit source {other}"),
                    }
                    kept[lane] = None;
                    assert!(lanes[lane].replace((job, cached)).is_none());
                }
                "build_step" => {
                    assert!(!executing);
                    executing = true;
                    let phase = ev["mode"].as_str().unwrap();
                    phases.insert(phase);
                    let mut chosen = BTreeSet::new();
                    let mut total = 0;
                    for span in ev["spans"].as_array().unwrap() {
                        let lane = span["lane"].as_u64().unwrap() as usize;
                        assert!(chosen.insert(lane));
                        let (job, pos) = lanes[lane].as_mut().unwrap();
                        assert_eq!((*job, *pos), (span["job"].as_u64().unwrap(), span["pos"].as_u64().unwrap()));
                        let n = span["tokens"].as_u64().unwrap();
                        assert!(n > 0);
                        if phase == "decode" { assert_eq!(n, 1); }
                        *pos += n;
                        total += n;
                        if phase == "prefill" && *pos == prompts[&*job] { kept[lane] = Some(*pos); }
                    }
                    assert!(!chosen.is_empty());
                    if phase == "prefill" { assert!(total <= CHUNK as u64); }
                }
                "finish_step" => { assert!(executing); executing = false; }
                "finish" | "cancel" => {
                    assert!(!executing);
                    let lane = ev["lane"].as_u64().unwrap() as usize;
                    assert_eq!(lanes[lane].take().unwrap().0, ev["job"].as_u64().unwrap());
                }
                "disconnect" => {}
                other => panic!("unexpected trace event {other}"),
            }
        }
        assert!(pending.is_empty() && lanes.iter().all(Option::is_none) && !executing);
        assert_eq!(phases, BTreeSet::from(["decode", "prefill"]));
        assert_eq!(cases["arrive"], REQUESTS);
        for kind in ["finish", "cancel", "refuse429"] { assert!(cases[kind] > 0); }
        assert_eq!(cases["refuse_ctx"], 1);
    }
    for source in ["cold", "own", "copy"] { assert!(sources.get(source).is_some_and(|&n| n > 0), "no {source} admission in any seed"); }
}

#[test]
#[ignore = "writes versioned scheduler JSONL to INKY_SCHED_TRACE_OUT"]
fn dump() {
    let path = std::env::var_os("INKY_SCHED_TRACE_OUT").expect("set INKY_SCHED_TRACE_OUT");
    let seed = std::env::var("INKY_SCHED_TRACE_SEED").map(|s| s.parse().expect("seed must be a u64")).unwrap_or(1);
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for event in walk(seed) { writeln!(out, "{event}").unwrap(); }
    out.flush().unwrap();
}
