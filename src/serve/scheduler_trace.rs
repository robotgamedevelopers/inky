//! Test driver for the real scheduler. The workload advances only at tick boundaries;
//! receiver drops exercise the same output-channel failure as a disconnected client.
use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

const LANES: usize = 2;
const QUEUE: usize = 1;
const CHUNK: usize = 2;
const CONTEXT: usize = 32;
const REQUESTS: u64 = 32;

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
    fn prepare(&self, _: &[LaneS<()>], _: Admission, _: &Job) {}
    fn prefill(&self, _: &[LaneS<()>], _: &[(usize, usize, usize)]) -> Duration { Duration::ZERO }
    fn decode(&self, _: &[LaneS<()>], _: &[usize]) -> Duration { Duration::ZERO }
    fn save_state(&self, _: &()) -> Vec<u8> { vec![] }
    fn result_logits(&self, _: Option<usize>) -> &[f32] { &[0.0, 1.0] }
}

struct Workload<'a> {
    random: u64,
    tick: usize,
    submitted: u64,
    outstanding: usize,
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
        let prompt = 1 + self.pick(7);
        let max = 2 + self.pick(4);
        self.events.push(json!({"event":"arrive","job":job,"prompt":prompt,"max":max}));
        let Some(slot) = try_slot(self.inflight, LANES + QUEUE) else {
            self.events.push(json!({"event":"refuse429","job":job}));
            return;
        };
        let (out, rx) = channel();
        self.clients.insert(job, (rx, slot));
        self.outstanding += 1;
        self.metrics.lock().unwrap().queued += 1;
        // A unique leading token keeps this bounded scheduling workload cold.
        self.jobs.as_ref().unwrap().send(Job {
            seq: job, queued_at: Instant::now(), toks: vec![job as u32; prompt],
            max, temp: 0.0, top_p: 1.0, seed: self.random, out,
        }).unwrap();
    }
}

impl Trace for Workload<'_> {
    fn boundary(&mut self) {
        self.tick += 1;
        assert!(self.tick < 512, "scheduler did not drain the seeded workload");
        self.clients.retain(|_, (rx, _)| !rx.try_iter().any(|ev| matches!(ev, Ev::Done { .. })));
        if self.submitted == REQUESTS && self.outstanding == 0 {
            self.jobs.take();
            return;
        }
        if !self.clients.is_empty() && (self.tick == 2 || self.pick(8) == 0) {
            let index = self.pick(self.clients.len());
            let job = *self.clients.keys().nth(index).unwrap();
            self.clients.remove(&job);
            self.events.push(json!({"event":"disconnect","job":job}));
        }
        let arrivals = if self.tick == 1 { LANES + QUEUE + 1 } else if self.outstanding == 0 {
            // An idle scheduler blocks until a real request arrives.
            1
        } else {
            self.pick(3)
        };
        for _ in 0..arrivals.min((REQUESTS - self.submitted) as usize) { self.arrive(); }
    }

    fn admit(&mut self, job: u64, lane: usize, cached: usize) {
        self.events.push(json!({"event":"admit","job":job,"lane":lane,"cached":cached}));
    }
    fn step(&mut self, mode: &str, spans: impl Iterator<Item = (u64,usize,usize,usize)>) {
        let spans: Vec<_> = spans.map(|(job,lane,pos,tokens)| json!({
            "job":job, "lane":lane, "pos":pos, "tokens":tokens,
        })).collect();
        self.events.push(json!({"event":"build_step","mode":mode,"spans":spans}));
    }
    fn step_finished(&mut self) { self.events.push(json!({"event":"finish_step"})); }
    fn terminal(&mut self, job: u64, lane: usize, reason: &str, generated: usize) {
        self.outstanding -= 1;
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
        random:seed, tick:0, submitted:0, outstanding:0, jobs:Some(sender),
        clients:BTreeMap::new(), inflight:&inflight, metrics:&metrics,
        events:vec![json!({"event":"config","version":1,"seed":seed,"lanes":LANES,"queue":QUEUE,"chunk":CHUNK,"context":CONTEXT})],
    };
    run_scheduler(&FakeEngine, [2, 3], LANES, receiver, &metrics, &mut workload);
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.lock().unwrap().queued, 0);
    workload.events
}

#[test]
fn seeded_decisions_are_repeatable_and_drain() {
    use std::collections::{BTreeSet, HashMap};
    for seed in (0..32).chain([u64::MAX]) {
        let trace = walk(seed);
        assert_eq!(trace, walk(seed));
        let mut pending = BTreeSet::new();
        let mut lanes = [None; LANES];
        let mut phases = BTreeSet::new();
        let mut cases = HashMap::new();
        let mut executing = false;
        for ev in &trace[1..] {
            let kind = ev["event"].as_str().unwrap();
            *cases.entry(kind).or_insert(0) += 1;
            match kind {
                "arrive" => assert!(pending.insert(ev["job"].as_u64().unwrap())),
                "refuse429" => assert!(pending.remove(&ev["job"].as_u64().unwrap())),
                "admit" => {
                    let job = ev["job"].as_u64().unwrap();
                    assert!(pending.remove(&job));
                    let lane = ev["lane"].as_u64().unwrap() as usize;
                    assert!(lanes[lane].replace((job, ev["cached"].as_u64().unwrap())).is_none());
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
    }
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
