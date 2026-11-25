//! Scheduler wall time. Each participating lane is charged the duration of its batch. A request's wall time
//! from arrival to its last token is queue wait, then lane preparation, then its prefill and decode batches.
use std::fmt;
use std::time::{Duration, Instant};

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() { 0.0 } else { tokens as f64 / elapsed.as_secs_f64() }
}

#[derive(Default)]
pub(super) struct Phase {
    tokens: usize,
    elapsed: Duration,
}

impl Phase {
    pub(super) fn add(&mut self, tokens: usize, elapsed: Duration) {
        self.tokens += tokens;
        self.elapsed += elapsed;
    }

    pub(super) fn elapsed_ms(&self) -> u64 {
        self.elapsed.as_millis() as u64
    }
}

/// Completed per-request phase timing carried to request handlers, logs, and the traffic recorder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct TimingSnapshot {
    pub(super) queue_ms: u64,
    pub(super) prefill_ms: u64,
    pub(super) decode_ms: u64,
    pub(super) summary: String,
}

pub(super) struct RequestTiming {
    pub(super) queue_wait: Duration,
    pub(super) busy_at_start: usize,
    /// Readying the lane after admission: prefix match/restore, reset, kept state, prompt upload.
    pub(super) prep: Duration,
    pub(super) prefill: Phase,
    pub(super) generation: Phase,
}

impl RequestTiming {
    pub(super) fn snapshot(&self) -> TimingSnapshot {
        TimingSnapshot {
            queue_ms: self.queue_wait.as_millis() as u64,
            prefill_ms: self.prefill.elapsed_ms(),
            decode_ms: self.generation.elapsed_ms(),
            summary: self.to_string(),
        }
    }
}

impl fmt::Display for RequestTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "queue_ms={:.1} prep_ms={:.1} prefill_ms={:.1} prefill_tok_s={:.1} decode_ms={:.1} decode_tok_s={:.1} lanes_busy_start={}",
            self.queue_wait.as_secs_f64() * 1000.0,
            self.prep.as_secs_f64() * 1000.0,
            self.prefill.elapsed.as_secs_f64() * 1000.0,
            rate(self.prefill.tokens, self.prefill.elapsed),
            self.generation.elapsed.as_secs_f64() * 1000.0,
            rate(self.generation.tokens, self.generation.elapsed),
            self.busy_at_start,
        )
    }
}

pub(super) struct Counters {
    start: Instant,
    last_change: Instant,
    busy: usize,
    lane_seconds: f64,
    pub(super) queued: usize,
    pub(super) starts_spilled: u64,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) admit_deferrals: u64,
    pub(super) step_deferrals: u64,
}

impl Counters {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            start: now,
            last_change: now,
            busy: 0,
            lane_seconds: 0.0,
            queued: 0,
            starts_spilled: 0,
            input: 0,
            output: 0,
            admit_deferrals: 0,
            step_deferrals: 0,
        }
    }

    fn accrue(&mut self, now: Instant) {
        self.lane_seconds += now.saturating_duration_since(self.last_change).as_secs_f64() * self.busy as f64;
        self.last_change = now;
    }

    pub(super) fn admit(&mut self, now: Instant, queued_at: Instant) -> RequestTiming {
        self.accrue(now);
        self.queued = self.queued.saturating_sub(1);
        self.busy += 1;
        RequestTiming {
            queue_wait: now.saturating_duration_since(queued_at),
            busy_at_start: self.busy,
            prep: Duration::ZERO,
            prefill: Phase::default(),
            generation: Phase::default(),
        }
    }

    pub(super) fn release(&mut self, now: Instant) {
        self.accrue(now);
        self.busy = self.busy.saturating_sub(1);
    }

    pub(super) fn summarize(&mut self, now: Instant, waiting_in_gate: usize) -> String {
        self.accrue(now);
        let elapsed = now.saturating_duration_since(self.start);
        let lanes_busy_avg = if elapsed.is_zero() {
            self.busy as f64
        } else {
            self.lane_seconds / elapsed.as_secs_f64()
        };
        let queue_depth = self.queued.saturating_add(waiting_in_gate);
        let line = format!(
            "[serve] summary interval_s={:.1} lanes_busy_avg={:.2} input_tok_s={:.2} output_tok_s={:.2} queue_depth={}",
            elapsed.as_secs_f64(),
            lanes_busy_avg,
            rate(self.input, elapsed),
            rate(self.output, elapsed),
            queue_depth,
        );
        self.start = now;
        self.lane_seconds = 0.0;
        self.input = 0;
        self.output = 0;
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_charges_only_its_prefill_and_decode_batches() {
        let start = Instant::now();
        let mut counters = Counters::new(start);
        counters.queued = 2;
        counters.admit(start, start);
        let mut timing = counters.admit(start + Duration::from_millis(50), start);
        timing.prefill.add(4, Duration::from_millis(25));
        timing.prefill.add(6, Duration::from_millis(75));
        assert_eq!(
            timing.to_string(),
            "queue_ms=50.0 prep_ms=0.0 prefill_ms=100.0 prefill_tok_s=100.0 decode_ms=0.0 decode_tok_s=0.0 lanes_busy_start=2"
        );
        timing.generation.add(1, Duration::from_millis(50));
        let snap = timing.snapshot();
        assert_eq!(snap.queue_ms, 50);
        assert_eq!(snap.prefill_ms, 100);
        assert_eq!(snap.decode_ms, 50);
        assert!(snap.summary.contains("decode_ms=50.0 decode_tok_s=20.0"));
    }

    #[test]
    fn lane_preparation_is_its_own_phase() {
        // Admission at 40 ms, preparation 15 ms, then a 30 ms prefill: preparation is charged to neither the
        // queue wait (it starts after admission) nor the prefill rate (it moves no prompt tokens).
        let queued = Instant::now();
        let mut counters = Counters::new(queued);
        counters.queued = 1;
        let mut timing = counters.admit(queued + Duration::from_millis(40), queued);
        timing.prep = Duration::from_millis(15);
        timing.prefill.add(6, Duration::from_millis(30));
        assert_eq!(
            timing.to_string(),
            "queue_ms=40.0 prep_ms=15.0 prefill_ms=30.0 prefill_tok_s=200.0 decode_ms=0.0 decode_tok_s=0.0 lanes_busy_start=1"
        );
        let total = timing.queue_wait + timing.prep + timing.prefill.elapsed;
        assert_eq!(total, Duration::from_millis(85));
    }

    #[test]
    fn summary_accounts_for_busy_idle_and_cross_window_jobs() {
        let start = Instant::now();
        let at = |s| start + Duration::from_secs(s);
        let mut c = Counters::new(start);
        c.queued = 3;
        c.admit(at(10), start);
        c.admit(at(20), start);
        c.release(at(40));
        c.input = 300;
        c.output = 60;
        assert_eq!(
            c.summarize(at(60), 0),
            "[serve] summary interval_s=60.0 lanes_busy_avg=1.17 input_tok_s=5.00 output_tok_s=1.00 queue_depth=1"
        );
        c.release(at(90));
        assert_eq!(
            c.summarize(at(120), 2),
            "[serve] summary interval_s=60.0 lanes_busy_avg=0.50 input_tok_s=0.00 output_tok_s=0.00 queue_depth=3"
        );
        assert!(c.summarize(at(180), 0).contains("lanes_busy_avg=0.00"));
        // Zero-duration summary must never produce NaN.
        assert!(!c.summarize(at(180), 0).contains("NaN"));
    }
}
