//! Scheduler wall time. Each participating lane is charged the duration of its batch.
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
}

pub(super) struct RequestTiming {
    pub(super) queue_wait: Duration,
    pub(super) busy_at_start: usize,
    pub(super) prefill: Phase,
    pub(super) generation: Phase,
}

impl fmt::Display for RequestTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "queue_ms={:.1} prefill_ms={:.1} prefill_tok_s={:.1} gen_ms={:.1} gen_tok_s={:.1} lanes_busy_start={}",
               self.queue_wait.as_secs_f64() * 1000.0,
               self.prefill.elapsed.as_secs_f64() * 1000.0, rate(self.prefill.tokens, self.prefill.elapsed),
               self.generation.elapsed.as_secs_f64() * 1000.0, rate(self.generation.tokens, self.generation.elapsed),
               self.busy_at_start)
    }
}

pub(super) struct Counters {
    start: Instant,
    last_change: Instant,
    busy: usize,
    lane_seconds: f64,
    pub(super) queued: usize,
    pub(super) input: usize,
    pub(super) output: usize,
}

impl Counters {
    pub(super) fn new(now: Instant) -> Self {
        Self { start: now, last_change: now, busy: 0, lane_seconds: 0.0, queued: 0, input: 0, output: 0 }
    }

    fn accrue(&mut self, now: Instant) {
        self.lane_seconds += (now - self.last_change).as_secs_f64() * self.busy as f64;
        self.last_change = now;
    }

    pub(super) fn admit(&mut self, now: Instant, queued_at: Instant) -> RequestTiming {
        self.accrue(now);
        self.queued -= 1;
        self.busy += 1;
        RequestTiming { queue_wait: now - queued_at, busy_at_start: self.busy,
                        prefill: Phase::default(), generation: Phase::default() }
    }

    pub(super) fn release(&mut self, now: Instant) {
        self.accrue(now);
        self.busy -= 1;
    }

    pub(super) fn summarize(&mut self, now: Instant) -> String {
        self.accrue(now);
        let elapsed = now - self.start;
        let line = format!("[serve] summary interval_s={:.1} lanes_busy_avg={:.2} input_tok_s={:.2} output_tok_s={:.2} queue_depth={}",
                           elapsed.as_secs_f64(), self.lane_seconds / elapsed.as_secs_f64(),
                           rate(self.input, elapsed), rate(self.output, elapsed), self.queued);
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
        assert_eq!(timing.to_string(), "queue_ms=50.0 prefill_ms=100.0 prefill_tok_s=100.0 gen_ms=0.0 gen_tok_s=0.0 lanes_busy_start=2");
        timing.generation.add(1, Duration::from_millis(50));
        assert!(timing.to_string().contains("gen_ms=50.0 gen_tok_s=20.0"));
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
        assert_eq!(c.summarize(at(60)), "[serve] summary interval_s=60.0 lanes_busy_avg=1.17 input_tok_s=5.00 output_tok_s=1.00 queue_depth=1");
        c.release(at(90));
        assert_eq!(c.summarize(at(120)), "[serve] summary interval_s=60.0 lanes_busy_avg=0.50 input_tok_s=0.00 output_tok_s=0.00 queue_depth=1");
        assert!(c.summarize(at(180)).contains("lanes_busy_avg=0.00"));
    }
}
