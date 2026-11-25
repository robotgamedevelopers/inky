//! The scheduler chooses the lane and the batch. This boundary executes that
//! choice and supplies its logits; the trace workload can run it without Metal.
use super::{Admission, Job, LaneS};
use crate::engine::{Engine, Lane, Span};
use objc2_metal::MTLCommandBuffer;
use std::time::{Duration, Instant};

pub(super) trait SchedulerEngine {
    type Lane;
    fn new_lane(&self) -> Self::Lane;
    fn max_ctx(&self) -> usize;
    fn chunk(&self) -> usize;
    fn softcap(&self) -> f32;
    fn tier_on(&self, lane: &Self::Lane) -> bool;
    /// Readies the destination lane for the request (reset, reservation, a cross-lane prefix copy, the kept
    /// convolution state, the prompt tokens) and returns the wall time that took.
    fn prepare(&self, lanes: &[LaneS<Self::Lane>], adm: Admission, j: &Job) -> Duration;
    fn prefill(&self, lanes: &[LaneS<Self::Lane>], spans: &[(usize, usize, usize)]) -> Duration;
    fn decode(&self, lanes: &[LaneS<Self::Lane>], chosen: &[usize]) -> Duration;
    fn save_state(&self, lane: &Self::Lane) -> Vec<u8>;
    /// A single-lane prefill writes the engine's logits; all other steps write per-span rows.
    fn result_logits(&self, row: Option<usize>) -> Vec<f32>;
}

impl SchedulerEngine for Engine {
    type Lane = Lane;

    fn new_lane(&self) -> Lane { Lane::new(self) }
    fn max_ctx(&self) -> usize { self.max_ctx }
    fn chunk(&self) -> usize { self.chunk }
    fn softcap(&self) -> f32 { self.softcap }
    fn tier_on(&self, lane: &Lane) -> bool { self.on(lane).tier_on() }

    fn prepare(&self, lanes: &[LaneS<Lane>], adm: Admission, j: &Job) -> Duration {
        let began = Instant::now();
        let dst = &lanes[adm.dst].lane;
        let run = self.on(dst);
        let capacity = j.toks.len() + j.max + 2;
        if adm.prefix > 0 && adm.src == adm.dst {
            run.set_cs(lanes[adm.dst].cs_at_prompt.as_ref().unwrap());
        } else if adm.prefix > 0 {
            let src = &lanes[adm.src];
            run.reset();
            run.reserve(capacity, true);
            self.copy_lane_prefix(&src.lane, dst, adm.prefix);
            run.set_cs(src.cs_at_prompt.as_ref().unwrap());
        } else {
            run.reset();
        }
        run.reserve(capacity, true);
        run.set_tokens(0, &j.toks);
        began.elapsed()
    }

    fn prefill(&self, lanes: &[LaneS<Lane>], plan: &[(usize, usize, usize)]) -> Duration {
        let spans: Vec<Span> = plan.iter().map(|&(i, p0, t)| Span { lane: &lanes[i].lane, p0, t }).collect();
        let began = Instant::now();
        let cb = self.gpu.cmd();
        self.encode_spans(&cb, &spans, true, false);
        cb.commit();
        cb.waitUntilCompleted();
        began.elapsed()
    }

    fn decode(&self, lanes: &[LaneS<Lane>], chosen: &[usize]) -> Duration {
        for &i in chosen {
            let a = lanes[i].job.as_ref().unwrap();
            self.on(&lanes[i].lane).set_tokens(a.n_past, &[a.pending.unwrap()]);
        }
        let spans: Vec<Span> = chosen.iter().map(|&i| Span {
            lane: &lanes[i].lane, p0: lanes[i].job.as_ref().unwrap().n_past, t: 1,
        }).collect();
        let began = Instant::now();
        let cb = self.gpu.cmd();
        self.encode_decode(&cb, &spans, true, false);
        cb.commit();
        cb.waitUntilCompleted();
        began.elapsed()
    }

    fn save_state(&self, lane: &Lane) -> Vec<u8> { self.on(lane).get_cs() }
    fn result_logits(&self, row: Option<usize>) -> Vec<f32> {
        match row { Some(k) => self.span_logits(k).to_vec(), None => self.logits().to_vec() }
    }
}
