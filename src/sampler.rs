//! CPU sampler over the GPU logits: softcap -> top-k -> temperature -> top-p -> draw.

pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub softcap: f32,
    state: u64,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(temp: f32, top_k: usize, top_p: f32, softcap: f32, seed: u64) -> Sampler {
        Sampler { temp, top_k: top_k.max(1), top_p, softcap, state: seed ^ 0x9E37_79B9_7F4A_7C15, cand: Vec::with_capacity(256) }
    }

    pub fn greedy(&self) -> bool { self.temp <= 0.0 }

    fn next_f32(&mut self) -> f32 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Draft stream for one request. The mix is `0x85ebca6b`, applied once here, not again per round.
    pub fn draft(temp: f32, top_k: usize, top_p: f32, softcap: f32, request_seed: u64) -> Sampler {
        Sampler::new(temp, top_k, top_p, softcap, request_seed ^ DRAFT_SEED_MIX)
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // top-k on raw logits (softcap is monotonic, so the order is the same)
        let k = self.top_k.min(logits.len());
        select_top_k(logits, k, &mut self.cand);
        self.cand.sort_by(|a, b| b.0.total_cmp(&a.0));     // total order: a NaN logit cannot panic the scheduler
        let c = self.softcap;
        let t = self.temp;
        let scaled: Vec<f32> = self.cand.iter().map(|&(x, _)| (if c > 0.0 { c * (x / c).tanh() } else { x }) / t).collect();
        let m = scaled[0];
        let mut p: Vec<f32> = scaled.iter().map(|&x| (x - m).exp()).collect();
        let sum: f32 = p.iter().sum();
        for v in p.iter_mut() { *v /= sum; }
        // top-p cut
        let mut cum = 0.0;
        let mut n = p.len();
        for (i, &v) in p.iter().enumerate() { cum += v; if cum >= self.top_p { n = i + 1; break; } }
        let total: f32 = p[..n].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..n { r -= p[i]; if r <= 0.0 { return self.cand[i].1; } }
        self.cand[n - 1].1
    }

    /// One drafted position. Draw from q, accept with min(1, p/q), else draw `norm(max(0, p-q))`.
    /// Every uniform comes from this sampler. `temp <= 0` does not belong here.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reject_one(&mut self, target: &[f32], draft: &[f32]) -> u32 {
        self.reject_at(target, draft).0
    }

    /// One verify round on the probabilistic path. `target_rows` is the draft positions plus the
    /// bonus row. A rejection stops the round. The bonus draw, when every draft is accepted, uses
    /// this same stream. One call is one round: it is not reseeded per emitted token.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn verify_round(&mut self, target_rows: &[&[f32]], draft_rows: &[&[f32]]) -> Vec<u32> {
        assert!(self.temp > 0.0, "greedy does not reject");
        assert_eq!(target_rows.len(), draft_rows.len() + 1, "one target row per draft token, plus the bonus");
        let mut out = Vec::with_capacity(target_rows.len());
        for i in 0..draft_rows.len() {
            let (tok, accepted) = self.reject_at(target_rows[i], draft_rows[i]);
            out.push(tok);
            if !accepted { return out; }
        }
        out.push(self.sample(target_rows[draft_rows.len()]));
        out
    }

    fn reject_at(&mut self, target: &[f32], draft: &[f32]) -> (u32, bool) {
        assert!(self.temp > 0.0, "greedy does not reject");
        let masses = self.masses(target, draft);
        let q: Vec<(u32, f32)> = masses.iter().filter(|m| m.q > 0.0).map(|m| (m.token, m.q)).collect();
        let x = self.draw_dist(&q);
        let px = prob_of(&masses, x, true);
        let qx = prob_of(&masses, x, false);
        // p >= q accepts without a draw; the uniform is only the case that can reject
        let accept = qx > 0.0 && (px >= qx || self.next_f32() < px / qx);
        if accept { return (x, true); }
        let mut resid: Vec<(u32, f32)> = Vec::new();
        for m in &masses {
            let r = m.p - m.q;
            if r > 0.0 { resid.push((m.token, r)); }
        }
        if resid.is_empty() {
            let p: Vec<(u32, f32)> = masses.iter().filter(|m| m.p > 0.0).map(|m| (m.token, m.p)).collect();
            return (self.draw_dist(&p), false);
        }
        (self.draw_dist(&resid), false)
    }

    fn draw_dist(&mut self, dist: &[(u32, f32)]) -> u32 {
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.next_f32() * total;
        for &(tok, p) in dist {
            r -= p;
            if r <= 0.0 { return tok; }
        }
        dist[dist.len() - 1].0
    }

    /// p and q on the union of the two top-k sets, after softcap-after-top-k, `/temp`, and top-p.
    /// A token the other side alone kept is present with a zero. Neither side is renormalized for that zero.
    fn masses(&self, target: &[f32], draft: &[f32]) -> Vec<Mass> {
        let k = self.top_k;
        let mut target_kept = Vec::new();
        let mut draft_kept = Vec::new();
        select_top_k(target, k.min(target.len()), &mut target_kept);
        select_top_k(draft, k.min(draft.len()), &mut draft_kept);
        let draft_ids: Vec<u32> = draft_kept.iter().map(|c| c.1).collect();
        let target_ids: Vec<u32> = target_kept.iter().map(|c| c.1).collect();
        // zeros of the other side sit on the support, after this side's own top-k, and do not enter its softmax
        let p_rows = side_probs(&mut target_kept, &draft_ids, self.softcap, self.temp, self.top_p);
        let q_rows = side_probs(&mut draft_kept, &target_ids, self.softcap, self.temp, self.top_p);
        let mut by_token: Vec<Mass> = Vec::new();
        for (tok, pr) in p_rows {
            match by_token.iter_mut().find(|m| m.token == tok) {
                Some(m) => m.p = pr,
                None => by_token.push(Mass { token: tok, p: pr, q: 0.0 }),
            }
        }
        for (tok, pr) in q_rows {
            match by_token.iter_mut().find(|m| m.token == tok) {
                Some(m) => m.q = pr,
                None => by_token.push(Mass { token: tok, p: 0.0, q: pr }),
            }
        }
        by_token.sort_by_key(|m| m.token);
        by_token
    }
}

/// Mixed into the request seed once, for the draft sampler only.
pub const DRAFT_SEED_MIX: u64 = 0x85ebca6b;

struct Mass { token: u32, p: f32, q: f32 }

fn select_top_k(logits: &[f32], k: usize, cand: &mut Vec<(f32, u32)>) {
    cand.clear();
    let mut floor = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x <= floor && cand.len() == k { continue; }
        if cand.len() < k {
            cand.push((x, i as u32));
            if cand.len() == k { floor = cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min); }
        } else {
            let j = cand.iter().position(|c| c.0 == floor).unwrap();
            cand[j] = (x, i as u32);
            floor = cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
        }
    }
}

/// Softcap after the top-k cut, then `/temp`, then the same top-p. `zeros` are the other side's
/// top-k tokens: they extend the support with probability 0 and are not in the softmax.
fn side_probs(cand: &mut Vec<(f32, u32)>, zeros: &[u32], softcap: f32, temp: f32, top_p: f32) -> Vec<(u32, f32)> {
    if cand.is_empty() { return Vec::new(); }
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let scaled: Vec<f32> = cand.iter().map(|&(x, _)| (if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x }) / temp).collect();
    let m = scaled[0];
    let mut pr: Vec<f32> = scaled.iter().map(|&x| (x - m).exp()).collect();
    let sum: f32 = pr.iter().sum();
    for v in pr.iter_mut() { *v /= sum; }
    let mut rows: Vec<(u32, f32)> = cand.iter().zip(pr).map(|(&(_, tok), p)| (tok, p)).collect();
    for &tok in zeros {
        if !rows.iter().any(|r| r.0 == tok) { rows.push((tok, 0.0)); }
    }
    let mut cum = 0.0;
    let mut n = rows.len();
    for (i, &(_, v)) in rows.iter().enumerate() {
        cum += v;
        if cum >= top_p { n = i + 1; break; }
    }
    let total: f32 = rows[..n].iter().map(|r| r.1).sum();
    let mut out = Vec::with_capacity(n);
    if total > 0.0 {
        for (tok, p) in rows.into_iter().take(n) {
            if p > 0.0 { out.push((tok, p / total)); }
        }
    }
    out
}

fn prob_of(masses: &[Mass], token: u32, target_side: bool) -> f32 {
    match masses.iter().find(|m| m.token == token) {
        Some(m) => if target_side { m.p } else { m.q },
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOFTCAP: f32 = 20.0;
    const TEMP: f32 = 1.0;
    const TOP_P: f32 = 0.8;
    const K: usize = 64;

    /// Target argmax is token 100, outside the draft's top-64. Token 0 is in the draft's top-64
    /// and outside the target's. Token 150 is in neither.
    fn logits() -> (Vec<f32>, Vec<f32>) {
        let mut target = vec![-50.0f32; 200];
        let mut draft = vec![-50.0f32; 200];
        for i in 0..64 { draft[i] = 6.0 - i as f32 * 0.01; }
        target[100] = 5.0;
        for i in 10..73 { target[i] = 4.0 - (i as f32 - 10.0) * 0.01; }
        (target, draft)
    }

    /// The order this step uses: top-k on the raw logits, then softcap, then divide by temperature, then top-p.
    /// The other side's tokens are not in the softmax.
    fn expected_side(logits: &[f32], k: usize, softcap: f32, temp: f32, top_p: f32) -> Vec<(u32, f32)> {
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]).then(a.cmp(&b)));
        order.truncate(k.min(logits.len()));
        let scaled: Vec<f32> = order.iter().map(|&i| {
            let x = logits[i as usize];
            (if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x }) / temp
        }).collect();
        let m = scaled[0];
        let mut pr: Vec<f32> = scaled.iter().map(|&x| (x - m).exp()).collect();
        let sum: f32 = pr.iter().sum();
        for v in pr.iter_mut() { *v /= sum; }
        let mut cum = 0.0;
        let mut n = pr.len();
        for (i, &v) in pr.iter().enumerate() {
            cum += v;
            if cum >= top_p { n = i + 1; break; }
        }
        let total: f32 = pr[..n].iter().sum();
        order.into_iter().zip(pr).take(n).filter(|&(_, p)| p > 0.0).map(|(t, p)| (t, p / total)).collect()
    }

    #[test]
    fn residual_emits_target_argmax_outside_draft_top64_softcap_after_topk() {
        let (target, draft) = logits();
        let argmax = target.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        assert_eq!(argmax, 100, "the synthetic target argmax is the token outside the draft top-64");
        let mut draft_order: Vec<usize> = (0..draft.len()).collect();
        draft_order.sort_by(|&a, &b| draft[b].total_cmp(&draft[a]));
        assert!(!draft_order[..K].contains(&(argmax as usize)), "argmax must sit outside the draft top-64");
        assert!(draft_order[..K].contains(&0), "token 0 is in the draft top-64");
        let mut target_order: Vec<usize> = (0..target.len()).collect();
        target_order.sort_by(|&a, &b| target[b].total_cmp(&target[a]));
        assert!(!target_order[..K].contains(&0), "token 0 is outside the target top-64");
        assert!(!target_order[..K].contains(&150) && !draft_order[..K].contains(&150), "150 is in neither top-64");

        let sampler = Sampler::new(TEMP, K, TOP_P, SOFTCAP, 1);
        let masses = sampler.masses(&target, &draft);
        let exp_p = expected_side(&target, K, SOFTCAP, TEMP, TOP_P);
        let exp_q = expected_side(&draft, K, SOFTCAP, TEMP, TOP_P);
        assert!(exp_p.len() < K && exp_q.len() < K, "top-p cuts both sides on this draw");
        let p_of = |tok: u32| masses.iter().find(|m| m.token == tok).map(|m| m.p).unwrap_or(0.0);
        let q_of = |tok: u32| masses.iter().find(|m| m.token == tok).map(|m| m.q).unwrap_or(0.0);
        for &(tok, p) in &exp_p { assert!((p_of(tok) - p).abs() < 1e-6, "p({tok})"); }
        for &(tok, q) in &exp_q { assert!((q_of(tok) - q).abs() < 1e-6, "q({tok})"); }
        assert!(masses.iter().all(|m| exp_p.iter().any(|e| e.0 == m.token) || m.p == 0.0));
        assert!(masses.iter().all(|m| exp_q.iter().any(|e| e.0 == m.token) || m.q == 0.0));
        assert_eq!(q_of(argmax), 0.0, "a token the draft dropped has q=0 and is not renormed away");
        assert!(p_of(argmax) > 0.02, "the target keeps its argmax");
        assert_eq!(p_of(0), 0.0, "a token the target dropped has p=0");
        assert!(q_of(0) > 0.0, "the draft keeps token 0 through top-p");
        assert!(masses.iter().all(|m| m.token != 150), "a token in neither top-64 is not in the support");
        let p_sum: f32 = masses.iter().map(|m| m.p).sum();
        let q_sum: f32 = masses.iter().map(|m| m.q).sum();
        assert!((p_sum - 1.0).abs() < 1e-5 && (q_sum - 1.0).abs() < 1e-5);
        let p_where_q_zero: f32 = masses.iter().filter(|m| m.q == 0.0).map(|m| m.p).sum();
        assert!(p_where_q_zero > 0.02, "zeros on q do not drop the target-only mass");
        let p_where_q_pos: f32 = masses.iter().filter(|m| m.q > 0.0).map(|m| m.p).sum();
        assert!((p_where_q_zero + p_where_q_pos - 1.0).abs() < 1e-5);

        let n = 30_000u32;
        let mut counts = vec![0u32; target.len()];
        let mut rng = Sampler::draft(TEMP, K, TOP_P, SOFTCAP, 0x51E3_0003);
        for _ in 0..n { counts[rng.reject_one(&target, &draft) as usize] += 1; }
        assert!(counts[argmax as usize] > 0, "the residual draw can emit the target argmax");
        assert_eq!(counts[0], 0, "p=0 is not emitted, including a draft token the target dropped");
        assert_eq!(counts[150], 0, "outside the union is not emitted");
        let mut max_abs = 0.0f32;
        for &(tok, p) in &exp_p {
            let freq = counts[tok as usize] as f32 / n as f32;
            max_abs = max_abs.max((freq - p).abs());
        }
        for (tok, &c) in counts.iter().enumerate() {
            if c > 0 { assert!(exp_p.iter().any(|e| e.0 == tok as u32), "emitted {tok} is not in p"); }
        }
        assert!(max_abs < 0.012, "emitted distribution max |freq-p| {max_abs} over {n} draws");
    }

    #[test]
    fn draft_sampler_mixed_once_per_request_not_reseeded_per_round() {
        let seed = 0xA11C_E5EDu64;
        let mut draft = Sampler::draft(TEMP, K, TOP_P, SOFTCAP, seed);
        let mixed = Sampler::new(TEMP, K, TOP_P, SOFTCAP, seed ^ DRAFT_SEED_MIX);
        assert_eq!(draft.state, mixed.state, "the mix is 0x85ebca6b, once");
        let target = Sampler::new(TEMP, K, TOP_P, SOFTCAP, seed);
        assert_ne!(draft.state, target.state, "the draft stream is not the stream that emits tokens");

        let (tg, dr) = logits();
        let target_state = target.state;
        let _ = draft.reject_one(&tg, &dr);
        assert_eq!(target.state, target_state, "rejection draws do not advance the emit stream");

        // identical logits: p == q, so the draft is accepted and one round emits the drafts plus the bonus
        let row = tg.clone();
        let mut round = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let before = round.state;
        let out = round.verify_round(&[&row, &row, &row], &[&row, &row]);
        assert_eq!(out.len(), 3, "one round may emit more than one token");
        assert_ne!(round.state, before);
        let mut continued = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let _ = continued.verify_round(&[&row, &row, &row], &[&row, &row]);
        let after_first = continued.state;
        let second = continued.verify_round(&[&row, &row, &row], &[&row, &row]);
        let mut reseeded = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let repeated = reseeded.verify_round(&[&row, &row, &row], &[&row, &row]);
        assert_eq!(repeated, out, "the same request seed repeats the first round");
        assert_ne!(second, repeated, "the next round continues; it is not a reseed");
        assert_ne!(continued.state, after_first);
    }
}
