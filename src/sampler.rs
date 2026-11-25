//! CPU sampler over the GPU logits: softcap -> top-k -> temperature -> top-p -> draw.

pub(crate) fn checked_argmax(logits: &[f32]) -> Result<u32, &'static str> {
    if logits.is_empty() {
        return Err("model returned no logits");
    }
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, &x) in logits.iter().enumerate() {
        if !x.is_finite() {
            return Err("model returned non-finite logits");
        }
        if x > best.1 {
            best = (i as u32, x);
        }
    }
    Ok(best.0)
}

pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub softcap: f32,
    state: u64,
    cand: Vec<(f32, u32)>,
}

impl Sampler {
    /// `top_k == 0` disables top-k truncation (full-vocab sampling with monotonic `atanh` logit cut).
    pub fn new(temp: f32, top_k: usize, top_p: f32, softcap: f32, seed: u64) -> Sampler {
        Sampler { temp, top_k, top_p, softcap, state: seed ^ 0x9E37_79B9_7F4A_7C15, cand: Vec::with_capacity(256) }
    }

    pub fn greedy(&self) -> bool { self.temp <= 0.0 }

    /// Whether this sampler can use GPU Gumbel-max sampling (`top_k == 0 && top_p >= 1.0`, chadlyb/erisinky).
    pub fn uses_gpu_gumbel(&self) -> bool {
        self.top_k == 0 && self.top_p >= 1.0 && self.temp.is_finite() && self.temp >= 0.0
    }

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
        self.sample_next(logits).expect("model sampling failed")
    }

    /// Sample one model row, or fail the request before consuming an RNG draw.
    pub fn sample_next(&mut self, logits: &[f32]) -> Result<u32, &'static str> {
        self.sample_next_with_ban(logits, None)
    }

    /// Sample one model row with an optional banned token ID (used for hard thinking-budget close),
    /// using a monotonic `atanh` raw-logit cutoff (chadlyb/erisinky) to skip `tanh`/`exp` on tokens
    /// more than 30 nats below the maximum.
    pub fn sample_next_with_ban(&mut self, logits: &[f32], ban: Option<u32>) -> Result<u32, &'static str> {
        self.sample_next_with_ban_and_softcap(logits, ban, self.softcap)
    }

    /// Sample one model row with an optional banned token ID and an explicit per-step `softcap`
    /// override (`0.0` when logits are already post-softcap, e.g. after `k_loopcd_blend`).
    pub fn sample_next_with_ban_and_softcap(
        &mut self,
        logits: &[f32],
        ban: Option<u32>,
        softcap: f32,
    ) -> Result<u32, &'static str> {
        self.sample_next_filtered(logits, ban, softcap, |_| true)
    }

    /// Sample one model row while masking out both `ban` and any token rejected by `allow(token_id)`
    /// (for example, byte-level JSON pushdown grammar constraints during active tool-call spans).
    /// Checks finiteness of all logits first and evaluates `allow` lazily only on record-highs and
    /// candidates within the 30-nat raw-logit cutoff.
    pub fn sample_next_filtered<F: FnMut(u32) -> bool>(
        &mut self,
        logits: &[f32],
        ban: Option<u32>,
        softcap: f32,
        allow: F,
    ) -> Result<u32, &'static str> {
        let total = self.prepare_row_filtered(logits, ban, softcap, allow)?;
        if self.temp < 1e-5 { return Ok(self.cand[0].1); }
        let mut r = self.next_f32() * total;
        let mut last = self.cand[0].1;
        for &(mass, token) in &self.cand {
            if mass == 0.0 { continue; }
            last = token;
            r -= mass;
            if r <= 0.0 { return Ok(token); }
        }
        Ok(last)
    }

    /// Fill `cand` with the masses used for a draw and return their total after the top-p cut.
    /// Both ordinary draws and speculative rejection need the exact same token support.
    fn prepare_row(&mut self, logits: &[f32], ban: Option<u32>, softcap: f32) -> Result<f32, &'static str> {
        self.prepare_row_filtered(logits, ban, softcap, |_| true)
    }

    fn prepare_row_filtered<F: FnMut(u32) -> bool>(
        &mut self,
        logits: &[f32],
        ban: Option<u32>,
        softcap: f32,
        mut allow: F,
    ) -> Result<f32, &'static str> {
        if logits.is_empty() { return Err("model returned no logits"); }
        if !self.temp.is_finite() { return Err("invalid sampling temperature"); }
        let mut best: Option<(f32, u32)> = None;
        for (i, &x) in logits.iter().enumerate() {
            if !x.is_finite() { return Err("model returned non-finite logits"); }
            let tok = i as u32;
            if Some(tok) == ban { continue; }
            if best.is_none_or(|(b, _)| x > b) && allow(tok) {
                best = Some((x, tok));
            }
        }
        let Some((best_val, best_tok)) = best else { return Err("no sampling candidates") };
        self.cand.clear();
        if self.temp < 1e-5 {
            self.cand.push((1.0, best_tok));
            return Ok(1.0);
        }
        if !softcap.is_finite() || !self.top_p.is_finite() {
            return Err("invalid sampling parameters");
        }
        let c = softcap;
        let t = self.temp;
        let capped = |x: f32| if c > 0.0 { c * (x / c).tanh() } else { x };
        let max_cap = capped(best_val);
        let scaled_max = max_cap / t;
        // Monotonic raw-logit cutoff: tokens >30 nats below max carry < e^-30 (< f32::EPSILON) weight.
        let raw_cut = if scaled_max.is_finite() {
            let y = t * (scaled_max - 30.0);
            let rc = if c <= 0.0 {
                y
            } else if y <= -c {
                f32::NEG_INFINITY
            } else {
                c * (y / c).atanh()
            };
            if rc.is_finite() { rc - 1e-3 * rc.abs() - 1e-3 } else { f32::NEG_INFINITY }
        } else {
            f32::NEG_INFINITY
        };
        for (i, &x) in logits.iter().enumerate() {
            let tok = i as u32;
            if Some(tok) != ban && x >= raw_cut && (tok == best_tok || allow(tok)) {
                self.cand.push((x, tok));
            }
        }
        if self.cand.is_empty() {
            self.cand.push((best_val, best_tok));
        }
        truncate_and_sort_top_k(&mut self.cand, self.top_k);
        let mut sum = 0.0;
        for (mass, _) in &mut self.cand {
            let x = capped(*mass);
            let centered = if scaled_max.is_finite() { x / t - scaled_max } else { (x - max_cap) / t };
            *mass = centered.exp();
            sum += *mass;
        }
        if !sum.is_finite() || sum <= 0.0 { return Err("model returned invalid sampling probabilities"); }
        let mut total = 0.0;
        let mut keep = self.cand.len();
        for (i, (mass, _)) in self.cand.iter_mut().enumerate() {
            *mass /= sum;
            total += *mass;
            if total >= self.top_p {
                keep = i + 1;
                break;
            }
        }
        self.cand.truncate(keep);
        Ok(total)
    }

    /// One drafted position. Given the actual drafted token `draft_tok` drawn from `q`, accept with
    /// `min(1, p/q)`, else draw `norm(max(0, p-q))`. Every uniform comes from this sampler.
    /// `temp <= 0` does not belong here.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reject_one(&mut self, draft_tok: u32, target: &[f32], draft: &[f32]) -> u32 {
        self.reject_at(draft_tok, target, draft).0
    }

    /// One verify round on the probabilistic path. `draft_tokens` are the actual drafted tokens
    /// fed into the target forward pass; `target_rows` is those draft positions plus the bonus row.
    /// A rejection stops the round. The bonus draw, when every draft is accepted, uses this same
    /// stream. One call is one round: it is not reseeded per emitted token.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn verify_round(&mut self, draft_tokens: &[u32], target_rows: &[&[f32]], draft_rows: &[&[f32]]) -> Vec<u32> {
        assert!(self.temp > 0.0, "greedy does not reject");
        assert_eq!(draft_tokens.len(), draft_rows.len(), "one draft token per draft row");
        assert_eq!(target_rows.len(), draft_rows.len() + 1, "one target row per draft token, plus the bonus");
        let mut out = Vec::with_capacity(target_rows.len());
        for i in 0..draft_rows.len() {
            let (tok, accepted) = self.reject_at(draft_tokens[i], target_rows[i], draft_rows[i]);
            out.push(tok);
            if !accepted { return out; }
        }
        out.push(self.sample(target_rows[draft_rows.len()]));
        out
    }

    fn reject_at(&mut self, draft_tok: u32, target: &[f32], draft: &[f32]) -> (u32, bool) {
        assert!(self.temp > 0.0, "greedy does not reject");
        let masses = self.masses(target, draft);
        let x = draft_tok;
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
        assert!(!dist.is_empty(), "empty sampling distribution");
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.next_f32() * total;
        for &(tok, p) in dist {
            r -= p;
            if r <= 0.0 { return tok; }
        }
        dist[dist.len() - 1].0
    }

    /// p and q on the union of the surviving target and draft candidates, ordered by token ID.
    fn masses(&mut self, target: &[f32], draft: &[f32]) -> Vec<Mass> {
        let mut by_token = std::collections::BTreeMap::new();
        let target_total = self.prepare_row(target, None, self.softcap).expect("invalid target sampling row");
        for &(weight, token) in &self.cand {
            if weight > 0.0 {
                by_token.insert(token, Mass { token, p: weight / target_total, q: 0.0 });
            }
        }
        let draft_total = self.prepare_row(draft, None, self.softcap).expect("invalid draft sampling row");
        for &(weight, token) in &self.cand {
            if weight > 0.0 {
                by_token.entry(token).or_insert(Mass { token, p: 0.0, q: 0.0 }).q = weight / draft_total;
            }
        }
        by_token.into_values().collect()
    }
}

/// Mixed into the request seed once, for the draft sampler only.
pub const DRAFT_SEED_MIX: u64 = 0x85ebca6b;

struct Mass { token: u32, p: f32, q: f32 }

#[inline]
fn cmp_cand(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
}

fn truncate_and_sort_top_k(cand: &mut Vec<(f32, u32)>, top_k: usize) {
    if top_k > 0 && cand.len() > top_k {
        cand.select_nth_unstable_by(top_k - 1, cmp_cand);
        cand.truncate(top_k);
    }
    cand.sort_unstable_by(cmp_cand);
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

    #[test]
    fn invalid_logits_fail_in_greedy_and_sampled_modes() {
        for temp in [0.0, TEMP] {
            let mut sampler = Sampler::new(temp, 2, TOP_P, SOFTCAP, 1);
            let before = sampler.state;
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                // The invalid logit can be inside or outside the top-k candidates.
                assert!(sampler.sample_next(&[bad, 2.0, 1.0]).is_err());
                assert!(sampler.sample_next(&[2.0, 1.0, bad]).is_err());
            }
            assert!(sampler.sample_next(&[]).is_err());
            assert_eq!(sampler.state, before);
            assert!(sampler.sample_next(&[2.0, 2.0, 1.0]).is_ok());
        }
        let mut greedy = Sampler::new(0.0, 2, TOP_P, SOFTCAP, 1);
        assert_eq!(greedy.sample_next(&[1.0, 2.0, 2.0]), Ok(1));
    }

    #[test]
    fn small_positive_temperature_does_not_overflow_probabilities() {
        let mut sampler = Sampler::new(f32::from_bits(1), 3, 1.0, 0.0, 1);
        assert_eq!(sampler.sample_next(&[1.0, f32::MAX, -f32::MAX]), Ok(1));
        assert_eq!(sampler.sample_next(&[-f32::MAX, -2.0, -1.0]), Ok(2));
    }

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

        let mut sampler = Sampler::new(TEMP, K, TOP_P, SOFTCAP, 1);
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
        let mut draft_rng = Sampler::draft(TEMP, K, TOP_P, SOFTCAP, 0x51E3_0001);
        let mut rng = Sampler::draft(TEMP, K, TOP_P, SOFTCAP, 0x51E3_0003);
        for _ in 0..n {
            let x = draft_rng.sample(&draft);
            counts[rng.reject_one(x, &target, &draft) as usize] += 1;
        }
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
    fn verify_round_evaluates_supplied_draft_tokens_not_cpu_redraw() {
        let (target, draft) = logits();
        // Token 0 has p=0, q>0: must ALWAYS reject regardless of RNG and emit from target residual.
        let mut s1 = Sampler::new(TEMP, K, TOP_P, SOFTCAP, 0x1234);
        let out = s1.verify_round(&[0, 10], &[&target, &target, &target], &[&draft, &draft]);
        assert_eq!(out.len(), 1, "rejection on the first drafted token stops the round");
        assert_ne!(out[0], 0, "p(0)==0 must never be accepted");

        // Identical target & draft: p == q > 0 for drafted tokens 10 and 11, so BOTH exact drafted tokens
        // are accepted without consuming an acceptance uniform, followed by the bonus sample.
        let mut s2 = Sampler::new(TEMP, K, TOP_P, SOFTCAP, 0x5678);
        let out2 = s2.verify_round(&[10, 11], &[&target, &target, &target], &[&target, &target]);
        assert_eq!(out2.len(), 3);
        assert_eq!(&out2[..2], &[10, 11], "accepted tokens must be the exact supplied draft_tokens");
    }

    #[test]
    fn reject_one_respects_the_same_tied_top_k_cutoff_as_the_draft_sampler() {
        let target = [0.0, 0.0, 1.0];
        let draft = [1.0, 1.0, 1.0];
        let mut draw = Sampler::draft(1.0, 2, 1.0, 0.0, 23);
        let mut verify = Sampler::new(1.0, 2, 1.0, 0.0, 43);
        let mut direct = Sampler::new(1.0, 2, 1.0, 0.0, 43);
        let mut seen_verified = [false; 3];
        let mut seen_direct = [false; 3];
        for _ in 0..256 {
            let token = draw.sample(&draft);
            seen_verified[verify.reject_one(token, &target, &draft) as usize] = true;
            seen_direct[direct.sample(&target) as usize] = true;
        }
        assert_eq!(seen_direct, [true, false, true]);
        assert_eq!(seen_verified, seen_direct);
    }

    #[test]
    fn reject_one_uses_first_raw_argmax_at_near_zero_temperature() {
        // All three target logits saturate after softcap. The raw maximum is tied at 0 and 2.
        let target = [61.0, 60.0, 61.0];
        let draft = [0.0, 2.0, 1.0];
        let mut draw = Sampler::draft(1e-6, 3, 1.0, 1.0, 23);
        let mut verify = Sampler::new(1e-6, 3, 1.0, 1.0, 43);
        assert_eq!(draw.sample(&target), 0);
        for _ in 0..64 {
            let token = draw.sample(&draft);
            assert_eq!(token, 1);
            assert_eq!(verify.reject_one(token, &target, &draft), 0);
        }
    }

    #[test]
    fn tiny_temperature_never_panics_in_sample_or_verify_round() {
        let (target, draft) = logits();
        for temp in [1e-37_f32, f32::from_bits(1)] {
            let mut s = Sampler::new(temp, 8, 0.9, SOFTCAP, 7);
            assert_eq!(s.sample_next(&target), Ok(100));
            // draft argmax is 0 (where p=0), so verify_round rejects and draws from target argmax (100) without panicking.
            let out = s.verify_round(&[0], &[&target, &target], &[&draft]);
            assert_eq!(out, vec![100]);
            // identical rows accept the argmax draft token and emit the argmax bonus token.
            let out_same = s.verify_round(&[100], &[&target, &target], &[&target]);
            assert_eq!(out_same, vec![100, 100]);
        }
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
        let _ = draft.reject_one(0, &tg, &dr);
        assert_eq!(target.state, target_state, "rejection draws do not advance the emit stream");

        // identical logits: p == q, so the supplied draft tokens are accepted and one round emits the drafts plus the bonus
        let row = tg.clone();
        let drafted = [100u32, 10];
        let mut round = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let before = round.state;
        let out = round.verify_round(&drafted, &[&row, &row, &row], &[&row, &row]);
        assert_eq!(out.len(), 3, "one round may emit more than one token");
        assert_eq!(&out[..2], &drafted);
        assert_ne!(round.state, before);
        let mut continued = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let _ = continued.verify_round(&drafted, &[&row, &row, &row], &[&row, &row]);
        let after_first = continued.state;
        let second = continued.verify_round(&drafted, &[&row, &row, &row], &[&row, &row]);
        let mut reseeded = Sampler::draft(0.7, 8, 1.0, SOFTCAP, seed);
        let repeated = reseeded.verify_round(&drafted, &[&row, &row, &row], &[&row, &row]);
        assert_eq!(repeated, out, "the same request seed repeats the first round");
        assert_ne!(second, repeated, "the next round continues; it is not a reseed");
        assert_ne!(continued.state, after_first);
    }

    #[test]
    fn atanh_cutoff_and_ban_mask_preserve_distribution_and_support_full_vocab() {
        // Full-vocab (top_k = 0, top_p = 1.0) with softcap and deep tail logits
        let mut logits = vec![-100.0f32; 2048];
        logits[5] = 10.0;
        logits[42] = 9.5;
        logits[99] = 9.0;
        let mut s = Sampler::new(1.0, 0, 1.0, 20.0, 12345);
        assert!(s.uses_gpu_gumbel());
        assert!(!Sampler::new(1.0, 64, 0.95, 20.0, 12345).uses_gpu_gumbel());

        // Banning the top logit (5) must never emit token 5 in greedy or stochastic mode.
        let mut greedy = Sampler::new(0.0, 0, 1.0, 20.0, 1);
        assert_eq!(greedy.sample_next_with_ban(&logits, None), Ok(5));
        assert_eq!(greedy.sample_next_with_ban(&logits, Some(5)), Ok(42));
        for _ in 0..200 {
            let tok = s.sample_next_with_ban(&logits, Some(5)).unwrap();
            assert_ne!(tok, 5, "banned token must never be sampled");
            assert!(matches!(tok, 42 | 99), "atanh cutoff drops -100 tail cleanly, got {tok}");
        }
        // Banning the only candidate fails cleanly.
        assert!(greedy.sample_next_with_ban(&[1.0], Some(0)).is_err());
    }

    #[test]
    fn softcap_override_zero_matches_raw_softcapped_sampling() {
        let raw = [12.0f32, 11.2, 10.5, 4.0, -8.0, -30.0];
        let c = 20.0f32;
        let pre_capped: Vec<f32> = raw.iter().map(|&x| c * (x / c).tanh()).collect();
        let mut s_raw = Sampler::new(0.8, 4, 0.9, c, 0xCAFE_BABE);
        let mut s_pre = Sampler::new(0.8, 4, 0.9, c, 0xCAFE_BABE);
        for _ in 0..100 {
            let t1 = s_raw.sample_next_with_ban(&raw, Some(3)).unwrap();
            let t2 = s_pre.sample_next_with_ban_and_softcap(&pre_capped, Some(3), 0.0).unwrap();
            assert_eq!(t1, t2, "zero softcap override on pre-softcapped logits must match raw softcapped draw");
        }
    }

    #[test]
    fn top_k_selection_matches_full_sort_and_sampling_across_1000_rows() {
        fn full_sort_ref(cand: &[(f32, u32)], top_k: usize) -> Vec<(f32, u32)> {
            let mut out = cand.to_vec();
            out.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            if top_k > 0 && out.len() > top_k {
                out.truncate(top_k);
            }
            out
        }

        fn sample_full_sort_ref(logits: &[f32], temp: f32, top_k: usize, top_p: f32, softcap: f32, state: &mut u64) -> u32 {
            let best_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let best_tok = logits.iter().position(|&x| x == best_val).unwrap() as u32;
            let capped = |x: f32| if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x };
            let max_cap = capped(best_val);
            let scaled_max = max_cap / temp;
            let raw_cut = {
                let y = temp * (scaled_max - 30.0);
                let rc = if softcap <= 0.0 { y } else if y <= -softcap { f32::NEG_INFINITY } else { softcap * (y / softcap).atanh() };
                if rc.is_finite() { rc - 1e-3 * rc.abs() - 1e-3 } else { f32::NEG_INFINITY }
            };
            let mut cand: Vec<(f32, u32)> = logits.iter().enumerate()
                .filter(|&(_, &x)| x >= raw_cut)
                .map(|(i, &x)| (x, i as u32))
                .collect();
            if cand.is_empty() { cand.push((best_val, best_tok)); }
            cand = full_sort_ref(&cand, top_k);
            let mut p: Vec<f32> = cand.iter().map(|&(x, _)| (capped(x) / temp - scaled_max).exp()).collect();
            let sum: f32 = p.iter().sum();
            for v in &mut p { *v /= sum; }
            let mut cum = 0.0;
            let mut n = p.len();
            for (i, &v) in p.iter().enumerate() { cum += v; if cum >= top_p { n = i + 1; break; } }
            let total: f32 = p[..n].iter().sum();
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let mut r = ((z >> 40) as f32 / (1u64 << 24) as f32) * total;
            let mut last = 0;
            for (i, &mass) in p[..n].iter().enumerate() {
                if mass == 0.0 { continue; }
                last = i;
                r -= mass;
                if r <= 0.0 { return cand[i].1; }
            }
            cand[last].1
        }

        let mut rng = 0xD1CE_5EED_1234_5678u64;
        let mut next_u64 = || {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        let ks = [0usize, 1, 2, 7, 16, 64, 128, 300];
        for row_idx in 0..1_000usize {
            let vocab = 64 + (row_idx % 257);
            let mut row = Vec::with_capacity(vocab);
            for i in 0..vocab {
                let u = next_u64();
                let x = if row_idx % 5 == 0 {
                    // Heavy multi-way exact ties right across top_k boundaries (including -0.0 vs +0.0).
                    match u % 8 {
                        0 => -0.0f32,
                        1 => 0.0f32,
                        b => (b as f32) * 0.5,
                    }
                } else {
                    (((u & 0xFFFF) as f32) / 2048.0) - 16.0 + ((i % 7) as f32) * 0.25
                };
                row.push(x);
            }
            let top_k = ks[row_idx % ks.len()];
            let cand_in: Vec<(f32, u32)> = row.iter().enumerate().map(|(i, &x)| (x, i as u32)).collect();
            let mut got = cand_in.clone();
            truncate_and_sort_top_k(&mut got, top_k);
            let want = full_sort_ref(&cand_in, top_k);
            assert_eq!(got.len(), want.len(), "row {row_idx} k={top_k} len");
            for ((gx, gt), (wx, wt)) in got.iter().zip(&want) {
                assert_eq!((gx.to_bits(), *gt), (wx.to_bits(), *wt), "row {row_idx} k={top_k} order mismatch");
            }

            let seed = 0x1000_0000u64 + row_idx as u64;
            let temp = if row_idx % 3 == 0 { 0.7 } else { 1.1 };
            let top_p = if row_idx % 4 == 0 { 1.0 } else { 0.9 };
            let softcap = if row_idx % 2 == 0 { 20.0 } else { 0.0 };
            let mut sampler = Sampler::new(temp, top_k, top_p, softcap, seed);
            let mut ref_state = seed ^ 0x9E37_79B9_7F4A_7C15;
            let got_tok = sampler.sample_next(&row).unwrap();
            let want_tok = sample_full_sort_ref(&row, temp, top_k, top_p, softcap, &mut ref_state);
            assert_eq!(got_tok, want_tok, "row {row_idx} sampled token mismatch");
            assert_eq!(sampler.state, ref_state, "row {row_idx} RNG state mismatch");
        }
    }

    #[test]
    fn checked_sampling_rejects_invalid_rows_without_advancing_rng() {
        let mut sampler = Sampler::new(0.5, 64, 1.0, 0.0, 7);
        let before = sampler.state;
        for row in [
            &[][..],
            &[1.0, f32::NAN],
            &[f32::INFINITY, 2.0],
            &[2.0, f32::NEG_INFINITY],
        ] {
            assert!(checked_argmax(row).is_err());
            assert!(sampler.sample_next(row).is_err());
            assert_eq!(sampler.state, before);
        }
        assert_eq!(checked_argmax(&[0.0, 2.0, 2.0]), Ok(1));
        assert!(sampler.sample_next(&[1.0, 2.0]).is_ok());
    }

    #[test]
    fn speculative_verification_matches_direct_sampling_at_topk_ties_and_controls() {
        const DRAWS: usize = 10_000;
        // 1. Tied top_k=3 boundary: token 1 leads (1.5), tokens 2, 3, 4, 5 tie at 0.9.
        //    Shared `(logit_desc, token_id_asc)` ordering must admit {1, 2, 3} and assign
        //    strictly zero probability to tokens 0, 4, and 5 in BOTH direct sampling and
        //    draft + verify_round, even when the draft model heavily favors tokens 4 and 5!
        let target_tied = [0.2f32, 1.5, 0.9, 0.9, 0.9, 0.9];
        let draft_adversarial = [0.1f32, 0.4, 0.5, 0.6, 2.4, 2.3];
        let mut direct_tied = [0usize; 6];
        let mut spec_tied = [0usize; 6];
        let mut direct_s = Sampler::new(0.8, 3, 1.0, SOFTCAP, 0xA11C_5EED);
        let mut draft_s = Sampler::draft(0.8, 6, 1.0, SOFTCAP, 0xB0BB_5EED);
        let mut verify_s = Sampler::new(0.8, 3, 1.0, SOFTCAP, 0xB0BB_5EED);
        for _ in 0..DRAWS {
            let d = direct_s.sample_next(&target_tied).unwrap() as usize;
            direct_tied[d] += 1;
            let proposed = draft_s.sample_next(&draft_adversarial).unwrap();
            let accepted = verify_s.verify_round(
                &[proposed],
                &[&target_tied, &target_tied],
                &[&draft_adversarial],
            );
            spec_tied[accepted[0] as usize] += 1;
        }
        for excluded in [0usize, 4, 5] {
            assert_eq!(direct_tied[excluded], 0, "direct sampled excluded tied token {excluded}");
            assert_eq!(spec_tied[excluded], 0, "speculative verify sampled excluded tied token {excluded}");
        }
        assert!(direct_tied[2] > 1_000 && direct_tied[3] > 1_000);
        for tok in 0..6 {
            let p_dir = direct_tied[tok] as f64 / DRAWS as f64;
            let p_spec = spec_tied[tok] as f64 / DRAWS as f64;
            assert!(
                (p_dir - p_spec).abs() < 0.025,
                "tied histogram mismatch at token {tok}: direct={p_dir:.4} vs spec={p_spec:.4}"
            );
        }

        // 2. Untied top_k=3 control across 10,000 draws.
        let target_untied = [0.2f32, 1.5, 1.1, 0.8, 0.4, 0.1];
        let mut direct_untied = [0usize; 6];
        let mut spec_untied = [0usize; 6];
        for _ in 0..DRAWS {
            direct_untied[direct_s.sample_next(&target_untied).unwrap() as usize] += 1;
            let proposed = draft_s.sample_next(&draft_adversarial).unwrap();
            let accepted = verify_s.verify_round(
                &[proposed],
                &[&target_untied, &target_untied],
                &[&draft_adversarial],
            );
            spec_untied[accepted[0] as usize] += 1;
        }
        for excluded in [0usize, 4, 5] {
            assert_eq!(direct_untied[excluded], 0);
            assert_eq!(spec_untied[excluded], 0);
        }
        for tok in 0..6 {
            let p_dir = direct_untied[tok] as f64 / DRAWS as f64;
            let p_spec = spec_untied[tok] as f64 / DRAWS as f64;
            assert!(
                (p_dir - p_spec).abs() < 0.025,
                "untied histogram mismatch at token {tok}: direct={p_dir:.4} vs spec={p_spec:.4}"
            );
        }

        // 3. Zero-temperature & near-zero temperature controls on tied and untied rows:
        //    lowest token ID wins deterministically across direct sampling, greedy verify, and near-zero verify_round.
        let mut z_direct = Sampler::new(0.0, 3, 1.0, SOFTCAP, 42);
        let mut nz_verify = Sampler::new(1e-6, 3, 1.0, SOFTCAP, 99);
        for (row, want_winner) in [
            (&[0.0f32, 2.0, 2.0, 2.0][..], 1u32),
            (&[0.0f32, 1.0, 3.0, 2.0][..], 2u32),
        ] {
            assert_eq!(checked_argmax(row), Ok(want_winner));
            assert_eq!(z_direct.sample_next(row), Ok(want_winner));
            for cand_draft in 0..4u32 {
                let g = crate::scheduler::accept_greedy(&[cand_draft], &[want_winner, want_winner], &[]);
                assert_eq!(g.emitted[0], want_winner);
                let out = nz_verify.verify_round(&[cand_draft], &[row, row], &[row]);
                assert_eq!(out[0], want_winner);
            }
        }
    }
}
