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

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        // top-k on raw logits (softcap is monotonic, so the order is the same)
        let k = self.top_k.min(logits.len());
        self.cand.clear();
        let mut floor = f32::NEG_INFINITY;
        for (i, &x) in logits.iter().enumerate() {
            if x <= floor && self.cand.len() == k { continue; }
            if self.cand.len() < k {
                self.cand.push((x, i as u32));
                if self.cand.len() == k { floor = self.cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min); }
            } else {
                let j = self.cand.iter().position(|c| c.0 == floor).unwrap();
                self.cand[j] = (x, i as u32);
                floor = self.cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
            }
        }
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
}
