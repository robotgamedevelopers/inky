use crate::blocks::{BlockIndex, BlockKind};
use crate::engine::{Engine, Lane, Span};
use crate::metal::CmdBuf;
use crate::{default_model, Sampler, Session, Tokenizer};
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

fn complete(cb: CmdBuf) {
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed, "{:?}", cb.error());
}

fn same_row(actual: &[f32], reference: &[f32]) -> bool {
    actual.len() == reference.len()
        && actual.iter().zip(reference).all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
}

fn differs_from_plain(guided: &[f32], plain: &[f32], cap: f32) -> bool {
    guided.iter().zip(plain).any(|(&a, &b)| (a - cap * (b / cap).tanh()).abs() > 0.1)
}

const OMEGA: f32 = 4.0;

fn load_fixture() -> (Engine, Tokenizer, Vec<u32>) {
    let model = std::env::var("INKY_MODEL").unwrap_or_else(|_| default_model());
    let eng = Engine::load(&model, 128);
    let tk = Tokenizer::load(&eng.g);
    let toks = tk.encode("At the edge of the forest, the traveler found", false);
    assert!(toks.len() > 1 && toks.len() <= crate::blocks::BLOCK);
    eng.ensure_batch(64);
    (eng, tk, toks)
}

#[test]
#[ignore = "needs the Inkling Q4_K_M model and an M5 Metal device"]
fn loopcd_cli_first_token() {
    let (eng, tk, toks) = load_fixture();
    let pos = toks.len() - 1;

    // Reference: evaluate the final prompt token in a standalone ordinary decode after
    // the prior tokens have warmed the lane. The weak pass must leave the strong KV/CS intact.
    let prefix = Lane::new(&eng);
    let run = eng.on(&prefix);
    run.reserve(toks.len() + 1, true);
    run.set_tokens(0, &toks);
    let cb = eng.gpu.cmd();
    run.encode_prefill(&cb, 0, pos, false, false);
    complete(cb);
    let cli_reference = |omega| {
        eng.set_loopcd_omega(omega);
        let lane = Lane::new(&eng);
        let run = eng.on(&lane);
        run.reserve(toks.len() + 1, true);
        eng.copy_lane(&prefix, &lane);
        run.set_tokens(0, &toks);
        let cb = eng.gpu.cmd();
        run.encode_step(&cb, pos, true, true);
        complete(cb);
        (eng.logits().to_vec(), run.token_at(toks.len()), eng.lane_hash(&lane))
    };
    let (plain, _, plain_state) = cli_reference(0.0);
    let (guided, guided_token, guided_state) = cli_reference(OMEGA);
    assert_eq!(guided_state, plain_state, "CLI weak pass changed the strong KV/CS");
    assert!(differs_from_plain(&guided, &plain, eng.softcap));
    eng.set_loopcd_omega(0.0);
    let mut ordinary = Session::new(&eng, &tk);
    let ordinary_first = ordinary.prefill(&toks).unwrap();
    // Find a deterministic draw where changing only the weight changes the first emitted token.
    let sampler = |seed| Sampler::new(1.0, 64, 0.95, eng.softcap, seed);
    let (seed, plain_draw, guided_draw) = (1..=128).find_map(|seed| {
        let plain_draw = sampler(seed).sample_next(&eng.logits()).unwrap();
        let guided_draw = sampler(seed).sample_next_with_ban_and_softcap(&guided, None, 0.0).unwrap();
        (plain_draw != guided_draw && plain_draw != tk.eos && guided_draw != tk.eos)
            .then_some((seed, plain_draw, guided_draw))
    }).expect("fixture should distinguish an actual first output token");
    let draw = |session: &mut Session<'_>, first| {
        let mut out = Vec::new();
        assert_eq!(session.generate(first, 1, &mut sampler(seed), |token| { out.push(token); true }).unwrap(), 1);
        out
    };
    assert_eq!(draw(&mut ordinary, ordinary_first), [plain_draw]);
    eng.set_loopcd_omega(OMEGA);
    let mut session = Session::new(&eng, &tk);
    let first = session.prefill(&toks).unwrap();
    assert_eq!(first, guided_token);
    assert!(same_row(&eng.logits(), &guided), "CLI did not guide its first sample");
    assert_eq!(eng.lane_hash(&session.lane), guided_state);
    assert_eq!(draw(&mut session, first), [guided_draw]);
    eprintln!("first-token reference: seed={seed} plain={plain_draw} guided={guided_draw}");

}

#[test]
#[ignore = "needs the Inkling Q4_K_M model and an M5 Metal device"]
fn loopcd_paged_mixed_lane_state() {
    let (eng, _, toks) = load_fixture();
    let pos = toks.len() - 1;
    // Seed three independent paged cache lanes. Decode two together at different weights
    // and the same plain lane alone. Compare both kinds of physical pages and recurrent state.
    eng.ensure_dbatch(2);
    let paged = || {
        let lane = Lane::new_blocked(&eng, 128);
        {
            let mut pools = eng.block_pool().borrow_mut();
            for kind in [BlockKind::Full, BlockKind::Window] {
                let pool = pools.of_mut(kind);
                let (slot, _) = pool.take_free().unwrap();
                let generation = pool.record(slot).unwrap().generation();
                lane.block_table().borrow_mut().fill(pool, BlockIndex::new(0), slot, generation).unwrap();
            }
        }
        eng.on(&lane).set_tokens(0, &toks);
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0: 0, t: pos }], false, false);
        complete(cb);
        lane
    };
    let solo = paged();
    let disabled = paged();
    let enabled = paged();
    let final_row = |lanes: &[&Lane], weights: &[f32]| {
        let spans: Vec<_> = lanes.iter().map(|lane| Span { lane, p0: pos, t: 1 }).collect();
        let cb = eng.gpu.cmd();
        eng.encode_decode_with_omegas(&cb, &spans, true, false, weights);
        complete(cb);
    };
    final_row(&[&solo], &[0.0]);
    let solo_row = eng.span_logits(0).to_vec();
    final_row(&[&disabled, &enabled], &[0.0, OMEGA]);
    assert!(same_row(&eng.span_logits(0), &solo_row), "LoopCD paged weak pass changed strong logits or state: disabled row");
    let pools = eng.block_pool().borrow();
    for lane in [&disabled, &enabled] {
        assert!(eng.on(lane).get_cs() == eng.on(&solo).get_cs(), "LoopCD paged weak pass changed strong logits or state: recurrent state");
        for kind in [BlockKind::Full, BlockKind::Window] {
            let slot_at = |l: &Lane| l.block_table().borrow().get(kind, BlockIndex::new(0)).unwrap().unwrap().slot();
            let pool = pools.of(kind);
            assert!(pool.slot_bytes(slot_at(lane)).unwrap() == pool.slot_bytes(slot_at(&solo)).unwrap(),
                    "LoopCD paged weak pass changed strong logits or state: {kind:?} cache");
        }
    }
    assert!(differs_from_plain(&eng.span_logits(1), &solo_row, eng.softcap));
    assert_eq!(eng.effective_sampler_softcap(0), eng.softcap);
    assert_eq!(eng.effective_sampler_softcap(1), 0.0);
}

#[test]
#[ignore = "needs the Inkling Q4_K_M model and an M5 Metal device"]
fn loopcd_cached_prompt_first_token_and_mixed_tie_isolation() {
    use crate::poolgate::{Layer3Pool, StepPath};

    // `t-audit-engine-loopcd-first` (both cold & cached PrefixTree start-point prompts!) +
    // `t-audit-engine-loopcd-lanes` (disabled lane raw logits & 1-ULP tie preservation alongside enabled neighbor!).
    let model = std::env::var("INKY_MODEL").unwrap_or_else(|_| default_model());
    let eng = Engine::load(&model, 256);
    eng.ensure_batch(128);
    eng.ensure_dbatch(2);
    let tk = Tokenizer::load(&eng.g);
    let mut prompt = tk.encode("In the ancient library beneath the mountain, the archivist opened a", false);
    while prompt.len() < 72 {
        prompt.extend(tk.encode(" mysterious leather-bound manuscript filled with star maps and", false));
    }
    prompt.truncate(72); // 64-token sealed block 0 + 8-token suffix (final token at pos=71)
    let last_pos = prompt.len() - 1;

    // 1. Cold paged prefill `[0..71)` + terminal decode at `pos = 71` (`omega = 0` vs `omega = OMEGA`).
    let pool = Layer3Pool::with_persistent_starts(&eng, 3);
    // Donor seals block 0 (`[0..64)`) as a PROMPT start point!
    let (donor, m0) = pool.new_lane_matched(&prompt[..64], 256);
    assert_eq!(m0, 0);
    pool.prefill(&[(donor, 0, 64)]);
    pool.end_lane(donor);

    // Cold lane (`Layer3Pool::new` without prefix match) vs Cached lane (`new_lane_matched` hitting block 0 = 64!):
    let cold_pool = Layer3Pool::new(&eng, 1);
    let cold_id = cold_pool.new_lane(&prompt, 256);
    cold_pool.prefill(&[(cold_id, 0, last_pos)]);

    let (cached_plain_id, m_plain) = pool.new_lane_matched(&prompt, 256);
    let (cached_guided_id, m_guided) = pool.new_lane_matched(&prompt, 256);
    assert_eq!(m_plain, 64, "cached prompt must match 64-token start point");
    assert_eq!(m_guided, 64);
    pool.prefill(&[(cached_plain_id, 64, last_pos - 64), (cached_guided_id, 64, last_pos - 64)]);

    // Evaluate cold solo at `omega = 0` and `omega = OMEGA`, then mixed `[0.0, OMEGA]` on the cached lanes!
    cold_pool.decode(&[(cold_id, last_pos)]);
    let cold_plain_logits = cold_pool.logits(0);

    // Run mixed `[0.0, OMEGA]` on the two cached-prefix lanes (`k_attn_grp_dec` + LoopCD first token!).
    let cached_lanes_vec = pool.borrow_lane_ptrs(&[cached_plain_id, cached_guided_id]);
    pool.prepare_decode(&[(cached_plain_id, last_pos), (cached_guided_id, last_pos)]);
    let cb = eng.gpu.cmd();
    eng.encode_decode_with_omegas(
        &cb,
        &[
            Span { lane: unsafe { &*cached_lanes_vec[0] }, p0: last_pos, t: 1 },
            Span { lane: unsafe { &*cached_lanes_vec[1] }, p0: last_pos, t: 1 },
        ],
        true,
        false,
        &[0.0, OMEGA],
    );
    complete(cb);
    let cached_plain_logits = eng.span_logits(0).to_vec();
    let cached_guided_logits = eng.span_logits(1).to_vec();
    assert!(same_row(&cached_plain_logits, &cold_plain_logits), "cached omega=0 first-token logits differ from cold forced decode");
    assert!(differs_from_plain(&cached_guided_logits, &cold_plain_logits, eng.softcap), "cached guided first-token skipped LoopCD");
    assert_eq!(eng.effective_sampler_softcap(0), eng.softcap);
    assert_eq!(eng.effective_sampler_softcap(1), 0.0);

    // Verify any two raw logits separated by 1 ULP near the softcap saturation tail on the disabled lane
    // keep their exact raw bits (whereas premature `tanh` softcapping collapses large raw logits to `softcap`!).
    let mut synthetic_disabled = cached_plain_logits.clone();
    synthetic_disabled[10] = 450.0;
    synthetic_disabled[11] = 460.0; // `30 * tanh(450/30) == 30.0 == 30 * tanh(460/30)` in f32, while raw `460.0 > 450.0`!
    assert_eq!(crate::sampler::checked_argmax(&synthetic_disabled), Ok(11));
    let mut greedy = Sampler::new(0.0, 64, 1.0, eng.softcap, 1);
    assert_eq!(
        greedy.sample_next_with_ban_and_softcap(&synthetic_disabled, None, eng.effective_sampler_softcap(0)),
        Ok(11)
    );
    pool.shutdown();
    cold_pool.shutdown();
}
