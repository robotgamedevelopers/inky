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
