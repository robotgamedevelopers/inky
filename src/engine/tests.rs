use super::model::{check_vocab_lengths, configured_prefill_chunk};
use super::*;

#[test]
fn disk_staging_holds_each_fixed_split_through_a_resize_interval() {
    for max_ctx in [65_536, 131_072, 131_073, 262_145] {
        // Check every 64-position resize interval across the entire 64..=65,536 span,
        // including the 32,760 -> 32,768 -> 32,824 boundary (`t-audit-engine-disk-split-window`).
        for start in (64usize..=65_536).step_by(64) {
            let cache_bytes = (40 * start.min(EXT) + 8 * start) * REC;
            let half = Run::tier_staging(cache_bytes, max_ctx) / 2 / REC;
            for pos in [start, start + 7, start + 8, start + 31, start + 63] {
                if pos >= max_ctx {
                    continue;
                }
                for swa in [true, false] {
                    let (_, kps, _, _) = decode_split_grid(swa, pos, max_ctx);
                    assert!(kps <= half, "position={pos} max_ctx={max_ctx} swa={swa}");
                }
            }
        }
        // Dense per-position check across [32_704..32_896).
        for start in [32_704usize, 32_760, 32_768, 32_824, 32_832] {
            let cache_bytes = (40 * start.min(EXT) + 8 * start) * REC;
            let half = Run::tier_staging(cache_bytes, max_ctx) / 2 / REC;
            for pos in start..start + 64 {
                for swa in [true, false] {
                    let (_, kps, _, _) = decode_split_grid(swa, pos, max_ctx);
                    assert!(kps <= half, "position={pos} max_ctx={max_ctx} swa={swa}");
                }
            }
        }
    }
}

#[test]
fn fixed_attn_batch_fits_every_row_in_the_arena() {
    // Production constants check corresponding to `specs/SplitArena.tla` (`t-audit-pr53-port`).
    for max_ctx in [128usize, 1024, 4096, 32_768, 65_536, 131_072, 262_144] {
        let slots = split_scratch_slots(max_ctx);
        let arena_limit = NH * slots;
        for swa in [true, false] {
            let kps = if swa { 64 } else { 256 };
            for &span_end in &[1usize, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025, 4096, max_ctx] {
                if span_end > max_ctx {
                    continue;
                }
                let (nsplit, rows) = fixed_attn_batch(swa, span_end, max_ctx);
                assert!(rows >= 1, "swa={swa} span_end={span_end} max_ctx={max_ctx}: rows must be >= 1");
                assert!(rows * NH * nsplit <= arena_limit, "writes exceed arena_limit");
                // Check edge-row visibility at `r_end = span_end` and `r_end = 1025` if in range.
                for r_end in [span_end, 1025.min(span_end)] {
                    let first = if swa && r_end > EXT { r_end - EXT } else { 0 };
                    let visible_splits = fixed_split_count(first, r_end, kps);
                    assert!(
                        visible_splits <= nsplit,
                        "swa={swa} r_end={r_end} max_ctx={max_ctx}: visible_splits {visible_splits} > reserved {nsplit}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires local Inkling-2-Mini GGUF on M5"]
fn private_prefill_clears_the_last_window_tile_between_lanes() {
    use crate::poolgate::{Layer2, Layer3Pool, StepPath};

    let model = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| crate::default_model());
    let eng = Engine::load(&model, 128);
    // The short span appends three keys after 62 saved positions. Its final key
    // starts a new 64-key tile, leaving 63 masked rows in that tile.
    let prefix = 62;
    let short_tokens: Vec<_> = (113..178).collect();
    let append = short_tokens.len() - prefix;
    let long_tokens: Vec<_> = (17..84).collect();

    let want = {
        let paged = Layer3Pool::new(&eng, 1);
        let alone = paged.new_lane(&short_tokens, 128);
        paged.prefill(&[(alone, 0, prefix)]);
        paged.prefill(&[(alone, prefix, append)]);
        paged.logits(0)
    };
    assert!(want.iter().all(|x| x.is_finite()));

    // 1. Multi-lane paged candidate (`Layer3Pool` with 2 lanes sharing the prefill step).
    {
        let paged_multi = Layer3Pool::new(&eng, 2);
        let p_long = paged_multi.new_lane(&long_tokens, 128);
        let p_short = paged_multi.new_lane(&short_tokens, 128);
        paged_multi.prefill(&[(p_short, 0, prefix)]);
        paged_multi.prefill(&[(p_long, 0, long_tokens.len()), (p_short, prefix, append)]);
        let got_paged = paged_multi.logits(1);
        assert_eq!(got_paged.len(), want.len());
        assert!(
            got_paged.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "paged multi-lane logits must match the independent solo lane"
        );
    }

    // 2. Multi-lane private Layer 2 candidate (`Layer2` with 2 lanes sharing `sk`/`sv` scratch).
    let private = Layer2::new(&eng, 2);
    let long = private.new_lane(&long_tokens, 128);
    let short = private.new_lane(&short_tokens, 128);
    private.prefill(&[(short, 0, prefix)]);
    private.prefill(&[(long, 0, long_tokens.len()), (short, prefix, append)]);
    let got = private.logits(1);
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "private multi-lane logits must match the independent paged lane");

    // Inspect both K and V across all NKV=8 head groups (serving all NH=24 query heads)
    // after the final private sliding-window layer.
    let batch = eng.ba.get();
    let stride = EXT + batch.cap + 2 * KGUARD;
    let row_bytes = HD * eng.kvb;
    let act = eng.act.borrow();
    let bytes = unsafe { std::slice::from_raw_parts(act.contents().as_ptr().cast::<u8>(), act.length()) };
    for base in [batch.sk, batch.sv] {
        for group in 0..NKV {
            let first = base as usize * 4 + (group * stride + EXT + KGUARD) * row_bytes;
            let live_end = first + append * row_bytes;
            let tile_end = live_end + 63 * row_bytes;
            assert!(bytes[first..live_end].iter().any(|&b| b != 0), "group {group} live rows empty");
            assert!(bytes[live_end..tile_end].iter().all(|&b| b == 0),
                    "group {group}: masked scratch rows still hold the previous private lane's values");
        }
    }
}

#[test]
#[ignore = "requires local Inkling-2-Mini GGUF on M5"]
fn debug_single_token_decode_native_checkpoint() {
    // `t-audit-engine-q8-debug`: single-token decode on the native checkpoint with debug_assertions
    // active (validating `Engine::seg` Q8_0 / Q6_K / Q4_K geometry and finite logits on M5).
    let model = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| crate::default_model());
    let eng = Engine::load(&model, 128);
    let lane = Lane::new(&eng);
    let run = eng.on(&lane);
    run.reserve(16, false);
    run.set_tokens(0, &[1, 42]);
    let cb = eng.gpu.cmd();
    run.encode_step(&cb, 0, true, true);
    cb.commit();
    crate::metal::wait_completed(&cb, "debug decode", 0).unwrap();
    let lg = run.logits();
    assert_eq!(lg.len(), eng.n_vocab);
    assert!(lg.iter().all(|x| x.is_finite()));
    assert!((run.token_at(1) as usize) < eng.n_vocab);
}

#[test]
#[ignore = "requires local Inkling-2-Mini GGUF on M5"]
fn disk_tier_matches_resident_logits_across_32k_staging_resize_transition() {
    // `t-audit-engine-disk-split-window`: prefill 32,756 tokens at ctx=65,536 and compare
    // `--kv-disk` vs resident fp16 decode logits bit-for-bit across positions 32,756..=32,832
    // (straddling 32,760, the 32,768 split growth, and the 32,824 staging resize).
    let model = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| crate::default_model());
    let ctx = 65_536usize;
    let pf_len = 32_756usize;
    let end_pos = 32_832usize;
    let toks: Vec<u32> = (0..=end_pos + 2).map(|i| ((i * 31 + 17) % 2048 + 10) as u32).collect();
    let eng = Engine::load(&model, ctx);
    eng.ensure_batch(eng.chunk);

    let run_trajectory = |use_disk: bool| -> (Vec<u64>, u64) {
        KV_DISK.store(use_disk, std::sync::atomic::Ordering::Relaxed);
        let lane = Lane::new(&eng);
        KV_DISK.store(false, std::sync::atomic::Ordering::Relaxed);
        let r = eng.on(&lane);
        r.reserve(end_pos + 4, false);
        r.set_tokens(0, &toks);
        let mut p0 = 0usize;
        while p0 < pf_len {
            let t = (pf_len - p0).min(eng.chunk);
            let cb = eng.gpu.cmd();
            r.encode_prefill(&cb, p0, t, false, false);
            cb.commit();
            crate::metal::wait_completed(&cb, "32k prefill", p0).unwrap();
            p0 += t;
        }
        let mut hashes = Vec::with_capacity(end_pos - pf_len + 1);
        for pos in pf_len..=end_pos {
            r.gen_reserve(pos);
            r.step(pos, true, false).wait(pos).unwrap();
            let lg = r.logits();
            assert!(lg.iter().all(|x| x.is_finite()), "non-finite logit at pos={pos} disk={use_disk}");
            let h = lg.iter().fold(0xcbf29ce484222325u64, |acc, x| {
                (acc ^ u64::from(x.to_bits())).wrapping_mul(0x100000001b3)
            });
            hashes.push(h);
        }
        let read_bytes = r.tier_io().0;
        (hashes, read_bytes)
    };

    let (res_hashes, res_io) = run_trajectory(false);
    let (dsk_hashes, dsk_io) = run_trajectory(true);
    assert_eq!(res_io, 0);
    assert!(dsk_io > 0, "expected --kv-disk to read cold blocks across 32,756..=32,832");
    for (idx, (&hr, &hd)) in res_hashes.iter().zip(&dsk_hashes).enumerate() {
        let pos = pf_len + idx;
        assert_eq!(
            hd, hr,
            "--kv-disk logit bits diverged from resident cache at pos={pos} ({hd:016x} != {hr:016x})"
        );
    }
}

#[test]
fn projection_segments_preserve_q8_attention_geometry() {
    let weights = QW { ty: T_Q8_0, k: E as u32, rows: (NH * HD) as u32,
        row_bytes: (E / 32 * 34) as u32, ..Default::default() };
    let seg = Engine::seg(&weights, 12, 34);
    assert_eq!((seg.wtype, seg.k, seg.nrows, seg.row_bytes),
               (T_Q8_0, weights.k, weights.rows, weights.row_bytes));
}

#[test]
fn prefill_chunk_requires_a_supported_positive_size() {
    assert_eq!(configured_prefill_chunk(Err(std::env::VarError::NotPresent), 16384), Ok(4096));
    assert_eq!(configured_prefill_chunk(Ok("4096".into()), 128), Ok(128));
    assert_eq!(configured_prefill_chunk(Ok("32".into()), 128), Ok(32));
    assert_eq!(configured_prefill_chunk(Ok("96".into()), 128), Ok(96));
    for bad in ["0", "1", "19", "4097", "4128", "-32", "invalid", ""] {
        assert!(configured_prefill_chunk(Ok(bad.into()), 128).is_err(), "{bad:?}");
    }
}

#[test]
fn token_writes_validate_the_whole_range_and_vocabulary() {
    assert!(validate_token_write(0, &[0, 1, 2, 3], 4, 4).is_ok());
    assert!(validate_token_write(3, &[3], 4, 4).is_ok());
    assert!(validate_token_write(4, &[], 4, 4).is_ok());
    assert!(validate_token_write(0, &[0; 5], 4, 4).is_err());
    assert!(validate_token_write(usize::MAX, &[0], 4, 4).is_err());
    assert!(validate_token_write(2, &[0, 4], 4, 4).is_err());
    assert!(validate_token_write(0, &[u32::MAX], 4, 4).is_err());
}

#[test]
fn vocabulary_must_fit_the_logits_rows_and_match_the_tokenizer() {
    assert_eq!(check_vocab_lengths(200_058, N_VOCAB_PAD, 200_058, 200_058, 200_006).unwrap(), 200_058);
    assert!(check_vocab_lengths(32, 31, 32, 32, 0).is_err());
    assert!(check_vocab_lengths(0, 32, 0, 0, 0).is_err());
    assert!(check_vocab_lengths(31, 32, 32, 32, 0).is_err());
    assert!(check_vocab_lengths(32, 32, 32, 31, 0).is_err());
    assert!(check_vocab_lengths(32, 32, 32, 32, 32).is_err());
}

#[test]
#[ignore = "requires Metal 4"]
fn router_handles_nonfinite_scores() {
    let src = format!("{}\n{}", include_str!("../kernels.metal"), include_str!("../prefill.metal"));
    let gpu = Gpu::new(&src, &["k_topk", "k_topk_b", "k_moe_topk_route3"]);
    let b = BArena::new(0, 64, true, 2, 256);
    let pb = gpu.buffer(NEXP * 4);
    let act = gpu.buffer(b.end as usize * 4);

    for (logit_val, bias_val, want_w) in [
        (f32::NAN, 0.0f32, 0.0f32),
        (f32::INFINITY, 0.0f32, 1.0f32 / TOPK as f32),
        (f32::NEG_INFINITY, 0.0f32, 0.0f32),
        (0.0f32, f32::INFINITY, 0.0f32),
    ] {
        unsafe {
            let bias = std::slice::from_raw_parts_mut(pb.contents().as_ptr().cast::<f32>(), NEXP);
            let scores = std::slice::from_raw_parts_mut(
                act.contents().as_ptr().cast::<f32>().add(b.rlog as usize),
                NEXP,
            );
            let sel = std::slice::from_raw_parts_mut(
                act.contents().as_ptr().cast::<f32>().add(b.sel as usize),
                2 * TOPK,
            );
            bias.fill(bias_val);
            scores.fill(logit_val);
            sel.fill(-1.0);
        }

        // 1. Standalone batched top-k (`k_topk_b` via real `TopkBArgs`).
        let ta = TopkBArgs {
            logits_off: b.rlog,
            bias_off: 0,
            sel_off: b.sel,
            t: 1,
            wscale: 1.0,
            pad: [0; 3],
        };
        let cb = gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&pb, 0, 1);
        e.buf(&act, 0, 2);
        e.bytes(&ta, 5);
        e.pso(&gpu.pso("k_topk_b"));
        e.dispatch((1, 1, 1), 32);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
        Engine::assert_cmd_ok(&cb, "router_handles_nonfinite_scores(k_topk_b)");
        let out = unsafe {
            std::slice::from_raw_parts(act.contents().as_ptr().cast::<f32>().add(b.sel as usize), 2 * TOPK)
        };
        for (i, &id) in out[..TOPK].iter().enumerate() {
            assert_eq!(id, i as f32, "k_topk_b id[{i}] on ({logit_val}, {bias_val})");
        }
        assert_eq!(out[TOPK..], [want_w; TOPK]);

        // 2. Fused 1-TG router (`k_moe_topk_route3` via real `TopkRoute3Args`).
        unsafe {
            std::slice::from_raw_parts_mut(act.contents().as_ptr().cast::<f32>().add(b.sel as usize), 2 * TOPK)
                .fill(-1.0);
        }
        let tra = TopkRoute3Args {
            logits_off: b.rlog,
            bias_off: 0,
            sel_off: b.sel,
            t: 1,
            wscale: 1.0,
            perm_off: b.perm,
            inv_off: b.inv,
            aw_off: b.aw,
            ind_off: b.ind,
            gu_y: (FF / 32) as u32,
            dn_y: (E / 64) as u32,
            tiles_off: b.tl,
            pad: [0; 3],
        };
        let cb2 = gpu.cmd();
        let e2 = Enc::new(&cb2);
        e2.buf(&pb, 0, 1);
        e2.buf(&act, 0, 2);
        e2.bytes(&tra, 5);
        e2.pso(&gpu.pso("k_moe_topk_route3"));
        e2.dispatch((1, 1, 1), 1024);
        e2.end();
        cb2.commit();
        cb2.waitUntilCompleted();
        Engine::assert_cmd_ok(&cb2, "router_handles_nonfinite_scores(k_moe_topk_route3)");
        let out2 = unsafe {
            std::slice::from_raw_parts(act.contents().as_ptr().cast::<f32>().add(b.sel as usize), 2 * TOPK)
        };
        for (i, &id) in out2[..TOPK].iter().enumerate() {
            assert_eq!(id, i as f32, "k_moe_topk_route3 id[{i}] on ({logit_val}, {bias_val})");
        }
        assert_eq!(out2[TOPK..], [want_w; TOPK]);
        let ind = unsafe {
            std::slice::from_raw_parts(act.contents().as_ptr().cast::<u32>().add(b.ind as usize), 8)
        };
        assert_eq!(ind[0], 0, "class 0 tile count must be 0 for t=1");
        assert_eq!(ind[4], TOPK as u32, "class 1 must emit 1 tile per distinct expert");
    }
}

#[test]
#[ignore = "requires Metal 4"]
fn k_attn_b_and_blk_are_split_invariant_across_sliding_window() {
    use crate::blocks::{key_row, BlockIndex, BlockKind, BlockPools, BlockTable, Kv, BLOCK as KV_BLOCK, CLEARED_GPU_TABLE_ENTRY};

    let src = format!("{}\n{}", include_str!("../kernels.metal"), include_str!("../prefill.metal"));
    let gpu = Gpu::new(&src, &["k_attn_b", "k_attn_b_blk"]);

    let total = 1100usize; // > EXT (1024), crossing multiple 64-token blocks and expired window block 0
    let n_blks = total.div_ceil(KV_BLOCK);
    let f16 = |v: f32| -> u16 {
        let bits = v.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        if v == 0.0 || exp <= 0 {
            sign as u16
        } else {
            (sign | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
        }
    };

    let pb = gpu.buffer(EXT * NREL * 4 + 256);
    let qp_len = total * NH * HD;
    let rp_len = total * NH * NREL;
    let sink_len = total * NH;
    let gate_len = total * NH * HD;
    let out_len = total * NH * HD;
    let qp_off = 0usize;
    let rp_off = qp_off + qp_len;
    let sink_off = rp_off + rp_len;
    let gate_off = sink_off + sink_len;
    let out_unsplit_off = gate_off + gate_len;
    let out_split_off = out_unsplit_off + out_len;
    let act = gpu.buffer((out_split_off + out_len) * 4);
    let kv_lin = gpu.buffer(2 * NKV * total * HD * 2);

    let mut pools = BlockPools::gpu(&gpu.dev, n_blks, 1, 2);
    let mut table = BlockTable::new(n_blks as u32).unwrap();
    for b in 0..n_blks {
        let (slot, _) = pools.window.take_free().unwrap();
        let gen = pools.window.record(slot).unwrap().generation();
        table.fill(&mut pools.window, BlockIndex::new(b as u32), slot, gen).unwrap();
    }
    let bt_buf = gpu.buffer(n_blks * 8);
    let mut host_bt = vec![CLEARED_GPU_TABLE_ENTRY; n_blks];
    table.sync_dirty_to_slice(BlockKind::Window, &pools.window, &mut host_bt).unwrap();
    assert_eq!(host_bt[0], 0, "Slot(1) of slab 0 encodes as 0, distinct from CLEARED_GPU_TABLE_ENTRY");
    assert!(host_bt.iter().all(|&w| w != CLEARED_GPU_TABLE_ENTRY));

    let mut st = 0x1234_5678_9abc_def0u64;
    let mut rnd = || -> f32 {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 1.6 - 0.8
    };

    unsafe {
        let pb_f = std::slice::from_raw_parts_mut(pb.contents().as_ptr().cast::<f32>(), EXT * NREL);
        for x in pb_f.iter_mut() { *x = rnd() * 0.25; }

        let a_f = std::slice::from_raw_parts_mut(
            act.contents().as_ptr().cast::<f32>(),
            out_split_off + out_len,
        );
        for slot in a_f.iter_mut().take(gate_off + gate_len) { *slot = rnd(); }

        let kv_h = std::slice::from_raw_parts_mut(
            kv_lin.contents().as_ptr().cast::<u16>(),
            2 * NKV * total * HD,
        );
        let slabs = pools.window.gpu_slabs();
        for p in 0..total {
            let blk = p / KV_BLOCK;
            let in_blk = (p % KV_BLOCK) as u32;
            let word = host_bt[blk];
            let slab_idx = (word >> 62) as usize;
            let slab_off = (word & ((1u64 << 62) - 1)) as usize;
            let slab_base = slabs[slab_idx].contents().as_ptr() as *mut u8;
            for g in 0..NKV {
                let k_blk_off = key_row(0, BlockKind::Window, 2, 0, Kv::Key, g as u32, in_blk).unwrap() as usize;
                let v_blk_off = key_row(0, BlockKind::Window, 2, 0, Kv::Value, g as u32, in_blk).unwrap() as usize;
                let k_dst = slab_base.add(slab_off + k_blk_off) as *mut u16;
                let v_dst = slab_base.add(slab_off + v_blk_off) as *mut u16;
                for d in 0..HD {
                    let hk = f16(rnd());
                    let hv = f16(rnd());
                    kv_h[(g * total + p) * HD + d] = hk;
                    kv_h[(NKV * total + g * total + p) * HD + d] = hv;
                    *k_dst.add(d) = hk;
                    *v_dst.add(d) = hv;
                }
            }
        }
    }

    let run_linear = |ranges: &[(usize, usize)], out_base: usize| {
        unsafe {
            let a_f = std::slice::from_raw_parts_mut(
                act.contents().as_ptr().cast::<f32>().add(out_base),
                out_len,
            );
            a_f.fill(f32::NAN);
        }
        for &(p0, chunk_t) in ranges {
            let aa = AttnBArgs {
                t: chunk_t as u32,
                p0: p0 as u32,
                ring: 0,
                cap: total as u32,
                k_base: 0,
                v_base: (NKV * total * HD) as u32,
                window: EXT as u32,
                qp_off: (qp_off + p0 * NH * HD) as u32,
                rp_off: (rp_off + p0 * NH * NREL) as u32,
                sink_off: (sink_off + p0 * NH) as u32,
                gate_off: (gate_off + p0 * NH * HD) as u32,
                out_off: (out_base + p0 * NH * HD) as u32,
                relbias_w: 0,
                ghalf: 0,
                kpos0: 0,
                pad: 0,
            };
            let cb = gpu.cmd();
            let e = Enc::new(&cb);
            e.buf(&pb, 0, 1);
            e.buf(&act, 0, 2);
            e.buf(&kv_lin, 0, 3);
            e.bytes(&aa, 5);
            e.pso(&gpu.pso("k_attn_b"));
            e.dispatch((chunk_t.div_ceil(8), NKV, 1), 128);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            Engine::assert_cmd_ok(&cb, "k_attn_b");
        }
    };

    let run_blocked = |ranges: &[(usize, usize)], out_base: usize, clear_expired: bool| {
        unsafe {
            let a_f = std::slice::from_raw_parts_mut(
                act.contents().as_ptr().cast::<f32>().add(out_base),
                out_len,
            );
            a_f.fill(f32::NAN);
        }
        let slabs = pools.window.gpu_slabs();
        for &(p0, chunk_t) in ranges {
            let lo_win_blk = p0.saturating_sub(EXT - 1) / KV_BLOCK;
            unsafe {
                let bt_dst = std::slice::from_raw_parts_mut(bt_buf.contents().as_ptr().cast::<u64>(), n_blks);
                bt_dst.copy_from_slice(&host_bt);
                if clear_expired {
                    bt_dst[..lo_win_blk].fill(CLEARED_GPU_TABLE_ENTRY);
                }
            }
            let aa = AttnBArgs {
                t: chunk_t as u32,
                p0: p0 as u32,
                ring: 0,
                cap: 0,
                k_base: 0,
                v_base: 0,
                window: EXT as u32,
                qp_off: (qp_off + p0 * NH * HD) as u32,
                rp_off: (rp_off + p0 * NH * NREL) as u32,
                sink_off: (sink_off + p0 * NH) as u32,
                gate_off: (gate_off + p0 * NH * HD) as u32,
                out_off: (out_base + p0 * NH * HD) as u32,
                relbias_w: 0,
                ghalf: 0,
                kpos0: 0,
                pad: 0,
            };
            let cb = gpu.cmd();
            let e = Enc::new(&cb);
            e.buf(&pb, 0, 1);
            e.buf(&act, 0, 2);
            e.buf(slabs[0], 0, 3);
            e.bytes(&aa, 5);
            e.buf(&bt_buf, 0, 8);
            e.buf(slabs.get(1).copied().unwrap_or(slabs[0]), 0, 10);
            e.buf(slabs.get(2).copied().unwrap_or(slabs[0]), 0, 11);
            e.buf(slabs.get(3).copied().unwrap_or(slabs[0]), 0, 12);
            e.pso(&gpu.pso("k_attn_b_blk"));
            e.dispatch((chunk_t.div_ceil(8), NKV, 1), 128);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            Engine::assert_cmd_ok(&cb, "k_attn_b_blk");
        }
    };

    let assert_bits_eq = |label: &str| {
        let (u, s) = unsafe {
            let base = act.contents().as_ptr().cast::<f32>();
            (
                std::slice::from_raw_parts(base.add(out_unsplit_off), out_len),
                std::slice::from_raw_parts(base.add(out_split_off), out_len),
            )
        };
        for (idx, (&a, &b)) in u.iter().zip(s.iter()).enumerate() {
            assert!(a.is_finite() && b.is_finite(), "{label}: non-finite output at {idx} ({a}, {b})");
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}: bit mismatch at flat={idx} (pos={}, elem={}): unsplit={a} ({:#010x}) vs split={b} ({:#010x})",
                idx / (NH * HD),
                idx % (NH * HD),
                a.to_bits(),
                b.to_bits(),
            );
        }
    };

    // Non-32-aligned splits across the 1024-token window + expired block-0 tail [1089..1100).
    let splits_a = [(0, 23), (23, 1032), (1055, 34), (1089, 11)];
    let splits_b = [(0, 17), (17, 514), (531, 494), (1025, 62), (1087, 13)];

    run_linear(&[(0, total)], out_unsplit_off);
    run_linear(&splits_a, out_split_off);
    assert_bits_eq("k_attn_b unsplit vs splits_a");
    run_linear(&splits_b, out_split_off);
    assert_bits_eq("k_attn_b unsplit vs splits_b");

    // Blocked `k_attn_b_blk` must also match `k_attn_b` unsplit AND stay bit-identical when expired
    // window blocks (`0..lo_win_blk`) are marked `CLEARED_GPU_TABLE_ENTRY` (`u64::MAX`).
    run_blocked(&[(0, total)], out_split_off, false);
    assert_bits_eq("k_attn_b vs k_attn_b_blk unsplit");
    run_blocked(&splits_a, out_split_off, true);
    assert_bits_eq("k_attn_b vs k_attn_b_blk splits_a (expired block cleared)");
    run_blocked(&splits_b, out_split_off, true);
    assert_bits_eq("k_attn_b vs k_attn_b_blk splits_b (expired block cleared)");
}

/// Chunk 4096 keeps `min(t, MAX_SPANS)` vocabulary rows. A verify span of k=3 needs 4 of them.
/// Growing `lg` to one row per token would be 32× those rows; this layout does not.
/// fp16 path, fp16 K/V, vocab pad 201024 (`output.weight` rows of Inkling-2-Mini).
#[test]
fn chunk_4096_batch_arena_is_unchanged_for_verify_rows() {
    let b = BArena::new(0, 4096, true, 2, N_VOCAB_PAD);
    assert_eq!(b.lg_rows, 128);
    assert_eq!(b.cap, 4096);
    let bytes = b.end as usize * 4;
    assert_eq!(bytes, 1_791_665_152);
    assert!(4 <= b.lg_rows as usize, "k=3 is 4 rows and must fit the existing lg");
    assert!(b.lg_rows as usize * 32 == 4096);
}

/// One image is N_SLOT × NCH × 3 floats. At k=3 the span holds 4 of them (the page's 14.64 MiB); at k=7, 8 (29.28 MiB).
#[test]
fn verify_cs_image_is_one_convolution_state() {
    let one = N_SLOT * NCH * 3 * 4;
    assert_eq!(one, 3_833_856);
    assert_eq!(one * 4, 15_335_424);
    assert_eq!(one * 8, 30_670_848);
    assert_eq!(Engine::cs_image_bytes(), one);
}

#[test]
fn shader_arg_structs_match_metal_abi_sizes() {
    assert_eq!(std::mem::size_of::<EmbedArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<DecodeSpanDesc>(), 16 * 4);
    assert_eq!(std::mem::align_of::<DecodeSpanDesc>(), 4);
    assert_eq!(std::mem::size_of::<EmbedDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<CombineDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<DecAttnGroup>(), 6 * 4);
    assert_eq!(std::mem::align_of::<DecAttnGroup>(), 4);
    assert_eq!(std::mem::size_of::<KvPrepDecArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<AttnGrpDecArgs>(), 12 * 4);
    assert_eq!(std::mem::size_of::<PostConvDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<MixJob>(), 50 * 4);
    assert_eq!(std::mem::size_of::<Seg>(), 20 * 4);
    assert_eq!(std::mem::size_of::<MvArgs>(), (8 + 12 * 20) * 4);
    assert_eq!(std::mem::size_of::<CtrlJob>(), 10 * 4);
    assert_eq!(std::mem::size_of::<CtrlArgs>(), (4 + 4 * 10) * 4);
    assert_eq!(std::mem::size_of::<AttnArgs>(), 34 * 4);
    assert_eq!(std::mem::size_of::<DevCopyArgs>(), 4 * 4);
    assert_eq!(std::mem::size_of::<CombineArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<EpiArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<MoeArgs>(), 32 * 4);
    assert_eq!(std::mem::size_of::<GSeg>(), 12 * 4);
    assert_eq!(std::mem::size_of::<ToHalfArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<HSeg>(), 12 * 4);
    assert_eq!(std::mem::size_of::<GemmHArgs>(), (4 + 10 * 12) * 4);
    assert_eq!(std::mem::size_of::<MoeGemmHArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<CtrlBJob>(), 8 * 4);
    assert_eq!(std::mem::size_of::<CtrlGfJob>(), 8 * 4);
    assert_eq!(std::mem::size_of::<CtrlGfArgs>(), (4 + 4 * 8) * 4);
    assert_eq!(std::mem::size_of::<CtrlBArgs>(), (4 + 4 * 8) * 4);
    assert_eq!(std::mem::size_of::<GemmArgs>(), (4 + 10 * 12) * 4);
    assert_eq!(std::mem::size_of::<MoeGemmArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<TopkBArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<RouteArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<Route3Args>(), 12 * 4);
    assert_eq!(std::mem::size_of::<TopkRoute3Args>(), 16 * 4);
    assert_eq!(std::mem::size_of::<MoeCombArgs>(), 4 * 4);
    assert_eq!(std::mem::size_of::<ConvBArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<StateArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<StateKeepArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<KvPrepArgs>(), 24 * 4);
    assert_eq!(std::mem::size_of::<AttnBArgs>(), 16 * 4);
    assert_eq!(std::mem::size_of::<KvCopyArgs>(), 12 * 4);
    assert_eq!(std::mem::size_of::<ArgmaxArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<SampArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<LoopCdArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<BenchArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<TouchArgs>(), 4 * 4);

    // Verify exact field offsets of `DecodeSpanDesc` match `kernels.metal`.
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, tok_id), 0);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, pos), 4);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, cs_base), 8);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, bt_full_base), 12);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, bt_win_base), 16);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, part_off), 20);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, swa_nkeys), 24);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, swa_first_pos), 28);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, swa_kps), 32);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, swa_nsplit), 36);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, full_nkeys), 40);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, full_kps), 44);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, full_nsplit), 48);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, full_tau), 52);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, swa_tau), 56);
    assert_eq!(std::mem::offset_of!(DecodeSpanDesc, pad1), 60);

    // Verify key field offsets across decode, prefill, sampling, and bench structs match Metal declarations.
    assert_eq!(std::mem::offset_of!(CombineArgs, part_off), 0);
    assert_eq!(std::mem::offset_of!(CombineArgs, nsplit), 4);
    assert_eq!(std::mem::offset_of!(CombineArgs, gate_off), 8);
    assert_eq!(std::mem::offset_of!(CombineArgs, out_off), 12);
    assert_eq!(std::mem::offset_of!(CombineArgs, hout), 16);
    assert_eq!(std::mem::offset_of!(CombineArgs, pad), 20);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, desc_off), 0);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, gate_off), 4);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, out_off), 8);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, swa), 12);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, hout), 16);
    assert_eq!(std::mem::offset_of!(CombineDecArgs, pad), 20);
    assert_eq!(std::mem::offset_of!(DecAttnGroup, abs_sp), 0);
    assert_eq!(std::mem::offset_of!(DecAttnGroup, count), 4);
    assert_eq!(std::mem::offset_of!(DecAttnGroup, span_idx), 8);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, desc_off), 0);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, qp_off), 32);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, rp_off), 36);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, kv_cur_off), 40);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, kdyn_off), 44);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, vdyn_off), 48);
    assert_eq!(std::mem::offset_of!(KvPrepDecArgs, relclip_w), 64);
    assert_eq!(std::mem::offset_of!(AttnGrpDecArgs, desc_off), 0);
    assert_eq!(std::mem::offset_of!(AttnGrpDecArgs, grp_off), 4);
    assert_eq!(std::mem::offset_of!(AttnGrpDecArgs, kv_cur_off), 16);
    assert_eq!(std::mem::offset_of!(AttnGrpDecArgs, kps), 32);
    assert_eq!(std::mem::offset_of!(AttnGrpDecArgs, swa), 36);
    assert_eq!(std::mem::offset_of!(PostConvDecArgs, desc_off), 0);
    assert_eq!(std::mem::offset_of!(PostConvDecArgs, cs_slot_off), 16);
    assert_eq!(std::mem::offset_of!(PostConvDecArgs, has_res), 24);
    assert_eq!(std::mem::offset_of!(PostConvDecArgs, no_state_write), 28);
    assert_eq!(std::mem::offset_of!(MoeArgs, no_state_write), 30 * 4);
    assert_eq!(std::mem::offset_of!(LoopCdArgs, logits_off), 0);
    assert_eq!(std::mem::offset_of!(LoopCdArgs, n), 4);
    assert_eq!(std::mem::offset_of!(LoopCdArgs, stride), 8);
    assert_eq!(std::mem::offset_of!(LoopCdArgs, softcap), 16);
    assert_eq!(std::mem::offset_of!(TopkBArgs, wscale), 16);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, logits_off), 0);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, bias_off), 4);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, sel_off), 8);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, t), 12);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, wscale), 16);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, perm_off), 20);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, inv_off), 24);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, aw_off), 28);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, ind_off), 32);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, gu_y), 36);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, dn_y), 40);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, tiles_off), 44);
    assert_eq!(std::mem::offset_of!(TopkRoute3Args, pad), 52);
    assert_eq!(std::mem::offset_of!(AttnBArgs, window), 24);
    assert_eq!(std::mem::offset_of!(AttnBArgs, kpos0), 56);
    assert_eq!(std::mem::offset_of!(BenchArgs, x_off), 20);
    assert_eq!(std::mem::offset_of!(TouchArgs, n16), 8);
}

#[test]
fn blocked_slab_geometry_and_decode_scratch_invariants() {
    assert_eq!(CS_STRIDE_BYTES, 3_833_856);
    assert_eq!(CS_STRIDE_BYTES % 256, 0);
    assert_eq!(BT_STRIDE_ALIGN_U64 * 8, 256);
    assert_eq!(BlockedSlab::next_cap(1), 1);
    assert_eq!(BlockedSlab::next_cap(2), 2);
    assert_eq!(BlockedSlab::next_cap(3), 4);
    assert_eq!(BlockedSlab::next_cap(8), 8);
    assert_eq!(BlockedSlab::next_cap(9), 16);
    assert_eq!(BlockedSlab::next_cap(32), 32);

    let slab = BlockedSlab::new(2048);
    assert_eq!(slab.bt_stride_u64, 2048);
    assert_eq!(slab.cs_f32_base(0), 0);
    assert_eq!(slab.cs_f32_base(31) as usize, 31 * CS_STRIDE_F32);
    assert_eq!(slab.bt_u64_base(0, false), 0);
    assert_eq!(slab.bt_u64_base(0, true), 2048);
    assert_eq!(slab.bt_u64_base(5, false), 10 * 2048);
    assert_eq!(slab.bt_u64_base(5, true), 11 * 2048);
    assert_eq!((slab.bt_u64_base(5, true) * 8) % 256, 0);

    let ar = Arena::new(MAX_SPLIT, N_VOCAB_PAD);
    let max_desc_floats = MAX_SPANS * (std::mem::size_of::<DecodeSpanDesc>() / 4);
    assert!(max_desc_floats <= ar.part as usize, "DecodeSpanDesc[{MAX_SPANS}] must fit before ar.part");
    // Batched Gumbel argmax1 partials use 512 floats per span inside `ar.part`.
    const { assert!(MAX_SPANS * 512 <= NH * MAX_SPLIT * (HD + 2)) };

    // Verify Layer 3 batched-decode extra scratch bounds for all batch sizes 1..=MAX_SPANS.
    let align64 = |x: usize| x.div_ceil(64) * 64;
    let grp_floats = std::mem::size_of::<DecAttnGroup>() / 4;
    for n in [1usize, 2, 8, 16, 32, 64, MAX_SPANS] {
        let extra = decode_batch_extra_floats(n);
        let desc = align64(n * (std::mem::size_of::<DecodeSpanDesc>() / 4));
        let kv_cur = align64(n * 2 * NKV * HD);
        let swa_g = align64(n * MAX_SWA_SPLIT * grp_floats);
        let full_g = align64(n * MAX_SPLIT * grp_floats);
        assert_eq!(extra, desc + kv_cur + swa_g + full_g);
        assert_eq!(extra % 64, 0);
        if n > 1 {
            assert!(extra >= decode_batch_extra_floats(n - 1));
        }
        // Must never overlap any lane's `ar.part` when placed at `dstart + n * ar.total`.
        let lane_end = (n - 1) * ar.total as usize + ar.part as usize + NH * MAX_SPLIT * (HD + 2);
        assert!(lane_end <= n * ar.total as usize);
    }
}

#[test]
fn fixed_decode_split_grid_boundaries_and_max_ctx_scaling() {
    // SWA always anchors at 64-token block boundaries, <= 17 splits <= MAX_SPLIT.
    assert_eq!(decode_split_kps(true, 4096), 64);
    assert_eq!(decode_split_kps(true, 262_144), 64);
    // Full anchors at 512 for max_ctx <= 131,072 and scales by ceil(max_ctx / 131_072) beyond.
    assert_eq!(decode_split_kps(false, 4096), 512);
    assert_eq!(decode_split_kps(false, 131_072), 512);
    assert_eq!(decode_split_kps(false, 131_136), 1024);
    assert_eq!(decode_split_kps(false, 262_144), 1024);

    for &pos in &[0usize, 1, 62, 63, 64, 65, 510, 511, 512, 513, 1022, 1023, 1024, 1025, 1086, 1087, 1088, 16_383, 16_384, 131_071] {
        for &swa in &[true, false] {
            let (first_pos, kps, sp_lo, nsplit) = decode_split_grid(swa, pos, 131_072);
            assert!((1..=MAX_SPLIT).contains(&nsplit));
            if swa {
                assert!(nsplit <= MAX_SWA_SPLIT);
                assert_eq!(first_pos, (pos + 1).saturating_sub(EXT));
            } else {
                assert_eq!(first_pos, 0);
                assert_eq!(sp_lo, 0);
            }
            // Clipped splits partition `[first_pos, pos + 1)` exactly with no empty split and no gap.
            let mut covered = first_pos;
            for s in 0..nsplit {
                let abs_sp = sp_lo + s;
                let lo = (abs_sp * kps).max(first_pos);
                let hi = ((abs_sp + 1) * kps).min(pos + 1);
                assert!(lo < hi, "empty split at pos={pos} swa={swa} s={s}");
                assert_eq!(lo, covered);
                covered = hi;
            }
            assert_eq!(covered, pos + 1);
        }
    }
    // Specific boundary checks
    assert_eq!(decode_split_grid(true, 63, 131_072), (0, 64, 0, 1));
    assert_eq!(decode_split_grid(true, 64, 131_072), (0, 64, 0, 2));
    assert_eq!(decode_split_grid(true, 1023, 131_072), (0, 64, 0, 16));
    assert_eq!(decode_split_grid(true, 1024, 131_072), (1, 64, 0, 17));
    assert_eq!(decode_split_grid(true, 1086, 131_072), (63, 64, 0, 17));
    assert_eq!(decode_split_grid(true, 1087, 131_072), (64, 64, 1, 16));
    assert_eq!(decode_split_grid(false, 511, 131_072), (0, 512, 0, 1));
    assert_eq!(decode_split_grid(false, 512, 131_072), (0, 512, 0, 2));
    assert_eq!(decode_split_grid(false, 131_071, 131_072), (0, 512, 0, 256));
    assert_eq!(decode_split_grid(false, 262_143, 262_144), (0, 1024, 0, 256));
}

#[test]
fn grouped_decode_layout_invariants_and_boundaries() {
    use std::collections::BTreeSet;
    let check_invariants = |ranges: &[(usize, usize)], kps: usize, groups: &[DecAttnGroup], block_entry: &dyn Fn(usize, usize) -> Option<(u32, u32)>| {
        let blks_per_split = kps / crate::blocks::BLOCK;
        let mut seen = BTreeSet::new();
        for g in groups {
            assert!(g.count >= 1 && (g.count as usize) <= DEC_GRP_MAX);
            let abs_sp = g.abs_sp as usize;
            let split_start = abs_sp * kps;
            let split_end = split_start + kps;
            let rep = g.span_idx[0] as usize;
            for u in 0..(g.count as usize) {
                let si = g.span_idx[u] as usize;
                assert!(si < ranges.len());
                assert!(seen.insert((si, abs_sp)), "duplicate (si={si}, abs_sp={abs_sp})");
                let (first_pos, pos) = ranges[si];
                assert!(abs_sp >= first_pos / kps && abs_sp <= pos / kps);
                if g.count > 1 {
                    // Grouped lanes MUST be strictly interior (no partial start, no p == pos)
                    assert!(first_pos <= split_start && pos >= split_end, "non-interior lane grouped at abs_sp={abs_sp}");
                    // AND MUST have identical Some((slot != 0, gen != 0)) across ALL blocks in the split.
                    for b in 0..blks_per_split {
                        let blk = abs_sp * blks_per_split + b;
                        let e_rep = block_entry(rep, blk).expect("rep entry must exist");
                        let e_cur = block_entry(si, blk).expect("cur entry must exist");
                        assert!(e_rep.0 != 0 && e_rep.1 != 0);
                        assert_eq!(e_rep, e_cur, "divergent block {blk} grouped between lane {rep} and {si}");
                    }
                }
            }
        }
        // Exact partition: every `(si, abs_sp)` in every lane's visible range appears once.
        let mut expected = BTreeSet::new();
        for (si, &(first_pos, pos)) in ranges.iter().enumerate() {
            for abs_sp in (first_pos / kps)..=(pos / kps) {
                expected.insert((si, abs_sp));
            }
        }
        assert_eq!(seen, expected);
    };

    // 1. Boundary splits: 63/64/65, 511/512/513, 1023/1024/1025 (SWA + Full)
    let boundary_positions = [63usize, 64, 65, 511, 512, 513, 1023, 1024, 1025];
    let shared_all = |_si: usize, blk: usize| Some(((blk as u32) + 1, 7u32));
    for &(swa, kps) in &[(true, 64usize), (false, 512usize)] {
        let ranges: Vec<(usize, usize)> = boundary_positions
            .iter()
            .map(|&p| (if swa { (p + 1).saturating_sub(EXT) } else { 0 }, p))
            .collect();
        let grps = build_decode_groups(&ranges, kps, false, shared_all);
        check_invariants(&ranges, kps, &grps, &shared_all);
        let grps_off = build_decode_groups(&ranges, kps, true, shared_all);
        check_invariants(&ranges, kps, &grps_off, &shared_all);
        assert!(grps_off.iter().all(|g| g.count == 1));
        // For SWA at pos=1024 (first_pos=1) and pos=1025 (first_pos=2), abs_sp=0 is partial -> never grouped with pos=64..1023!
        if swa {
            for g in &grps {
                if g.abs_sp == 0 && g.count > 1 {
                    for u in 0..(g.count as usize) {
                        let si = g.span_idx[u] as usize;
                        assert!(ranges[si].0 == 0 && ranges[si].1 >= 64);
                    }
                }
            }
        }
    }

    // 2. Divergent block inside a 512-token full split (same blk0..2, divergent blk3) and stale generation!
    let ranges_div = vec![(0usize, 1200usize); 4];
    let div_blocks = |si: usize, blk: usize| -> Option<(u32, u32)> {
        match (si, blk) {
            // Split 0 (blocks 0..8): lanes 0 and 1 identical; lane 2 diverges at block 3; lane 3 has different generation at block 5!
            (0 | 1, b) if b < 8 => Some((10 + b as u32, 1)),
            (2, b) if b < 3 => Some((10 + b as u32, 1)),
            (2, b) if b < 8 => Some((100 + b as u32, 1)),
            (3, 5) => Some((15, 2)), // SAME slot 15, DIFFERENT generation 2!
            (3, b) if b < 8 => Some((10 + b as u32, 1)),
            // Split 1 (blocks 8..16): all 4 private
            (s, b) => Some((200 + (s as u32) * 32 + b as u32, 1)),
        }
    };
    let grps_div = build_decode_groups(&ranges_div, 512, false, div_blocks);
    check_invariants(&ranges_div, 512, &grps_div, &div_blocks);
    let sp0_grps: Vec<&DecAttnGroup> = grps_div.iter().filter(|g| g.abs_sp == 0).collect();
    assert_eq!(sp0_grps.len(), 3, "only lanes 0 & 1 may group on split 0; lane 2 (blk3 diff) & lane 3 (gen diff) must be separate");
    assert!(sp0_grps.iter().any(|g| g.count == 2 && g.span_idx[0] == 0 && g.span_idx[1] == 1));

    // 3. 20 lanes sharing an 8192-token (16 full splits) prefix with staggered decode positions -> G_MAX = 8 packing!
    let ranges_20: Vec<(usize, usize)> = (0..20).map(|i| (0usize, 8192 + 64 + i * 17)).collect();
    let shared_8k = |si: usize, blk: usize| -> Option<(u32, u32)> {
        if blk < 128 {
            Some((1 + blk as u32, 3)) // 8192 shared prefix tokens (128 blocks = 16 full splits)
        } else {
            Some((1000 + (si as u32) * 64 + blk as u32, 1)) // private suffix
        }
    };
    let grps_20 = build_decode_groups(&ranges_20, 512, false, shared_8k);
    check_invariants(&ranges_20, 512, &grps_20, &shared_8k);
    // Each of the 16 shared splits (0..16) splits 20 lanes into 3 groups: 8 + 8 + 4!
    for abs_sp in 0..16u32 {
        let mut counts: Vec<u32> = grps_20.iter().filter(|g| g.abs_sp == abs_sp).map(|g| g.count).collect();
        counts.sort();
        assert_eq!(counts, vec![4, 8, 8]);
    }
    // Suffix split 16 (contains pos for all 20 lanes) must be 20 singletons.
    assert_eq!(grps_20.iter().filter(|g| g.abs_sp == 16 && g.count == 1).count(), 20);
    // Total groups: 16 * 3 + 20 = 68 vs 20 * 17 = 340 ungrouped (5.0x fewer TGs!).
    assert_eq!(grps_20.len(), 68);
}

fn run_shader_compiler_validation<F4, F3>(
    mut probe_and_compile_v4: F4,
    mut probe_and_compile_v3: F3,
    full_src: &str,
    dec_src: &str,
) -> Result<&'static str, String>
where
    F4: FnMut(&str, bool) -> Result<(), String>,
    F3: FnMut(&str) -> Result<(), String>,
{
    let probe = "#include <metal_stdlib>\nkernel void probe() {}\n";
    match probe_and_compile_v4(probe, false) {
        Ok(()) => {
            probe_and_compile_v4(full_src, true)
                .map_err(|e| format!("Metal 4.0 fp16 compile failed:\n{e}"))?;
            let kv8_src = format!("#define KV8 1\n{full_src}");
            probe_and_compile_v4(&kv8_src, true)
                .map_err(|e| format!("Metal 4.0 KV8 compile failed:\n{e}"))?;
            Ok("metal4.0")
        }
        Err(e4) => match probe_and_compile_v3(probe) {
            Ok(()) => {
                probe_and_compile_v3(dec_src)
                    .map_err(|e| format!("Metal 3.1 decode kernels (fp16) compile failed:\n{e}"))?;
                let dec_kv8_src = format!("#define KV8 1\n{dec_src}");
                probe_and_compile_v3(&dec_kv8_src)
                    .map_err(|e| format!("Metal 3.1 decode kernels (KV8) compile failed:\n{e}"))?;
                Ok("metal3.1")
            }
            Err(e3) => Err(format!(
                "no Metal shader compiler ran (refusing silent pass); Metal 4.0 probe error: {e4}; Metal 3.1 probe error: {e3}"
            )),
        },
    }
}

#[test]
fn shader_validation_fails_when_no_compiler_runs() {
    let err = run_shader_compiler_validation(
        |_, _| Err("v4 unavailable".into()),
        |_| Err("v3 unavailable".into()),
        "kernel void a() {}",
        "kernel void b() {}",
    )
    .unwrap_err();
    assert!(err.contains("no Metal shader compiler ran"), "{err}");

    // Verify BOTH fp16 and KV8 variants are compiled and checked in each tier.
    let mut seen = Vec::new();
    let tier = run_shader_compiler_validation(
        |src, check_syms| {
            seen.push((src.starts_with("#define KV8 1\n"), check_syms));
            Ok(())
        },
        |_| unreachable!(),
        "full_shader_body",
        "dec_shader_body",
    )
    .unwrap();
    assert_eq!(tier, "metal4.0");
    assert_eq!(seen, vec![(false, false), (false, true), (true, true)]);
}

#[test]
fn metal_shader_declarations_and_compiler_validation() {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode};

    let kernels_src = include_str!("../kernels.metal");
    let prefill_src = include_str!("../prefill.metal");
    let bench_src = include_str!("../bench_kernels.metal");
    let full_src = format!("{kernels_src}\n{prefill_src}\n{bench_src}");

    // Validate dead kernels have been removed and do not linger in shader sources.
    for dead in [
        "kernel void k_norm(",
        "kernel void k_ctrl_down(",
        "kernel void k_attn_blk(",
        "kernel void k_attn_blk_dec(",
        "struct NormJob",
        "struct NormArgs",
        "struct AttnBlkDecArgs",
        "template<int NR, int NSG> void kg_s(",
        "template<int BM, int BN, int NSG> void kg_d(",
        "template<int BM, int BN, int NSG> void kg_db(",
        "template<int NSG, int FUSE, int NACT> void kg_h(",
    ] {
        assert!(!full_src.contains(dead), "dead shader declaration `{dead}` must be removed");
    }

    // Validate every kernel listed in `KERNELS` is declared in the shader source.
    for &name in KERNELS {
        let decl = format!("kernel void {name}(");
        let macro_decl = name.starts_with("k_mix_b_")
            || name.starts_with("k_gemm_sg")
            || name.starts_with("k_moe_sg");
        if !macro_decl {
            assert!(full_src.contains(&decl), "missing Metal kernel declaration `{decl}`");
        }
    }
    for s in [
        "struct DecodeSpanDesc",
        "struct EmbedDecArgs",
        "struct CombineArgs",
        "struct CombineDecArgs",
        "struct DecAttnGroup",
        "struct KvPrepDecArgs",
        "struct AttnGrpDecArgs",
        "struct PostConvDecArgs",
        "struct TopkRoute3Args",
        "struct LoopCdArgs",
        "mem_flags::mem_device | mem_flags::mem_threadgroup",
        "#define KV_CLEARED_ENTRY (~0UL)",
        "kernel void k_embed_dec(",
        "kernel void k_kvprep_dec(",
        "kernel void k_attn_dec_solo(",
        "kernel void k_attn_grp_dec(",
        "kernel void k_attn_combine_dec(",
        "kernel void k_postconv_state_dec(",
        "kernel void k_moe_topk_route3(",
        "kernel void k_loopcd_blend(",
    ] {
        assert!(full_src.contains(s), "missing expected Metal declaration `{s}`");
    }

    let stripped_kernels = kernels_src
        .replace("#include <metal_tensor>\n", "")
        .replace("#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n", "");
    let topk_route_start = prefill_src
        .find("struct TopkRoute3Args")
        .expect("TopkRoute3Args in prefill.metal");
    let topk_route_end = prefill_src[topk_route_start..]
        .find("\n// ---- expert sum per token")
        .map(|rel| topk_route_start + rel)
        .unwrap_or(prefill_src.len());
    let postconv_start = prefill_src
        .find("struct PostConvDecArgs")
        .expect("PostConvDecArgs in prefill.metal");
    let postconv_end = prefill_src[postconv_start..]
        .find("\n// ---- per-position conv state")
        .map(|rel| postconv_start + rel)
        .unwrap_or(prefill_src.len());
    let dec_src = format!(
        "{stripped_kernels}\n#define NTC 2\n{}\n{}\n",
        &prefill_src[topk_route_start..topk_route_end],
        &prefill_src[postconv_start..postconv_end],
    );

    let bench_kernels_live = [
        "k_layout_sg",
        "k_layout_att",
        "kb_dev_2_2",
        "kb_dev_2_4",
        "kb_dev_4_2",
        "kb_dev_4_4",
        "kb_dev_8_2",
        "kb_dev_1_4",
        "kb_blk_2_4",
        "kb_blk_4_4",
        "kb_blk_8_2",
        "kb_blk_4_8",
        "kd_dev_4_4",
        "kd_dev_2_4",
        "kd_tg_2_8",
        "kd_dev_1_4",
        "kd_dev_2_2",
        "kd_dev_1_8",
        "kd_dev_2_8",
        "kd_dev_1_2",
        "kg_iso0_3",
        "kg_iso2_3",
        "kgb_64_128_8",
        "kgb_64_64_4",
        "kgs_64_128_8",
        "kgs_64_64_4",
        "kgs_64_32_4",
        "kgs_64_96_6",
        "kgs_64_112_7",
        "kgs_64_192_12",
        "kgs_64_256_16",
        "kgsg_4",
        "kgsg_8",
        "kgsg_16",
        "kgsgf0_4",
        "kgsgf0_8",
        "kgsgf0_16",
        "kgmode_0_0",
        "kgmode_1_0",
        "kgmode_0_1",
        "kgmode_1_1",
        "kgpeak_8_0",
        "kgpeak_8_32",
        "kgpeak_8_128",
        "kgpeak_8_576",
        "kgpeak_1_0",
        "kgpeak_4_0",
    ];

    if let Some(dev) = MTLCreateSystemDefaultDevice() {
        // Primary path: compile in-memory via Metal's API (zero temp files, zero subprocesses)
        // and verify every symbol in `KERNELS` + live `bench_kernels.metal` entry points resolves.
        let compile_in_mem = |ver: MTLLanguageVersion, src: &str, check_all: bool| -> Result<(), String> {
            let opts = MTLCompileOptions::new();
            opts.setMathMode(MTLMathMode::Fast);
            opts.setLanguageVersion(ver);
            let lib = dev
                .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&opts))
                .map_err(|e| e.localizedDescription().to_string())?;
            if check_all {
                for &name in KERNELS.iter().chain(bench_kernels_live.iter()) {
                    if lib.newFunctionWithName(&NSString::from_str(name)).is_none() {
                        return Err(format!("compiled MTLLibrary missing kernel function `{name}`"));
                    }
                }
            }
            Ok(())
        };
        run_shader_compiler_validation(
            |src, check_all| compile_in_mem(MTLLanguageVersion::Version4_0, src, check_all),
            |src| compile_in_mem(MTLLanguageVersion::Version3_1, src, false),
            &full_src,
            &dec_src,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    } else {
        // Headless fallback when no `MTLDevice` exists: stay strictly inside `./target/shader-check`.
        let work_dir = std::path::Path::new("target").join(format!("shader-check-{}", std::process::id()));
        std::fs::create_dir_all(&work_dir).expect("create ./target/shader-check dir");
        let seq = Cell::new(0usize);
        let compile_xcrun = |std_flag: &str, src: &str| -> Result<(), String> {
            let id = seq.get() + 1;
            seq.set(id);
            let tmp_in = work_dir.join(format!("s_{id}.metal"));
            let tmp_out = work_dir.join(format!("s_{id}.air"));
            std::fs::write(&tmp_in, src).map_err(|e| format!("write {}: {e}", tmp_in.display()))?;
            let res = std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metal", std_flag, "-ffast-math", "-c"])
                .arg(&tmp_in)
                .arg("-o")
                .arg(&tmp_out)
                .output();
            let _ = std::fs::remove_file(&tmp_in);
            let _ = std::fs::remove_file(&tmp_out);
            match res {
                Ok(out) if out.status.success() => Ok(()),
                Ok(out) => Err(String::from_utf8_lossy(&out.stderr).into_owned()),
                Err(e) => Err(format!("spawn xcrun: {e}")),
            }
        };
        let res = run_shader_compiler_validation(
            |src, _| compile_xcrun("-std=metal4.0", src),
            |src| compile_xcrun("-std=metal3.1", src),
            &full_src,
            &dec_src,
        );
        let _ = std::fs::remove_dir_all(&work_dir);
        res.unwrap_or_else(|e| panic!("{e}"));
    }
}

#[test]
fn step_input_rejects_duplicate_lanes_and_zero_ink_splits() {
    assert!(check_distinct_lane_ptrs(&[]).is_err());
    assert!(check_distinct_lane_ptrs(&[0x1000]).is_ok());
    assert!(check_distinct_lane_ptrs(&[0x1000, 0x2000, 0x3000]).is_ok());
    let dup_err = check_distinct_lane_ptrs(&[0x1000, 0x2000, 0x1000]).unwrap_err();
    assert!(dup_err.contains("spans 0 and 2 name the same lane"), "{dup_err}");
    let too_many: Vec<usize> = (1..=MAX_SPANS + 1).collect();
    assert!(check_distinct_lane_ptrs(&too_many).is_err());

    assert_eq!(configured_split_target(Err(std::env::VarError::NotPresent)), Ok(8));
    assert_eq!(configured_split_target(Ok("1".into())), Ok(1));
    assert_eq!(configured_split_target(Ok("256".into())), Ok(256));
    for bad in ["0", "257", "-1", "", "nan", "1,2"] {
        assert!(configured_split_target(Ok(bad.into())).is_err(), "INK_SPLITS={bad:?} must be refused");
    }
}

#[test]
fn prefill_scratch_zero_range_excludes_logits_and_skips_unread_span_lmhead() {
    for &(lean, kvb) in &[(true, 2usize), (false, 2), (true, 1)] {
        let b = BArena::new(0, 4096, lean, kvb, N_VOCAB_PAD);
        let [start, count] = prefill_scratch_zero_range(&b);
        assert_eq!(start, b.sk);
        assert_eq!(start + count, b.lg, "k_zero must stop at b.lg and never clear vocabulary logits rows");
        assert!(b.lg + b.lg_rows * (N_VOCAB_PAD as u32) <= b.perm);
        assert!(count < b.perm - b.sk, "zero range must be strictly smaller than old [b.sk, b.perm)");
    }
    assert!(prefill_span_reads_logits(true, None, 0));
    assert!(!prefill_span_reads_logits(false, None, 0));
    let mask = [false, true, false, true];
    assert!(!prefill_span_reads_logits(true, Some(&mask), 0));
    assert!(prefill_span_reads_logits(true, Some(&mask), 1));
    assert!(!prefill_span_reads_logits(true, Some(&mask), 2));
    assert!(prefill_span_reads_logits(true, Some(&mask), 3));
    assert!(!prefill_span_reads_logits(false, Some(&mask), 1));
}

#[test]
fn validate_lane_block_span_checks_interior_blocks_in_all_profiles() {
    use crate::blocks::{BlockIndex, BlockKind, BlockPools, BlockTable, CLEARED_GPU_TABLE_ENTRY};

    let mut pools = BlockPools::new(8);
    let mut table = BlockTable::new(8).unwrap();
    for b in 0..4u32 {
        let (wf, _) = pools.full.take_free().unwrap();
        let gf = pools.full.record(wf).unwrap().generation();
        table.fill(&mut pools.full, BlockIndex::new(b), wf, gf).unwrap();
        let (ww, _) = pools.window.take_free().unwrap();
        let gw = pools.window.record(ww).unwrap().generation();
        table.fill(&mut pools.window, BlockIndex::new(b), ww, gw).unwrap();
    }
    // First slot in slab 0 encodes as 0, strictly distinct from CLEARED_GPU_TABLE_ENTRY (`u64::MAX`).
    let first_entry = table.get(BlockKind::Full, BlockIndex::new(0)).unwrap().unwrap();
    assert_eq!(pools.full.gpu_table_entry(first_entry).unwrap(), 0);
    assert_ne!(0, CLEARED_GPU_TABLE_ENTRY);

    // Valid 4-block span [0, 256) passes.
    assert!(validate_lane_block_span(&table, &pools, 0, 256).is_ok());

    // Clear ONLY interior block 2 (endpoints 0 and 3 remain valid!) -> MUST fail in both debug and release builds!
    table.clear_kind_entry(&mut pools.full, BlockIndex::new(2)).unwrap();
    let err = validate_lane_block_span(&table, &pools, 0, 256).unwrap_err();
    assert!(err.contains("missing required Full block 2"), "{err}");

    // Restore block 2, then recycle interior Window block 1 so a stale table clone still holds gen 1 while endpoints 0 & 3 stay live!
    let (wf2, _) = pools.full.take_free().unwrap();
    let gf2 = pools.full.record(wf2).unwrap().generation();
    table.fill(&mut pools.full, BlockIndex::new(2), wf2, gf2).unwrap();
    assert!(validate_lane_block_span(&table, &pools, 0, 256).is_ok());

    let stale_table = table.clone();
    let win1 = table.get(BlockKind::Window, BlockIndex::new(1)).unwrap().unwrap();
    table.clear_kind_entry(&mut pools.window, BlockIndex::new(1)).unwrap();
    pools.window.take(win1.slot()).unwrap();
    let stale_err = validate_lane_block_span(&stale_table, &pools, 0, 256).unwrap_err();
    assert!(stale_err.contains("Window block 1 entry invalid"), "{stale_err}");
}

#[test]
fn engine_slice_bounds_and_refcell_borrow_lifetime_guard() {
    assert_eq!(checked_byte_end(0, 4, 4, 16, "t"), Ok(16));
    assert!(checked_byte_end(1, 4, 4, 16, "t").is_err());
    assert!(checked_byte_end(usize::MAX, 1, 4, 16, "t").is_err());
    assert!(checked_byte_end(0, (usize::MAX / 4) + 1, 4, usize::MAX, "t").is_err());

    if let Some(dev) = objc2_metal::MTLCreateSystemDefaultDevice() {
        use objc2_metal::{MTLDevice, MTLResourceOptions};
        let buf = dev.newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared).unwrap();
        unsafe {
            let s = std::slice::from_raw_parts_mut(buf.contents().as_ptr().cast::<f32>(), 16);
            for (i, v) in s.iter_mut().enumerate() { *v = (i + 1) as f32; }
        }
        let cell = RefCell::new(buf);
        {
            let view = borrow_buf_slice::<f32>(&cell, 4, 8, "test_slice");
            assert_eq!(&*view, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
            // While `Ref<'_, [f32]>` is alive, any mutable reallocation (`ensure_batch` / `release_batch` / `ensure_dbatch`)
            // cannot mutably borrow `cell`, preventing dangling slice references structurally.
            assert!(cell.try_borrow_mut().is_err());
        }
        assert!(cell.try_borrow_mut().is_ok());
    }
}

#[test]
fn loopcd_weak_buffer_sizing_and_budget_accounting() {
    assert_eq!(Engine::loopcd_weak_bytes_for(0, N_VOCAB_PAD), N_VOCAB_PAD * 4);
    assert_eq!(Engine::loopcd_weak_bytes_for(1, N_VOCAB_PAD), N_VOCAB_PAD * 4);
    assert_eq!(Engine::loopcd_weak_bytes_for(16, N_VOCAB_PAD), 16 * N_VOCAB_PAD * 4);
    assert_eq!(Engine::loopcd_weak_bytes_for(64, N_VOCAB_PAD), 64 * N_VOCAB_PAD * 4);

    // Verify BlockedSlab initializes unmapped GPU block-table entries to `CLEARED_GPU_TABLE_ENTRY` (`u64::MAX`).
    if objc2_metal::MTLCreateSystemDefaultDevice().is_some() {
        let gpu = Gpu::new("#include <metal_stdlib>\nkernel void p() {}\n", &["p"]);
        let mut slab = BlockedSlab::new(64);
        let s0 = slab.alloc_slot(&gpu, 64);
        assert_eq!(s0, 0);
        let bt_words = unsafe {
            std::slice::from_raw_parts(
                slab.bt_buf.as_ref().unwrap().contents().as_ptr().cast::<u64>(),
                2 * slab.bt_stride_u64,
            )
        };
        assert!(bt_words.iter().all(|&w| w == crate::blocks::CLEARED_GPU_TABLE_ENTRY));
        // Write 0 (valid slab-0 Slot(1) encoding!) into entry 0, recycle slot 0, and confirm re-alloc resets it to `CLEARED_GPU_TABLE_ENTRY`.
        unsafe {
            *slab.bt_buf.as_ref().unwrap().contents().as_ptr().cast::<u64>() = 0;
        }
        slab.free_slots.push(s0);
        let s0_reused = slab.alloc_slot(&gpu, 64);
        assert_eq!(s0_reused, 0);
        let word0 = unsafe { *slab.bt_buf.as_ref().unwrap().contents().as_ptr().cast::<u64>() };
        assert_eq!(word0, crate::blocks::CLEARED_GPU_TABLE_ENTRY);
    }
}

#[test]
fn wave2a_decode_dispatch_and_barrier_reduction_32_lanes() {
    let count_step = |n_lanes: usize, hgemm: bool, logits: bool, argmax: bool, skip: u32, wave2a: bool| -> (usize, usize) {
        let mut dispatches = 0usize;
        let mut barriers = 0usize;
        let add_go = |d: &mut usize, b: &mut usize, cls: u32, count: usize| {
            if skip & (1 << cls) == 0 {
                *d += count;
                *b += 1;
            }
        };
        let add_tohalf = |d: &mut usize, b: &mut usize, cls: u32, count: usize| {
            if skip & (1 << cls) == 0 {
                *d += count;
            }
            *b += 1;
        };
        add_go(&mut dispatches, &mut barriers, C_EMBED, 1);

        let n_uncontrolled = LOOP_START + (LOOP_END - LOOP_START) + (N_PHYS - LOOP_END);
        let n_controlled = (N_PASS - 1) * (LOOP_END - LOOP_START);
        assert_eq!(n_uncontrolled + n_controlled, N_SLOT);
        assert_eq!(n_uncontrolled, 24);
        assert_eq!(n_controlled, 24);

        for controlled in [false, true] {
            let repeat = if controlled { n_controlled } else { n_uncontrolled };
            for _ in 0..repeat {
                if controlled {
                    add_go(&mut dispatches, &mut barriers, C_MIXA, 2);
                    let fuse_down = wave2a && ctrl_down_direct_half(skip);
                    add_go(&mut dispatches, &mut barriers, C_CTRLD, 1);
                    if !fuse_down {
                        add_tohalf(&mut dispatches, &mut barriers, C_CTRLA, 3);
                    }
                    add_go(&mut dispatches, &mut barriers, C_CTRLA, 1);
                    add_go(&mut dispatches, &mut barriers, C_CTRLN, 1);
                } else {
                    add_go(&mut dispatches, &mut barriers, C_MIXA, 1);
                }
                add_go(&mut dispatches, &mut barriers, C_QKV, 1);
                add_go(&mut dispatches, &mut barriers, C_ATTN, 1);
                add_go(&mut dispatches, &mut barriers, C_COMB, 1);
                let direct_half = wave2a && decode_combine_direct_half(hgemm, skip);
                if !direct_half {
                    add_tohalf(&mut dispatches, &mut barriers, C_WO, 1);
                }
                add_go(&mut dispatches, &mut barriers, C_WO, 1);
                add_go(&mut dispatches, &mut barriers, C_WOPOST, 1);
                add_go(&mut dispatches, &mut barriers, C_MIXF, 1);
                if controlled {
                    let fuse_down = wave2a && ctrl_down_direct_half(skip);
                    add_go(&mut dispatches, &mut barriers, C_CTRLD, 1);
                    if !fuse_down {
                        add_tohalf(&mut dispatches, &mut barriers, C_CTRLA, 1);
                    }
                    add_go(&mut dispatches, &mut barriers, C_CTRLA, 1);
                    add_go(&mut dispatches, &mut barriers, C_CTRLN, 1);
                }
                add_go(&mut dispatches, &mut barriers, C_ROUTER, 1);
                if wave2a && use_fused_topk_route3(hgemm, n_lanes) {
                    add_go(&mut dispatches, &mut barriers, C_TOPK, 1);
                } else {
                    add_go(&mut dispatches, &mut barriers, C_TOPK, 1);
                    add_go(&mut dispatches, &mut barriers, C_TOPK, 1);
                }
                add_tohalf(&mut dispatches, &mut barriers, C_GATHER, 1);
                let cls_start = if wave2a { moe_class_start(n_lanes) } else { 0 };
                let n_cls = MOE_CLS.len() - cls_start;
                add_go(&mut dispatches, &mut barriers, C_GU, n_cls);
                add_go(&mut dispatches, &mut barriers, C_DOWN, n_cls);
                add_go(&mut dispatches, &mut barriers, C_COMB2, 1);
                add_go(&mut dispatches, &mut barriers, C_FFNPOST, 1);
            }
        }
        if logits {
            add_go(&mut dispatches, &mut barriers, C_OMIX, 1);
            add_go(&mut dispatches, &mut barriers, C_LMHEAD, 1);
            if argmax {
                for _ in 0..n_lanes {
                    add_go(&mut dispatches, &mut barriers, C_ARGMAX, 1);
                    add_go(&mut dispatches, &mut barriers, C_ARGMAX, 1);
                }
            }
        }
        (dispatches, barriers)
    };

    // 32-lane blocked decode (hgemm=true, logits=true, argmax=false, skip=0):
    //   Opt 1 (combine direct half):        -48 dispatches, -48 barriers
    //   Opt 2 (ctrl_b down hout=1):         -96 dispatches, -48 barriers (24 controlled layers x [3+1] jobs)
    //   Opt 3 (skip empty MoE class 0):     -96 indirect dispatches, 0 barriers (48 layers x [GU+DOWN])
    //   Opt 4 (fused k_moe_topk_route3):    -48 dispatches, -48 barriers
    //   Total reduction:                   -288 dispatches, -144 barriers
    let (base_d, base_b) = count_step(32, true, true, false, 0, false);
    let (opt_d, opt_b) = count_step(32, true, true, false, 0, true);
    assert_eq!((base_d, base_b), (1131, 963));
    assert_eq!((opt_d, opt_b), (843, 819));
    assert_eq!(base_d - opt_d, 288);
    assert_eq!(base_b - opt_b, 144);

    // Same reduction holds across 1..=64 lanes; above 64 lanes, only Opt 1 + Opt 2 apply (-144 d, -96 b).
    assert_eq!(count_step(1, true, true, false, 0, true), (843, 819));
    assert_eq!(count_step(64, true, true, false, 0, true), (843, 819));
    let (d65, b65) = count_step(65, true, true, false, 0, true);
    assert_eq!((base_d - d65, base_b - b65), (144, 96));

    // Skip-mask semantics: isolating C_WO/C_COMB disables Opt 1 fusion; isolating C_CTRLD/C_CTRLA disables Opt 2 fusion.
    assert!(decode_combine_direct_half(true, 0));
    assert!(!decode_combine_direct_half(false, 0));
    assert!(!decode_combine_direct_half(true, 1 << C_COMB));
    assert!(!decode_combine_direct_half(true, 1 << C_WO));
    assert!(ctrl_down_direct_half(0));
    assert!(!ctrl_down_direct_half(1 << C_CTRLD));
    assert!(!ctrl_down_direct_half(1 << C_CTRLA));
}

#[test]
fn moe_class0_empty_invariant_for_n_le_64() {
    // Exact Rust mirror of `topk_sg` (`kernels.metal`) + `k_moe_route3` (`prefill.metal`) tile class counts.
    let topk_pick = |logits: &[f32; NEXP], bias: &[f32; NEXP]| -> [usize; TOPK] {
        let mut sel = [0.0f32; NEXP];
        for e in 0..NEXP {
            let p = 1.0f32 / (1.0f32 + (-logits[e]).exp());
            let score = p + bias[e];
            let finite = (score.to_bits() & 0x7f80_0000u32) != 0x7f80_0000u32;
            sel[e] = if finite { score } else { f32::from_bits(0xff7f_ffffu32) };
        }
        let mut out = [0usize; TOPK];
        for (k, out_slot) in out.iter_mut().enumerate().take(TOPK) {
            let mut mv = f32::NEG_INFINITY;
            let mut win = NEXP;
            for (e, &s) in sel.iter().enumerate() {
                if s > mv || (s == mv && e < win) {
                    mv = s;
                    win = e;
                }
            }
            let pick = if win < NEXP { win } else { k };
            sel[pick] = f32::NEG_INFINITY;
            *out_slot = pick;
        }
        out
    };
    let route3_class_tiles = |picks: &[[usize; TOPK]]| -> [usize; NTC] {
        let mut cnt = [0usize; NEXP];
        for row in picks {
            let mut seen = [false; NEXP];
            for &e in row {
                assert!(e < NEXP);
                assert!(!seen[e], "a single token must pick TOPK distinct experts");
                seen[e] = true;
                cnt[e] += 1;
            }
        }
        let mut ts = [0usize; NTC];
        for &c in &cnt {
            let r = c % 128;
            ts[0] += c / 128 + usize::from(r > 64);
            ts[1] += usize::from(r > 0 && r <= 64);
        }
        ts
    };

    // Adversarial worst cases: identical finite logits, all-NaN logits, and strongly skewed bias.
    let cases: [([f32; NEXP], [f32; NEXP]); 3] = [
        ([0.0; NEXP], [0.0; NEXP]),
        ([f32::NAN; NEXP], [0.0; NEXP]),
        ([1.5; NEXP], std::array::from_fn(|e| if e >= NEXP - TOPK { 10.0 } else { -10.0 })),
    ];
    for (logits, bias) in &cases {
        let single = topk_pick(logits, bias);
        for n in 1..=64 {
            assert_eq!(moe_class_start(n), 1);
            assert!(use_fused_topk_route3(true, n));
            let picks = vec![single; n];
            let ts = route3_class_tiles(&picks);
            assert_eq!(ts[0], 0, "class 0 (65..128) must be empty for n={n}");
            assert!(ts[1] > 0, "class 1 (1..64) must cover all tiles for n={n}");
        }
        // Tightness check: at n = 65 with all tokens picking the same experts, class 0 becomes non-empty.
        assert_eq!(moe_class_start(65), 0);
        assert!(!use_fused_topk_route3(true, 65));
        let picks_65 = vec![single; 65];
        let ts_65 = route3_class_tiles(&picks_65);
        assert_eq!(ts_65[0], TOPK);
        assert_eq!(ts_65[1], 0);
    }
}

#[test]
fn loopcd_blend_reference_identities_and_adaptive_schedule() {
    let softcap = 20.0f32;
    let cap = |x: f32| softcap * (x / softcap).tanh();

    // 1. A disabled lane retains its raw logits and its normal sampling path.
    let z3_raw = [18.0f32, 12.5, -3.0, -40.0, 0.0];
    let z1_raw = [-5.0f32, 22.0, 10.0, 15.0, 3.0];
    let mut out0 = z3_raw;
    let w0 = loopcd_blend_row(&mut out0, &z1_raw, softcap, 0.0);
    assert_eq!(w0, 0.0);
    assert_eq!(out0, z3_raw);

    // 2. Dominant token (p_{3,(1)} -> 1, p_{3,(2)} -> 0): omega_t -> 0 (deterministic syntax/tool-calls preserved).
    let mut z3_dom = [-50.0f32; 64];
    z3_dom[7] = 30.0; // softcap(30) ~ 18.103, softcap(-50) ~ -19.732 -> gap > 37 nats
    let mut z1_dom = [0.0f32; 64];
    z1_dom[7] = -20.0;
    let w_dom = loopcd_blend_row(&mut z3_dom, &z1_dom, softcap, 0.5);
    assert!(w_dom < 1e-6, "confident distribution must suppress contrast, got omega_t={w_dom}");
    assert!((z3_dom[7] - cap(30.0)).abs() < 1e-4);

    // 3. Tied top-2 (p_{3,(1)} == p_{3,(2)} == 0.5): omega_t == omega_max, and the token refined
    //    upward from z1 -> z3 strictly beats the stagnant token.
    let mut z3_tie = [-100.0f32; 16];
    z3_tie[2] = 10.0; // stagnant: z1 == 10.0 -> z3 == 10.0
    z3_tie[5] = 10.0; // refined upward: z1 == 2.0 -> z3 == 10.0
    let mut z1_tie = [-100.0f32; 16];
    z1_tie[2] = 10.0;
    z1_tie[5] = 2.0;
    let omega_max = 0.4f32;
    let w_tie = loopcd_blend_row(&mut z3_tie, &z1_tie, softcap, omega_max);
    assert!((w_tie - omega_max).abs() < 1e-5, "tied top-2 must reach omega_max={omega_max}, got {w_tie}");
    assert!(z3_tie[5] > z3_tie[2], "token refined upward from z1->z3 must strictly beat stagnant token");

    // 4. Exact Appendix A.3 probability ratio identity across all pairs (u, v):
    //    log p'(v) - log p'(u) == (1 + omega_t)(s3(v) - s3(u)) - omega_t(s1(v) - s1(u)).
    let z3_orig = [4.2f32, 3.9, 3.1, 1.5, -2.0, 25.0, 24.8, -12.0];
    let z1_orig = [1.0f32, 5.5, 2.0, -1.0, 0.5, 10.0, 24.5, -8.0];
    let mut blended = z3_orig;
    let wt = loopcd_blend_row(&mut blended, &z1_orig, softcap, 0.35);
    assert!(wt > 0.0 && wt <= 0.35);
    let lse = |v: &[f32]| -> f32 {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        m + v.iter().map(|&x| (x - m).exp()).sum::<f32>().ln()
    };
    let log_z = lse(&blended);
    let log_p: Vec<f32> = blended.iter().map(|&x| x - log_z).collect();
    for v in 0..z3_orig.len() {
        for u in 0..z3_orig.len() {
            let lhs = log_p[v] - log_p[u];
            let rhs = (1.0 + wt) * (cap(z3_orig[v]) - cap(z3_orig[u]))
                - wt * (cap(z1_orig[v]) - cap(z1_orig[u]));
            assert!((lhs - rhs).abs() < 1e-4, "pair ({v},{u}): lhs={lhs} rhs={rhs}");
        }
    }
}

#[test]
fn loopcd_parallel_top2_butterfly_matches_exact_sort() {
    // Exact Rust mirror of the 1024-thread (32 simdgroups x 32 lanes) top2_merge butterfly in `k_loopcd_blend`.
    let top2_merge = |m1: &mut f32, m2: &mut f32, b1: f32, b2: f32| {
        if b1 >= *m1 {
            *m2 = (*m1).max(b2);
            *m1 = b1;
        } else if b1 > *m2 {
            *m2 = b1;
        }
    };
    let gpu_top2 = |vals: &[f32]| -> (f32, f32) {
        let mut sm1 = [f32::NEG_INFINITY; 32];
        let mut sm2 = [f32::NEG_INFINITY; 32];
        for sg in 0..32usize {
            let mut m1 = [f32::NEG_INFINITY; 32];
            let mut m2 = [f32::NEG_INFINITY; 32];
            for lane in 0..32usize {
                let tid = sg * 32 + lane;
                let mut v = tid;
                while v < vals.len() {
                    let s = vals[v];
                    if s >= m1[lane] { m2[lane] = m1[lane]; m1[lane] = s; }
                    else if s > m2[lane] { m2[lane] = s; }
                    v += 1024;
                }
            }
            for off in [16usize, 8, 4, 2, 1] {
                let prev1 = m1;
                let prev2 = m2;
                for lane in 0..32 {
                    top2_merge(&mut m1[lane], &mut m2[lane], prev1[lane ^ off], prev2[lane ^ off]);
                }
            }
            sm1[sg] = m1[0];
            sm2[sg] = m2[0];
        }
        let mut bm1 = sm1;
        let mut bm2 = sm2;
        for off in [16usize, 8, 4, 2, 1] {
            let prev1 = bm1;
            let prev2 = bm2;
            for lane in 0..32 {
                top2_merge(&mut bm1[lane], &mut bm2[lane], prev1[lane ^ off], prev2[lane ^ off]);
            }
        }
        (bm1[0], bm2[0])
    };

    let cases: Vec<Vec<f32>> = vec![
        vec![7.0],
        vec![5.0, 5.0],
        vec![1.0, 9.0, 9.0, 2.0],
        // Tied maxima in different simdgroups and same thread stride
        {
            let mut v = vec![-10.0; 2500];
            v[17] = 12.5;
            v[17 + 1024] = 12.5;
            v[500] = 11.0;
            v
        },
        {
            let mut v = vec![-10.0; 2500];
            v[3] = 12.5;
            v[1023] = 12.5;
            v
        },
        (0..2053).map(|i| ((i * 7919) % 1000) as f32 * 0.01 - 5.0).collect(),
    ];
    for vals in &cases {
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let want1 = sorted[0];
        let want2 = sorted.get(1).copied().unwrap_or(f32::NEG_INFINITY);
        let (got1, got2) = gpu_top2(vals);
        assert_eq!((got1, got2), (want1, want2), "top2 mismatch for len={}", vals.len());
    }
}

#[test]
fn vecalloc_liveness_loopcd_zero_collisions() {
    // Verify zero slot collisions across prelude + pass 0 + weak coda + pass 1..2 + real coda
    // for BOTH batched decode (`B_NVEC = 10`) and single-stream decode (`N_VEC = 16`).
    for n_vec in [B_NVEC, N_VEC] {
        let mut va = VecAlloc { next: 1, n: n_vec };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        let mut max_live = 0usize;

        let mut step_layer = |va: &mut VecAlloc, ip: usize, cur: usize, blocks: &mut Vec<usize>, pinned: &[usize], guard: &[usize]| -> usize {
            let block_start = ip.is_multiple_of(BLOCK);
            let mut live: Vec<usize> = blocks.clone();
            live.push(cur);
            live.extend_from_slice(pinned);
            let mut uniq = live.clone();
            uniq.sort_unstable();
            uniq.dedup();
            max_live = max_live.max(uniq.iter().filter(|&&x| x != 0).count());
            let mid = va.alloc(&live);
            assert!(!guard.contains(&mid), "mid={mid} collided with protected slot {guard:?}");
            assert!(!live.contains(&mid));
            if block_start { blocks.push(cur); }

            let mut live2: Vec<usize> = blocks.clone();
            live2.push(mid);
            live2.extend_from_slice(pinned);
            let mut uniq2 = live2.clone();
            uniq2.sort_unstable();
            uniq2.dedup();
            max_live = max_live.max(uniq2.iter().filter(|&&x| x != 0).count());
            let out = va.alloc(&live2);
            assert!(!guard.contains(&out), "out={out} collided with protected slot {guard:?}");
            assert!(!live2.contains(&out));
            out
        };

        for ip in 0..LOOP_START {
            cur = step_layer(&mut va, ip, cur, &mut blocks, &[], &[]);
        }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        assert_eq!(prelude_blocks, vec![0]);
        let mut bank = (0usize, 0usize);

        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = step_layer(&mut va, ip, cur, &mut blocks, &pinned, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
            if pass == 0 {
                // Weak coda immediately after pass 0 must preserve prelude, prelude_blocks, bank.0, and bank.1.
                let mut cur_w = cur;
                let mut blocks_w = blocks.clone();
                let pinned_w = [prelude, bank.0, bank.1];
                for ip in LOOP_END..N_PHYS {
                    cur_w = step_layer(&mut va, ip, cur_w, &mut blocks_w, &pinned_w, &pinned_w);
                }
                assert_eq!(blocks_w.len(), 4, "weak coda must produce all 4 depth-residual block states for output_res_query");
                assert_ne!(cur_w, prelude);
                assert_ne!(cur_w, bank.0);
                assert_ne!(cur_w, bank.1);
            }
        }
        for ip in LOOP_END..N_PHYS {
            cur = step_layer(&mut va, ip, cur, &mut blocks, &[], &[]);
        }
        assert_eq!(blocks.len(), 4);
        assert!(max_live < n_vec - 1, "peak non-zero live vectors ({max_live}) must fit within n_vec-1 ({})", n_vec - 1);
        assert_ne!(cur, 0);
    }
}

#[test]
fn sync_lane_block_table_64_decode_positions_writes_once_then_63_zero_steps() {
    use crate::blocks::{BlockIndex, BlockKind, BlockPools, BlockTable, BLOCK as KV_BLOCK};
    use objc2_metal::MTLCreateSystemDefaultDevice;

    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("skipping Metal block-table sync test: no Metal device");
        return;
    }
    let trivial_metal = "#include <metal_stdlib>\nusing namespace metal;\nkernel void k_nop() {}\n";
    let gpu = Gpu::new(trivial_metal, &["k_nop"]);
    let mut pools = BlockPools::gpu(&gpu.dev, 8, 8, 2);
    let slab = Rc::new(RefCell::new(BlockedSlab::new(32)));
    let slot_idx = slab.borrow_mut().alloc_slot(&gpu, 32);
    let lane = Lane {
        kv_swa: RefCell::new(gpu.buffer(16)),
        kv_full: RefCell::new(gpu.buffer(16)),
        swa_cap: Cell::new(0),
        full_cap: Cell::new(0),
        cs_own: None,
        toks: gpu.buffer(2048 * 4),
        tier: None,
        prune: RefCell::new(None),
        block_table: RefCell::new(BlockTable::new(32).unwrap()),
        use_blocks: Cell::new(true),
        blocked_slot: Some(BlockedSlot {
            slab: slab.clone(),
            slot_idx,
            bt_blocks: 32,
        }),
    };

    let ensure_blocks = |pools: &mut BlockPools, lane: &Lane, p0: usize, t: usize| {
        let mut tbl = lane.block_table.borrow_mut();
        for b in (p0 / KV_BLOCK)..=((p0 + t - 1) / KV_BLOCK) {
            let bi = BlockIndex::new(b as u32);
            for kind in [BlockKind::Full, BlockKind::Window] {
                if tbl.get(kind, bi).unwrap().is_none() {
                    let pool = pools.of_mut(kind);
                    let (s, _) = pool.take_free().unwrap();
                    let gen = pool.record(s).unwrap().generation();
                    tbl.fill(pool, bi, s, gen).unwrap();
                }
            }
        }
    };
    let finish_span = |pools: &mut BlockPools, lane: &Lane, p0: usize, t: usize, seal_full: bool| {
        let end = p0 + t;
        let tbl = lane.block_table.borrow();
        for b in (p0 / KV_BLOCK)..=((end - 1) / KV_BLOCK) {
            let bi = BlockIndex::new(b as u32);
            let blen = (end - b * KV_BLOCK).min(KV_BLOCK) as u32;
            for kind in [BlockKind::Full, BlockKind::Window] {
                let e = tbl.get(kind, bi).unwrap().unwrap();
                let rec = pools.of_mut(kind).record_mut(e.slot()).unwrap();
                rec.set_length(blen).unwrap();
                if seal_full && blen == KV_BLOCK as u32 {
                    rec.seal();
                }
            }
        }
    };

    // Prefill block 0 (`[0, 64)`): initial full sync writes all 32 entries per kind, sealing block 0.
    ensure_blocks(&mut pools, &lane, 0, 64);
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Full), Some((0, 31)));
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Window), Some((0, 31)));
    assert_eq!(sync_blocked_lane_table(&pools, &lane, 0, 64), (32, 32));
    finish_span(&mut pools, &lane, 0, 64, true);
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Full), None);
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Window), None);

    // Plant sentinels in unused GPU block-table entries `[2..32)` to prove steady-state sync touches nothing.
    for swa in [false, true] {
        lane.with_bt_buf(swa, |bt_buf, byte_off, cap_blk| {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (bt_buf.contents().as_ptr() as *mut u8).add(byte_off) as *mut u64,
                    cap_blk,
                )
            };
            for w in &mut dst[2..32] {
                *w = 0xDEAD_BEEF_CAFE_BABEu64;
            }
        });
    }

    // Step across 64 consecutive decode positions `pos = 64..128`:
    for pos in 64..128 {
        ensure_blocks(&mut pools, &lane, pos, 1);
        if pos == 64 {
            assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Full), Some((1, 1)));
            assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Window), Some((1, 1)));
        } else {
            assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Full), None, "pos={pos} must have clean Full table");
            assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Window), None, "pos={pos} must have clean Window table");
        }
        let written = sync_blocked_lane_table(&pools, &lane, pos, 1);
        if pos == 64 {
            assert_eq!(written, (1, 1), "pos=64 must write only newly allocated block 1");
        } else {
            assert_eq!(written, (0, 0), "steady-state decode at pos={pos} must write 0 table words");
        }
        finish_span(&mut pools, &lane, pos, 1, false);
    }

    for swa in [false, true] {
        lane.with_bt_buf(swa, |bt_buf, byte_off, cap_blk| {
            let dst = unsafe {
                std::slice::from_raw_parts(
                    (bt_buf.contents().as_ptr() as *mut u8).add(byte_off) as *mut u64,
                    cap_blk,
                )
            };
            assert!(dst[2..32].iter().all(|&w| w == 0xDEAD_BEEF_CAFE_BABEu64));
        });
    }

    ensure_blocks(&mut pools, &lane, 128, 1);
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Full), Some((2, 2)));
    assert_eq!(lane.block_table.borrow().dirty_range(BlockKind::Window), Some((2, 2)));
    assert_eq!(sync_blocked_lane_table(&pools, &lane, 128, 1), (1, 1));
    let full_b2 = lane.block_table.borrow().get(BlockKind::Full, BlockIndex::new(2)).unwrap().unwrap();
    assert_eq!(pools.full.record(full_b2.slot()).unwrap().refs(), 1);
}

#[test]
#[ignore = "requires Inkling Q4_K_M model and Metal 4"]
fn engine_encode_decode_64_positions_syncs_once_then_63_zero_steps() {
    use crate::blocks::{BlockIndex, BlockKind};
    use objc2_metal::MTLCommandBuffer;

    let model = std::env::var("INKY_MODEL").unwrap_or_else(|_| crate::default_model());
    let eng = Engine::load(&model, 256);
    eng.ensure_batch(64);
    eng.ensure_dbatch(1);
    let lane = Lane::new_blocked(&eng, 192);
    let toks: Vec<u32> = (0..130).map(|i| ((i * 17 + 5) % 500) as u32).collect();
    eng.on(&lane).set_tokens(0, &toks);
    let alloc_blk = |b: u32| {
        let mut pools = eng.block_pool().borrow_mut();
        let mut tbl = lane.block_table().borrow_mut();
        let bi = BlockIndex::new(b);
        for kind in [BlockKind::Full, BlockKind::Window] {
            let pool = pools.of_mut(kind);
            let s = pool.lowest_free().unwrap_or_else(|| pool.grow().unwrap());
            let _ = pool.take(s).unwrap();
            let gen = pool.record(s).unwrap().generation();
            tbl.fill(pool, bi, s, gen).unwrap();
        }
    };
    alloc_blk(0);
    {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &lane, p0: 0, t: 64 }], false, false);
        cb.commit();
        cb.waitUntilCompleted();
    }
    alloc_blk(1);
    for pos in 64..128 {
        if pos == 64 {
            assert_eq!(lane.block_table().borrow().dirty_range(BlockKind::Full), Some((1, 1)));
            assert_eq!(lane.block_table().borrow().dirty_range(BlockKind::Window), Some((1, 1)));
            assert_eq!(eng.sync_lane_block_table_counts(&lane, pos, 1), (1, 1));
        } else {
            assert_eq!(lane.block_table().borrow().dirty_range(BlockKind::Full), None);
            assert_eq!(lane.block_table().borrow().dirty_range(BlockKind::Window), None);
            assert_eq!(eng.sync_lane_block_table_counts(&lane, pos, 1), (0, 0));
        }
        let cb = eng.gpu.cmd();
        eng.encode_decode(&cb, &[Span { lane: &lane, p0: pos, t: 1 }], true, false);
        cb.commit();
        cb.waitUntilCompleted();
        assert!(eng.span_logits(0).iter().all(|x| x.is_finite()));
    }
    let mut pools = eng.block_pool().borrow_mut();
    lane.block_table().borrow_mut().clear_entries(&mut pools).unwrap();
}

#[test]
#[ignore = "requires Metal 4"]
fn loopcd_mixed_batch_preserves_disabled_rows() {
    // In binary32, softcapping the first row can round its distinct logits to a tie.
    let a = 20.0f32;
    let next = f32::from_bits(a.to_bits() + 1);
    let strong = [a, next, 1.0, 1.0, f32::INFINITY, f32::NEG_INFINITY, 100.0, -100.0, f32::NAN, 4.0];
    let weak = [f32::NAN, 0.0, 3.0, -2.0, 0.0, 1.0, f32::NEG_INFINITY, f32::INFINITY, 2.0, f32::NAN];
    let omegas = [0.0f32, 0.5, 0.5, 0.5, 0.5];
    let args = LoopCdArgs {
        logits_off: 0, n: 2, stride: 2, pad0: 0, softcap: 30.0, pad1: 0, pad2: 0, pad3: 0,
    };
    let gpu = Gpu::new(include_str!("../kernels.metal"), &["k_loopcd_blend"]);
    let strong_buf = gpu.buffer_with(&strong.map(f32::to_ne_bytes).concat());
    let weak_buf = gpu.buffer_with(&weak.map(f32::to_ne_bytes).concat());
    let cb = gpu.cmd();
    let e = Enc::new(&cb);
    e.pso(&gpu.pso("k_loopcd_blend"));
    e.buf(&strong_buf, 0, 2);
    e.bytes(&args, 5);
    e.bytes_slice(&omegas, 8);
    e.buf(&weak_buf, 0, 9);
    e.dispatch((1, omegas.len(), 1), 1024);
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
    assert!(cb.error().is_none());

    let actual = unsafe { std::slice::from_raw_parts(strong_buf.contents().as_ptr().cast::<f32>(), strong.len()) };
    assert_eq!(&actual[..2], &strong[..2]);
    let mut expected = [strong[2], strong[3]];
    loopcd_blend_row(&mut expected, &weak[2..4], args.softcap, omegas[1]);
    for (&got, &want) in actual[2..].iter().zip(&expected) {
        assert!((got - want).abs() < 1e-5, "got {got}, expected {want}");
    }
    // Even a zero adaptive weight must not conceal an invalid required weak logit.
    assert!(actual[4..].iter().all(|x| x.is_nan()), "{actual:?}");
    assert_eq!(guided_rows(&omegas), 0b11110);
}

#[test]
fn blocked_slab_alloc_reuse_zero_fill_and_strided_growth() {
    assert_eq!(BlockedSlab::next_cap(0), 1);
    assert_eq!(BlockedSlab::next_cap(1), 1);
    assert_eq!(BlockedSlab::next_cap(2), 2);
    assert_eq!(BlockedSlab::next_cap(3), 4);
    assert_eq!(BlockedSlab::next_cap(4), 4);
    assert_eq!(BlockedSlab::next_cap(5), 8);
    assert_eq!(BlockedSlab::next_cap(8), 8);
    assert_eq!(BlockedSlab::next_cap(9), 16);

    if objc2_metal::MTLCreateSystemDefaultDevice().is_none() {
        return;
    }
    let gpu = Gpu::new("#include <metal_stdlib>\nusing namespace metal;\nkernel void noop() {}\n", &[]);
    let slab = Rc::new(RefCell::new(BlockedSlab::new(16))); // initial stride = 32 u64s

    let alloc_handle = |need_blocks: usize| -> BlockedSlot {
        let slot_idx = slab.borrow_mut().alloc_slot(&gpu, need_blocks);
        BlockedSlot { slab: Rc::clone(&slab), slot_idx, bt_blocks: need_blocks }
    };

    // 1. Allocate 3 slots at initial stride (32 u64s): cap grows 0 -> 1 -> 2 -> 4.
    let s0 = alloc_handle(16);
    assert_eq!(s0.slot_idx, 0);
    assert_eq!((slab.borrow().cap, slab.borrow().used, slab.borrow().bt_stride_u64), (1, 1, 32));

    let s1 = alloc_handle(32);
    assert_eq!(s1.slot_idx, 1);
    assert_eq!((slab.borrow().cap, slab.borrow().used, slab.borrow().bt_stride_u64), (2, 2, 32));

    let s2 = alloc_handle(20);
    assert_eq!(s2.slot_idx, 2);
    assert_eq!((slab.borrow().cap, slab.borrow().used, slab.borrow().bt_stride_u64), (4, 3, 32));

    // Stamp distinct sentinels in each slot's `cs_buf` and `[full | win]` `bt_buf` slices.
    {
        let sb = slab.borrow();
        let cs = unsafe {
            std::slice::from_raw_parts_mut(
                sb.cs_buf.as_ref().unwrap().contents().as_ptr() as *mut f32,
                sb.cap * CS_STRIDE_F32,
            )
        };
        let bt = unsafe {
            std::slice::from_raw_parts_mut(
                sb.bt_buf.as_ref().unwrap().contents().as_ptr() as *mut u64,
                sb.cap * 2 * sb.bt_stride_u64,
            )
        };
        for slot in 0..3usize {
            cs[sb.cs_f32_base(slot) as usize] = (slot as f32) + 10.5;
            cs[sb.cs_f32_base(slot) as usize + CS_STRIDE_F32 - 1] = (slot as f32) + 99.25;
            for swa in [false, true] {
                let base = sb.bt_u64_base(slot, swa);
                for b in 0..32usize {
                    bt[base + b] = ((slot as u64 + 1) << 40) | ((u64::from(swa)) << 32) | (b as u64 + 1);
                }
            }
        }
    }

    // 2. Drop `s1` then `s0` -> pushed LIFO `[1, 0]`. Re-allocating pops `0` first and zero-fills it!
    drop(s1);
    drop(s0);
    assert_eq!(slab.borrow().free_slots, vec![1, 0]);

    let s0_reused = alloc_handle(16);
    assert_eq!(s0_reused.slot_idx, 0, "LIFO pop must reuse most recently dropped slot 0");
    assert_eq!(slab.borrow().used, 3, "reusing free slot must not bump high-water `used`");
    {
        let sb = slab.borrow();
        let cs = unsafe {
            std::slice::from_raw_parts(
                sb.cs_buf.as_ref().unwrap().contents().as_ptr() as *const f32,
                sb.cap * CS_STRIDE_F32,
            )
        };
        let bt = unsafe {
            std::slice::from_raw_parts(
                sb.bt_buf.as_ref().unwrap().contents().as_ptr() as *const u64,
                sb.cap * 2 * sb.bt_stride_u64,
            )
        };
        let cs0 = sb.cs_f32_base(0) as usize;
        assert!(cs[cs0..cs0 + CS_STRIDE_F32].iter().all(|&x| x == 0.0), "reused cs_buf slice must be zeroed");
        let bt0 = sb.bt_u64_base(0, false);
        assert!(bt[bt0..bt0 + 2 * sb.bt_stride_u64].iter().all(|&w| w == crate::blocks::CLEARED_GPU_TABLE_ENTRY), "reused bt_buf slice must be reset to CLEARED_GPU_TABLE_ENTRY");
        // Live slot 2 must remain completely intact!
        assert_eq!(cs[sb.cs_f32_base(2) as usize], 12.5);
    }

    // 3. Request `need_blocks = 65` -> `bt_stride_u64` grows `32 -> 96` (32-aligned) while reusing slot 1!
    let s1_grown = alloc_handle(65);
    assert_eq!(s1_grown.slot_idx, 1);
    assert_eq!((slab.borrow().cap, slab.borrow().used, slab.borrow().bt_stride_u64), (4, 3, 96));
    {
        let sb = slab.borrow();
        let bt = unsafe {
            std::slice::from_raw_parts(
                sb.bt_buf.as_ref().unwrap().contents().as_ptr() as *const u64,
                sb.cap * 2 * sb.bt_stride_u64,
            )
        };
        // Slot 2's `[full | win]` entries must be re-strided to `(2 * 2 + swa) * 96` with zeroed `[32..96)` tail!
        for swa in [false, true] {
            let base = sb.bt_u64_base(2, swa);
            assert_eq!(base, (2 * 2 + usize::from(swa)) * 96);
            for b in 0..32usize {
                let want = (3u64 << 40) | ((u64::from(swa)) << 32) | (b as u64 + 1);
                assert_eq!(bt[base + b], want, "slot 2 swa={swa} b={b} lost across stride growth");
            }
            assert!(bt[base + 32..base + 96].iter().all(|&w| w == crate::blocks::CLEARED_GPU_TABLE_ENTRY), "grown stride tail must be CLEARED_GPU_TABLE_ENTRY");
        }
        // Recycled slot 1 must be completely zeroed across all `2 * 96` u64 entries!
        let bt1 = sb.bt_u64_base(1, false);
        assert!(bt[bt1..bt1 + 2 * 96].iter().all(|&w| w == crate::blocks::CLEARED_GPU_TABLE_ENTRY));
    }
}

#[test]
fn kv_tier_file_opens_exclusively_rejects_symlinks_and_locks_fd() {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::PermissionsExt;

    let dir = std::path::PathBuf::from("target").join(format!("test-kv-tier-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // 1. Pre-existing regular file must never be truncated or overwritten.
    let existing = dir.join("existing.bin");
    std::fs::write(&existing, b"keep-me").unwrap();
    assert_eq!(create_locked_kv_tier_file_at(&existing).unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep-me");

    // 2. Planted symlink (even dangling) must never be followed.
    let victim = dir.join("victim.txt");
    std::fs::write(&victim, b"untouched").unwrap();
    let symlink = dir.join("planted-link.bin");
    std::os::unix::fs::symlink(&victim, &symlink).unwrap();
    assert!(create_locked_kv_tier_file_at(&symlink).is_err());
    assert_eq!(std::fs::read(&victim).unwrap(), b"untouched");

    let dangling = dir.join("dangling-link.bin");
    std::os::unix::fs::symlink(dir.join("nonexistent-target"), &dangling).unwrap();
    assert!(create_locked_kv_tier_file_at(&dangling).is_err());
    assert!(!dir.join("nonexistent-target").exists());

    // 3. Fresh path opens with mode 0600 and holds LOCK_EX; open_exclusive_kv_tier_file unlinks immediately.
    let fresh = dir.join("fresh.bin");
    let f1 = create_locked_kv_tier_file_at(&fresh).unwrap();
    let mode = f1.metadata().unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let f2 = std::fs::OpenOptions::new().read(true).write(true).open(&fresh).unwrap();
    let lock_rc = unsafe { libc::flock(f2.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_ne!(lock_rc, 0, "second open description must observe LOCK_EX held by tier file");
    drop(f2);
    drop(f1);

    let unlinked = open_exclusive_kv_tier_file(&dir).unwrap();
    assert_eq!(unlinked.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    drop(unlinked);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn snapshot_validation_checks_conv_and_window_buffer_fit() {
    assert!(validate_snapshot_lengths(CS_STRIDE_BYTES, CS_STRIDE_BYTES, 4096, 4096).is_ok());
    assert!(validate_snapshot_lengths(CS_STRIDE_BYTES - 4, CS_STRIDE_BYTES, 4096, 4096).is_err());
    assert!(validate_snapshot_lengths(CS_STRIDE_BYTES, CS_STRIDE_BYTES, 8192, 4096).is_err());
    assert!(validate_snapshot_lengths(CS_STRIDE_BYTES, CS_STRIDE_BYTES, 2048, 4096).is_err());
}

#[test]
fn dspark_verify_capacity_guard_bounds_lg_rows_and_struct_layouts() {
    assert_eq!(std::mem::size_of::<DSparkSpanDesc>(), 48);
    assert_eq!(std::mem::size_of::<DSparkRejectArgs>(), 32);

    let b128 = BArena::new(0, 4096, true, 2, N_VOCAB_PAD);
    assert_eq!(b128.lg_rows, MAX_SPANS as u32);
    // 16 spans x (1 + 7) = 128 rows fits exactly; 17 spans x (1 + 7) = 136 > 128 fails cleanly.
    assert_eq!(16 * (1 + 7), b128.lg_rows as usize);
    assert!(17 * (1 + 7) > b128.lg_rows as usize);

    let b64 = BArena::new(0, 64, true, 2, N_VOCAB_PAD);
    assert_eq!(b64.lg_rows, 64);
    assert!(9 * (1 + 7) > b64.lg_rows as usize);
}

/// M5 GPU verification gate for `t-later-audit-dspark-guards`:
/// (a) `try_encode_verify` / `verify_speculative_gpu` arena row overflow (`> lg_rows`) returns `Err` cleanly with zero mutation,
/// (b) `verify_speculative_gpu` at `temp = 0.0` (mixed partial/rejected/full drafts) matches target greedy argmax token-for-token,
/// (c) `k_dspark_reject` on GPU at `temp = 1.0` matches exact target softmax distribution (`TVD < 0.02` over 8,192 GPU draws)
///     even when draft logits are heavily skewed away from target!
#[test]
#[ignore = "requires local Inkling-2-Mini GGUF model on Metal GPU"]
fn dspark_gpu_guards_and_rejection_sampling_gate() {
    let model = std::env::var("INKY_TEST_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.lmstudio/models/lmstudio-community/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q4_K_M.gguf")
    });
    assert!(std::path::Path::new(&model).exists(), "model not found at {model}");

    let eng = Engine::load(&model, 512);
    eng.ensure_batch(eng.chunk);

    // (a) Arena row-capacity overflow (`17 * (1 + 7) = 136 > 128 = lg_rows` and `1 * (1 + 128) = 129 > 128`)
    //     must return `Err` cleanly without panicking or mutating `verify_keep`.
    assert!(eng.check_verify_capacity(16, 7).is_ok());
    let over_err = eng.check_verify_capacity(17, 7).expect_err("136 rows > lg_rows=128 must fail");
    assert!(over_err.contains("lg_rows"), "got: {over_err}");

    let overflow_lane = Lane::new(&eng);
    eng.on(&overflow_lane).reserve(256, true);
    let cb = eng.gpu.cmd();
    let try_err = eng
        .try_encode_verify(&cb, &[Span { lane: &overflow_lane, p0: 0, t: 129 }], 128)
        .expect_err("129 rows > lg_rows=128 must return Err");
    assert!(try_err.contains("lg_rows"), "got: {try_err}");
    assert!(eng.verify_keep.borrow().is_none(), "failed try_encode_verify must not populate verify_keep");

    // (b) Greedy `temp = 0.0` speculative verification on real model matches cold target argmax token-for-token!
    let prompt: Vec<u32> = vec![101, 2054, 2003, 1996, 3007, 1997, 2605, 1029];
    let total_gen = 12usize;

    // Cold single-token target reference:
    let ref_lane = Lane::new(&eng);
    let ref_run = eng.on(&ref_lane);
    ref_run.reserve(prompt.len() + total_gen + 8, true);
    ref_run.set_tokens(0, &prompt);
    {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &ref_lane, p0: 0, t: prompt.len() }], true, true);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let mut target_greedy = vec![ref_run.token_at(prompt.len())];
    while target_greedy.len() < total_gen {
        let pos = prompt.len() + target_greedy.len() - 1;
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &ref_lane, p0: pos, t: 1 }], true, true);
        cb.commit();
        cb.waitUntilCompleted();
        target_greedy.push(ref_run.token_at(pos + 1));
    }

    // Speculative verify lane (`k = 4` drafts per round: mix of exact hits, partial hits, and immediate misses!):
    let spec_lane = Lane::new(&eng);
    let spec_run = eng.on(&spec_lane);
    spec_run.reserve(prompt.len() + total_gen + 16, true);
    spec_run.set_tokens(0, &prompt);
    {
        let cb = eng.gpu.cmd();
        eng.encode_spans(&cb, &[Span { lane: &spec_lane, p0: 0, t: prompt.len() }], true, true);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let mut spec_emitted = vec![spec_run.token_at(prompt.len())];
    let mut round = 0usize;
    while spec_emitted.len() < total_gen {
        let p0 = prompt.len() + spec_emitted.len() - 1;
        // Construct 4 draft tokens with deliberate partial matches and mismatches across rounds:
        let k = 4usize;
        let mut drafts = [0u32; 4];
        for (d, draft_slot) in drafts.iter_mut().enumerate().take(k) {
            let want_idx = spec_emitted.len() + d;
            let true_tok = target_greedy.get(want_idx).copied().unwrap_or(42);
            *draft_slot = match (round, d) {
                (0, 0..=1) => true_tok, // accept 2, reject at d=2
                (1, _) => true_tok ^ 7, // reject immediately at d=0
                (2, _) => true_tok,     // accept all 4 + bonus token!
                (_, 0) => true_tok,     // accept 1, reject at d=1
                _ => true_tok ^ 3,
            };
        }
        let outs = eng
            .verify_speculative_gpu(
                &[SpeculativeVerifySpan {
                    lane: &spec_lane,
                    p0,
                    draft_toks: &drafts,
                    draft_logits: &[],
                    draft_stride: eng.n_vocab,
                    seed: 12345 + round as u64,
                    ban_tok: None,
                }],
                0.0,
                eng.softcap,
            )
            .expect("greedy verify_speculative_gpu");
        assert_eq!(outs.len(), 1);
        for &tok in &outs[0].emitted {
            if spec_emitted.len() < total_gen {
                spec_emitted.push(tok);
            }
        }
        round += 1;
    }
    assert_eq!(
        spec_emitted, target_greedy,
        "greedy temp=0 verify_speculative_gpu must match target argmax token-for-token"
    );

    // (c) Stochastic `temp = 1.0` empirical distribution test on M5 GPU (`k_dspark_reject`):
    //     Construct a vocabulary distribution where 6 active tokens carry ~99.99% of mass in both target and
    //     heavily-misaligned draft (`TVD(p_target, p_draft) > 0.55`), draw draft tokens from `p_draft`, run
    //     8,192 independent GPU `k_dspark_reject` trials in batched waves of 64 spans, and verify the empirical
    //     output distribution matches `p_target` (`TVD < 0.025`)!
    let nv = eng.n_vocab;
    let stride = eng.vocab_pad();
    let hot_ids: [usize; 6] = [5, 19, 1024, 4099, 65537 % nv, nv - 3];
    let hot_t_raw: [f32; 6] = [4.2, 3.6, 3.1, 2.4, 1.9, 1.2];
    let hot_q_raw: [f32; 6] = [1.1, 4.4, 1.5, 3.9, 3.7, 0.8]; // strongly skewed away from target!

    let mut z_t = vec![-30.0f32; stride];
    let mut z_q = vec![-30.0f32; nv];
    for (&id, (&lt, &lq)) in hot_ids.iter().zip(hot_t_raw.iter().zip(hot_q_raw.iter())) {
        z_t[id] = lt;
        z_q[id] = lq;
    }
    let softcap_fn = |x: f32| -> f32 {
        if eng.softcap > 0.0 { eng.softcap * (x / eng.softcap).tanh() } else { x }
    };
    let softmax = |z: &[f32]| -> Vec<f64> {
        let capped: Vec<f64> = z[..nv].iter().map(|&x| softcap_fn(x) as f64).collect();
        let mx = capped.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = capped.iter().map(|&x| (x - mx).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.into_iter().map(|e| e / s).collect()
    };
    let p_t = softmax(&z_t);
    let p_q = softmax(&z_q);
    let draft_tvd: f64 = 0.5 * p_t.iter().zip(&p_q).map(|(a, b)| (a - b).abs()).sum::<f64>();
    assert!(draft_tvd > 0.45, "draft distribution must be substantially skewed from target (tvd={draft_tvd})");

    let batch = 64usize;
    let waves = 128usize; // 64 * 128 = 8,192 independent GPU rejection-sampling trials
    let total_trials = batch * waves;
    // Each span with k=1 uses 2 target rows ([0] verified against draft, [1] bonus row):
    let mut target_arena = Vec::with_capacity(batch * 2 * stride);
    let mut draft_arena = Vec::with_capacity(batch * nv);
    for _ in 0..batch {
        target_arena.extend_from_slice(&z_t);
        target_arena.extend_from_slice(&z_t);
        draft_arena.extend_from_slice(&z_q);
    }
    let target_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(target_arena.as_ptr() as *const u8, target_arena.len() * 4)
    };
    let target_buf = eng.gpu.buffer_with(target_bytes);

    // Deterministic PCG/SplitMix64 sampler on host to propose `x ~ p_draft` for each trial:
    let mut rng_state = 0x1234_5678_9ABC_DEF0u64;
    let mut next_f64 = || -> f64 {
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    };
    let mut cdf_q = Vec::with_capacity(nv);
    let mut acc = 0.0f64;
    for &p in &p_q {
        acc += p;
        cdf_q.push(acc);
    }

    let mut counts = vec![0usize; nv];
    for w in 0..waves {
        let mut draft_toks = Vec::with_capacity(batch);
        let mut seeds = Vec::with_capacity(batch);
        for b_idx in 0..batch {
            let u = next_f64();
            let tok = cdf_q.partition_point(|&c| c < u).min(nv - 1) as u32;
            draft_toks.push(tok);
            let trial_id = (w * batch + b_idx) as u64;
            seeds.push((0xCAFE_F00D_0000_0000u64 ^ trial_id.wrapping_mul(0x9E37_79B9_7F4A_7C15), trial_id as u32, None));
        }
        let outs = eng
            .reject_sample_gpu(
                &target_buf,
                0,
                stride,
                nv,
                1,
                &draft_toks,
                &draft_arena,
                &seeds,
                1.0,
                eng.softcap,
                eng.softcap,
            )
            .expect("temp=1.0 reject_sample_gpu");
        for o in outs {
            counts[o.emitted[0] as usize] += 1;
        }
    }
    let emp_tvd: f64 = 0.5
        * counts
            .iter()
            .zip(&p_t)
            .map(|(&c, &pt)| ((c as f64) / (total_trials as f64) - pt).abs())
            .sum::<f64>();
    eprintln!(
        "[dspark temp=1.0 gate] trials={total_trials} draft_tvd={draft_tvd:.4} empirical_tvd={emp_tvd:.4}"
    );
    assert!(
        emp_tvd < 0.025,
        "GPU rejection sampling empirical TVD ({emp_tvd:.4}) must match p_target closely (draft TVD was {draft_tvd:.4})"
    );
}
