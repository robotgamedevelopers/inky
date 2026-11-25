use super::model::{check_vocab_lengths, configured_prefill_chunk};
use super::*;

#[test]
fn disk_staging_holds_each_fixed_split_through_a_resize_interval() {
    for max_ctx in [65_536, 131_072, 131_073, 262_145] {
        // A small cache forces staging to grow to fit a complete Full split.
        // The second interval straddles the old 32,768-token failure.
        for start in [64usize, 32_760] {
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

    let private = Layer2::new(&eng, 2);
    let long = private.new_lane(&long_tokens, 128);
    let short = private.new_lane(&short_tokens, 128);
    private.prefill(&[(short, 0, prefix)]);
    private.prefill(&[(long, 0, long_tokens.len()), (short, prefix, append)]);
    let got = private.logits(1);
    assert_eq!(got.len(), want.len());
    assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
            "private multi-lane logits must match the independent paged lane");

    // Inspect both K and V after the final private sliding-window layer.
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
            assert!(bytes[first..live_end].iter().any(|&b| b != 0));
            assert!(bytes[live_end..tile_end].iter().all(|&b| b == 0),
                    "masked scratch rows still hold the previous private lane's values");
        }
    }
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
fn projection_segments_preserve_q8_attention_geometry() {
    let weights = QW { ty: T_Q8_0, k: E as u32, rows: (NH * HD) as u32,
        row_bytes: (E / 32 * 34) as u32, ..Default::default() };
    let seg = Engine::seg(&weights, 12, 34);
    assert_eq!((seg.wtype, seg.k, seg.nrows, seg.row_bytes),
               (T_Q8_0, weights.k, weights.rows, weights.row_bytes));
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
    let gpu = Gpu::new(&src, &["k_topk_b"]);
    let bias = gpu.buffer(NEXP * 4);
    let act = gpu.buffer((NEXP + 2 * TOPK) * 4);
    unsafe {
        let bias = std::slice::from_raw_parts_mut(bias.contents().as_ptr().cast::<f32>(), NEXP);
        let scores = std::slice::from_raw_parts_mut(act.contents().as_ptr().cast::<f32>(), NEXP);
        bias.fill(0.0);
        scores.fill(f32::NAN);
    }

    let args = [0, 0, NEXP as u32, 1, 1.0f32.to_bits(), 0, 0, 0];
    let cb = gpu.cmd();
    let e = Enc::new(&cb);
    e.buf(&bias, 0, 1);
    e.buf(&act, 0, 2);
    e.bytes(&args, 5);
    e.pso(&gpu.pso("k_topk_b"));
    e.dispatch((1, 1, 1), 32);
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
    assert!(cb.error().is_none());
    let out = unsafe {
        std::slice::from_raw_parts(act.contents().as_ptr().cast::<f32>().add(NEXP), 2 * TOPK)
    };
    for (i, &id) in out[..TOPK].iter().enumerate() { assert_eq!(id, i as f32); }
    assert_eq!(out[TOPK..], [0.0; TOPK]);
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
    assert_eq!(std::mem::size_of::<MoeArgs>(), 32 * 4);
    assert_eq!(std::mem::size_of::<MoeGemmArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<MoeGemmHArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<SampArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<LoopCdArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<DecodeSpanDesc>(), 16 * 4);
    assert_eq!(std::mem::align_of::<DecodeSpanDesc>(), 4);
    assert_eq!(std::mem::size_of::<EmbedDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<CombineArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<CombineDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<AttnBlkDecArgs>(), 16 * 4);
    assert_eq!(std::mem::size_of::<DecAttnGroup>(), 6 * 4);
    assert_eq!(std::mem::align_of::<DecAttnGroup>(), 4);
    assert_eq!(std::mem::size_of::<KvPrepDecArgs>(), 20 * 4);
    assert_eq!(std::mem::size_of::<AttnGrpDecArgs>(), 12 * 4);
    assert_eq!(std::mem::size_of::<PostConvDecArgs>(), 8 * 4);
    assert_eq!(std::mem::size_of::<TopkRoute3Args>(), 16 * 4);

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

    // Verify `CombineArgs`, `CombineDecArgs`, `AttnBlkDecArgs`, `PostConvDecArgs`, and `TopkRoute3Args` offsets match Metal structs.
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
    assert_eq!(std::mem::offset_of!(AttnBlkDecArgs, desc_off), 0);
    assert_eq!(std::mem::offset_of!(AttnBlkDecArgs, swa), 12);
    assert_eq!(std::mem::offset_of!(AttnBlkDecArgs, kdyn_off), 36);
    assert_eq!(std::mem::offset_of!(AttnBlkDecArgs, vdyn_off), 40);
    assert_eq!(std::mem::offset_of!(AttnBlkDecArgs, relbias_w), 60);
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
    assert!(MAX_SPANS * 512 <= NH * MAX_SPLIT * (HD + 2));

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
            assert!(nsplit >= 1 && nsplit <= MAX_SPLIT);
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

#[test]
fn metal_shader_declarations_and_compiler_validation() {
    let kernels_src = include_str!("../kernels.metal");
    let prefill_src = include_str!("../prefill.metal");
    let bench_src = include_str!("../bench_kernels.metal");
    let full_src = format!("{kernels_src}\n{prefill_src}\n{bench_src}");

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
        "struct AttnBlkDecArgs",
        "struct DecAttnGroup",
        "struct KvPrepDecArgs",
        "struct AttnGrpDecArgs",
        "struct PostConvDecArgs",
        "struct TopkRoute3Args",
        "struct LoopCdArgs",
        "kernel void k_embed_dec(",
        "kernel void k_attn_blk_dec(",
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

    // Compile via `xcrun -sdk macosx metal`: full shader when `-std=metal4.0` is supported,
    // otherwise compile `kernels.metal` + `k_moe_topk_route3` + `k_postconv_state_dec` under `-std=metal3.1` if available.
    let compile_metal = |std_flag: &str, src: &str, tag: &str| -> Result<(), String> {
        let pid = std::process::id();
        let tmp_in = std::env::temp_dir().join(format!("inky_check_{pid}_{tag}.metal"));
        let tmp_out = std::env::temp_dir().join(format!("inky_check_{pid}_{tag}.air"));
        std::fs::write(&tmp_in, src).map_err(|e| format!("write temp shader: {e}"))?;
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

    if compile_metal("-std=metal4.0", "#include <metal_stdlib>\nkernel void probe() {}\n", "probe4").is_ok() {
        compile_metal("-std=metal4.0", &full_src, "full_fp16")
            .unwrap_or_else(|e| panic!("Metal 4.0 fp16 compile failed:\n{e}"));
        let kv8_src = format!("#define KV8 1\n{full_src}");
        compile_metal("-std=metal4.0", &kv8_src, "full_kv8")
            .unwrap_or_else(|e| panic!("Metal 4.0 KV8 compile failed:\n{e}"));
    } else if compile_metal("-std=metal3.1", "#include <metal_stdlib>\nkernel void probe() {}\n", "probe3").is_ok() {
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
        compile_metal("-std=metal3.1", &dec_src, "dec_fp16")
            .unwrap_or_else(|e| panic!("Metal 3.1 decode kernels (fp16) compile failed:\n{e}"));
        let dec_kv8_src = format!("#define KV8 1\n{dec_src}");
        compile_metal("-std=metal3.1", &dec_kv8_src, "dec_kv8")
            .unwrap_or_else(|e| panic!("Metal 3.1 decode kernels (KV8) compile failed:\n{e}"));
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
        for k in 0..TOPK {
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
            out[k] = pick;
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
            let block_start = ip % BLOCK == 0;
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

    // Step across 64 consecutive decode positions `pos = 64..128` (the entire second 64-token block):
    // - `pos == 64` allocates block 1 -> `dirty_range == Some((1, 1))` -> writes `(1, 1)` table words.
    // - `pos == 65..128` (63 steady-state decode steps) -> `dirty_range == None` -> writes `(0, 0)` words!
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

    // Sentinels in `[2..32)` are still completely untouched after all 64 decode steps.
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

    // Next block boundary at `pos = 128` allocates block 2 -> writes `(1, 1)` table words.
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
