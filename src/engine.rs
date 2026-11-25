//! Inkling 2 Mini decode engine: weight binding, GPU memory layout and the per-token dispatch schedule.

use crate::gguf::*;
use crate::metal::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice};
use std::cell::{Cell, RefCell};
use std::io;
use std::os::unix::fs::FileExt;
use std::rc::Rc;

pub const E: usize = 2304;
pub const NH: usize = 24;
pub const NKV: usize = 8;
pub const HD: usize = 128;
pub const NREL: usize = 16;
pub const EXT: usize = 1024;
pub const NEXP: usize = 128;
pub const TOPK: usize = 8;
pub const FF: usize = 1152;
const N_VOCAB_PAD: usize = 201_024;
const CONV_TAPS: usize = 4;
const CTRL_RANK: usize = 64;
pub const NCH: usize = 2 * NKV * HD + 2 * E; // conv-state channels per slot
pub const N_PHYS: usize = 24;
pub const LOOP_START: usize = 6;
pub const LOOP_END: usize = 18;
pub const N_PASS: usize = 3;
pub const N_SLOT: usize = N_PHYS + (N_PASS - 1) * (LOOP_END - LOOP_START);
const CS_STRIDE_F32: usize = N_SLOT * NCH * 3;
const CS_STRIDE_BYTES: usize = CS_STRIDE_F32 * 4;
/// Align each blocked lane's `full` and `win` GPU block tables to 32 `u64`s (256 bytes)
/// so `setBuffer_offset_atIndex` is always 256-byte aligned on Apple Silicon.
const BT_STRIDE_ALIGN_U64: usize = 32;
const _: () = assert!(CS_STRIDE_BYTES % 256 == 0);
const _: () = assert!((BT_STRIDE_ALIGN_U64 * 8) % 256 == 0);
const _: () = assert!(crate::blocks::NKV == NKV && crate::blocks::HD == HD);
const _: () = assert!(crate::blocks::WINDOW_PASSES + crate::blocks::FULL_PASSES == N_SLOT);
pub const BLOCK: usize = 6;
pub const KPS_CHUNK: usize = 32;
/// keys in front of the prefill K/V scratch that the 64-key attention chunks may read (always masked)
const KGUARD: usize = 64;
/// k_attn_h's key tile (AK in prefill.metal): a prefill span's attention reads keys up to the next multiple of this
const ATT_AK: usize = 64;
/// most spans one step can carry (each has a logits row in the batch arena)
pub const MAX_SPANS: usize = 128;
const MAX_PREFILL_CHUNK: usize = 4096;

fn configured_prefill_chunk(value: Result<String, std::env::VarError>, context: usize) -> Result<usize, &'static str> {
    let chunk = match value {
        Err(std::env::VarError::NotPresent) => Some(MAX_PREFILL_CHUNK),
        Ok(value) => value.parse::<usize>().ok(),
        Err(std::env::VarError::NotUnicode(_)) => None,
    };
    chunk.filter(|&n| (32..=MAX_PREFILL_CHUNK).contains(&n) && n % 32 == 0)
        .map(|n| n.min(context))
        .ok_or("INK_CHUNK must be a multiple of 32 between 32 and 4096")
}

/// process-wide default for `Engine::load`: keep only part of the (exact, fp16) K/V cache in memory while generating and
/// stream the older part from a temporary file on every token (`--kv-disk`)
pub static KV_DISK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// one position of one slot in the K/V file: K then V, 8 groups of 128 fp16 each
const REC: usize = 2 * NKV * HD * 2;

/// The disk tier of the K/V cache (`--kv-disk`). While generating, the two cache buffers are rings holding only the newest
/// positions ("hot"); every older position a layer still attends to is in `file` and is read into `stg` for each token.
struct Tier {
    file: std::fs::File,
    on: Cell<bool>,
    /// per kind [sliding-window, full]: oldest position the hot ring holds / positions below this are in the file
    lo: [Cell<usize>; 2], cold: [Cell<usize>; 2],
    /// positions below this were written to the file earlier in this session (cached K/V never change, so a later tiering
    /// after a prefill only writes what is new)
    filed: [Cell<usize>; 2],
    /// position at which the sizes are reviewed again (needs an idle GPU)
    next: Cell<usize>,
    /// smallest context that is tiered
    min_n: usize,
    /// staging for the file reads, in two halves: the GPU works on one while the next reads land in the other
    stg: RefCell<Buf>, half: Cell<usize>, pending: RefCell<[Option<CmdBuf>; 2]>,
    /// this step's file reads in execution order, and the round (staging half fill) that is loaded
    plan: RefCell<Vec<Chunk>>, execs: RefCell<Vec<ExecPlan>>, loaded: Cell<usize>,
    pub read_bytes: Cell<u64>, pub read_time: Cell<std::time::Duration>, unaligned: Cell<bool>,
}
#[derive(Clone, Copy)]
struct Chunk { exec: usize, kind: usize, i: usize, c0: usize, c1: usize, round: usize, off: usize, sp0: usize, ns: usize, kps: usize, rel_split: bool }
/// how one layer execution's keys are divided: file chunks cover the splits before `b`, the hot ring the `ns` splits from `sp0`
#[derive(Clone, Copy)]
struct ExecPlan { b: usize, kps: usize, nsplit: usize, sp0: usize, ns: usize, rel_split: bool }

/// the encoder a decode step is being written to; with the disk tier a step spans several command buffers
pub struct Cx { e: Enc, cb: Option<CmdBuf>, prior: Vec<CmdBuf> }

/// All submitted command buffers for a single decode step.
pub struct DecodeStep { prior: Vec<CmdBuf>, last: CmdBuf }

impl DecodeStep {
    pub fn wait(&self, pos: usize) -> Result<(), String> {
        wait_all(self.prior.iter().chain([&self.last]), |cb| wait_completed(cb, "decode", pos))
    }
}

/// process-wide default for `Engine::load`: the opt-in 8-bit K/V cache (`--kv8`)
pub static KV8: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// initial capacity (positions) of the K/V buffers; they grow with the context
const KV_CAP0: usize = 64;
/// MoE tile classes of the fp16 prefill path: (kernel, max tokens per tile, threads per threadgroup)
pub const NTC: usize = 2;
const MOE_CLS: [(&str, usize, usize); NTC] = [("k_moe_sg8", 128, 256), ("k_moe_sg4", 64, 128)];
pub const MAX_SPLIT: usize = 256;
/// Maximum decode lanes grouped onto one shared-KV tile read in `k_attn_grp_dec`.
pub const DEC_GRP_MAX: usize = 8;
/// Maximum fixed 64-token splits in a sliding-window (`EXT = 1024`) decode step (`1024 / 64 + 1 = 17`).
pub const MAX_SWA_SPLIT: usize = EXT / crate::blocks::BLOCK + 1;

/// kernel classes, for skip-mask profiling
pub const CLS_NAMES: &[&str] = &["embed", "mix_attn", "qkv", "attn", "combine", "wo", "mix_ffn", "router", "moe_gu", "moe_down",
                                 "ctrl_down", "ctrl_apply", "out_mix", "lm_head", "argmax",
                                 "kvprep", "wo_post", "gather", "moe_comb", "ffn_post", "ctrl_norm", "topk_route"];
const C_KVPREP: u32 = 15; const C_WOPOST: u32 = 16; const C_GATHER: u32 = 17; const C_COMB2: u32 = 18; const C_FFNPOST: u32 = 19;
const C_CTRLN: u32 = 20; const C_TOPK: u32 = 21;
const C_EMBED: u32 = 0; const C_MIXA: u32 = 1; const C_QKV: u32 = 2; const C_ATTN: u32 = 3; const C_COMB: u32 = 4; const C_WO: u32 = 5;
const C_MIXF: u32 = 6; const C_ROUTER: u32 = 7; const C_GU: u32 = 8; const C_DOWN: u32 = 9; const C_CTRLD: u32 = 10; const C_CTRLA: u32 = 11;
const C_OMIX: u32 = 12; const C_LMHEAD: u32 = 13; const C_ARGMAX: u32 = 14;

pub const KERNELS: &[&str] = &[
    "k_embed", "k_embed_dec", "k_mix", "k_mv", "k_mvb", "k_epi", "k_ctrl_down", "k_ctrl_apply", "k_attn", "k_attn_blk", "k_attn_blk_dec", "k_kvprep_dec", "k_attn_dec_solo", "k_attn_grp_dec", "k_attn_combine", "k_attn_combine_dec",
    "k_moe_gu", "k_moe_down", "k_argmax1", "k_argmax2", "k_gumbel", "k_loopcd_blend",
    "k_topk", "k_norm",
    "k_gemm", "k_moe_gemm", "k_gemm_t", "k_moe_gemm_t", "k_topk_b", "k_moe_route", "k_moe_comb_b", "k_postconv_b", "k_postconv_state_dec", "k_state_upd", "k_state_keep", "k_kvprep_b", "k_kvprep_b_blk", "k_attn_b", "k_attn_b_blk", "k_attn_bt", "k_attn_bt_blk", "k_ctrl_apply_b", "k_tohalf", "k_gemm_sg8", "k_gemm_sg4", "k_moe_sg8", "k_moe_sg4", "k_ctrl_gf", "k_moe_route3", "k_moe_topk_route3", "k_moe_comb_h", "k_attn_h", "k_attn_h_blk", "k_kvcopy", "k_devcopy", "k_zero", "k_mix_b_1_1", "k_mix_b_2_1", "k_mix_b_3_1", "k_mix_b_4_1", "k_mix_b_5_1", "k_mix_b_6_1", "k_mix_b_1_3", "k_mix_b_2_3", "k_mix_b_3_3", "k_mix_b_4_3", "k_mix_b_5_3", "k_mix_b_6_3", "k_mix_b_1_4", "k_mix_b_2_4", "k_mix_b_3_4", "k_mix_b_4_4", "k_mix_b_5_4", "k_mix_b_6_4", "k_mix_b_id",
    "k_touch",
];

// ----------------------------------------------------------------------------- kernel arg structs (mirror kernels.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct EmbedArgs { emb_lo: u32, emb_hi: u32, row_bytes: u32, norm_off: u32, out0: u32, out1: u32, tok_index: u32, eps: f32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct DecodeSpanDesc {
    tok_id: i32,
    pos: u32,
    cs_base: u32,
    bt_full_base: u32,
    bt_win_base: u32,
    part_off: u32,
    swa_nkeys: u32,
    swa_first_pos: u32,
    swa_kps: u32,
    swa_nsplit: u32,
    full_nkeys: u32,
    full_kps: u32,
    full_nsplit: u32,
    full_tau: f32,
    swa_tau: f32,
    pad1: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct EmbedDecArgs { emb_lo: u32, emb_hi: u32, row_bytes: u32, norm_off: u32, out_off: u32, desc_off: u32, eps: f32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CombineDecArgs { desc_off: u32, gate_off: u32, out_off: u32, swa: u32, hout: u32, pad: [u32; 3] }
#[allow(dead_code)]
#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnBlkDecArgs {
    desc_off: u32, pass_idx: u32, cs_slot: u32, swa: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32,
    sink_off: u32, kdyn_off: i32, vdyn_off: i32, kconv_w: u32,
    vconv_w: u32, qclip_w: u32, relclip_w: u32, relbias_w: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DecAttnGroup {
    pub abs_sp: u32,
    pub count: u32,
    pub span_idx: [u16; DEC_GRP_MAX],
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct KvPrepDecArgs {
    desc_off: u32, pass_idx: u32, cs_slot: u32, swa: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32,
    qp_off: u32, rp_off: u32, kv_cur_off: u32, kdyn_off: i32,
    vdyn_off: i32, kconv_w: u32, vconv_w: u32, qclip_w: u32,
    relclip_w: u32, pad0: u32, pad1: u32, pad2: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnGrpDecArgs {
    desc_off: u32, grp_off: u32, qp_off: u32, rp_off: u32,
    kv_cur_off: u32, sink_off: u32, relbias_w: u32, pass_idx: u32,
    kps: u32, swa: u32, pad0: u32, pad1: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct PostConvDecArgs { desc_off: u32, y_off: u32, res_off: u32, out_off: u32, cs_slot_off: u32, convw_off: u32, has_res: u32, no_state_write: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct MixJob { n_s: u32, n_q: u32, s_off: [u32; 6], q_off: i32, out_off: [u32; 4], norm_off: [i32; 4], eps: f32, tstride: u32, ncopy: [[i32; 3]; 4], ntok: u32,
                 hcopy: [[i32; 3]; 4], hout: [i32; 4], hstride: u32, nof32: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct Seg {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32,
    nrows: u32, k: u32, tg_start: u32, in_off: u32,
    in2_off: i32, norm_off: i32, ss_off: i32, ss_n: i32,
    out_off: u32, epi: u32, res_off: i32, cs_off: u32,
    convw_off: u32, pad: [u32; 3],
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MvArgs { nseg: u32, eps: f32, post: u32, ctr: u32, bias_off: u32, sel_off: u32, wscale: f32, pad: u32, seg: [Seg; 12] }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlJob { u_off: u32, m_off: u32, d_off: u32, out_off: u32, gain_off: u32, forget_off: u32, res_out: u32, ss_off: u32, norm_w: i32, ctr: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlArgs { n_c: u32, tstride: u32, pad: [u32; 2], job: [CtrlJob; 4] }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnArgs {
    pos: u32, nkeys: u32, first_pos: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, cs_off: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32, sink_off: u32, kdyn_off: i32, vdyn_off: i32, part_off: u32,
    kconv_w: u32, vconv_w: u32, qclip_w: u32, relclip_w: u32, relbias_w: u32, tau: f32, kps: u32, nsplit: u32,
    ctr: u32, gate_off: u32, out_off: u32, pad: u32,
    sp0: u32, noown: u32, gstr: u32, istr: u32, idx0: u32, lastbias: f32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct DevCopyArgs { src: u32, dst: u32, n: u32, pad: u32 }
/// dev (key eviction experiments): the far keys of the full-attention slots replaced by a selection (`rows` per group,
/// the last one a summary of the evicted keys with score bias `lastbias`); keys from position `r0` on are untouched
pub struct Prune { bag: Buf, rows: usize, r0: usize, lastbias: f32, use_sum: bool }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CombineArgs { part_off: u32, nsplit: u32, gate_off: u32, out_off: u32, hout: u32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct EpiArgs { out_off: u32, res_off: i32, cs_off: u32, convw_off: u32, n: u32, pad: [u32; 3] }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeArgs {
    logits_off: u32, bias_off: u32, sel_off: u32, x_off: u32,
    norm_off: i32, ss_off: i32, ss_n: i32, eps: f32,
    gu_lo: u32, gu_hi: u32, gu_expert_bytes: u32, gu_row_bytes: u32,
    act_off: u32, wscale: f32, beta: f32, lbeta: f32,
    dn_lo: u32, dn_hi: u32, dn_expert_bytes: u32, dn_row_bytes: u32,
    dn_type: u32, out_off: u32, res_off: i32, cs_off: u32,
    convw_off: u32, gu_type: u32, gu_gs_lo: u32, gu_gs_hi: u32,
    dn_gs_lo: u32, dn_gs_hi: u32, no_state_write: u32, pad1: u32,
}

// ---- prefill arg structs (mirror prefill.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GSeg { w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, tile_start: u32, in_off: u32, out_off: u32, ldo: u32, wsrc: u32, in2: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ToHalfArgs { src_off: u32, dst_off: u32, k: u32, rows: u32, perm_off: u32, gather: u32, src_stride: u32, dst_stride: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct HSeg { w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, tile_start: u32, xh_off: u32, out_off: u32, ldo: u32, wsrc: u32, hout: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GemmHArgs { nseg: u32, t: u32, pad: [u32; 2], seg: [HSeg; 10] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeGemmHArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, xh_off: u32,
    tiles_off: u32, ntiles_off: u32, out_off: u32, ldo: u32, situ: u32, aw_off: u32, beta: f32, lbeta: f32,
    gs_lo: u32, gs_hi: u32, pad0: u32, pad1: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlBJob { u_off: u32, m_off: u32, g_off: u32, f_off: u32, out_off: u32, norm_w: u32, h_off: i32, nof32: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlGfJob { gain_w: u32, forget_w: u32, gh_off: u32, u_off: u32, m_off: u32, raw_off: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlGfArgs { t: u32, njobs: u32, pad: [u32; 2], job: [CtrlGfJob; 4] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CtrlBArgs { n: u32, pad: [u32; 3], job: [CtrlBJob; 4] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct GemmArgs { nseg: u32, t: u32, pad: [u32; 2], seg: [GSeg; 10] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeGemmArgs {
    w_lo: u32, w_hi: u32, expert_bytes: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, x_off: u32,
    gather: u32, perm_off: u32, tiles_off: u32, ntiles_off: u32, out_off: u32, ldo: u32, situ: u32, pad: u32,
    beta: f32, lbeta: f32, gs_lo: u32, gs_hi: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct TopkBArgs { logits_off: u32, bias_off: u32, sel_off: u32, t: u32, wscale: f32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct RouteArgs { sel_off: u32, t: u32, perm_off: u32, inv_off: u32, tiles_off: u32, ntiles_off: u32, tile: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Route3Args { sel_off: u32, t: u32, perm_off: u32, inv_off: u32, aw_off: u32, ind_off: u32, gu_y: u32, dn_y: u32, tiles_off: [u32; NTC], pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct TopkRoute3Args {
    logits_off: u32, bias_off: u32, sel_off: u32, t: u32,
    wscale: f32, perm_off: u32, inv_off: u32, aw_off: u32,
    ind_off: u32, gu_y: u32, dn_y: u32, tiles_off: [u32; NTC],
    pad: [u32; 3],
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeCombArgs { yd_off: u32, inv_off: u32, sel_off: u32, out_off: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ConvBArgs { y_off: u32, ldy: u32, res_off: u32, out_off: u32, cs_off: u32, convw_off: u32, has_res: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct StateArgs { x_off: u32, ldx: u32, cs_off: u32, nch: u32, t: u32, pad: [u32; 3] }
#[repr(C)] #[derive(Clone, Copy)]
struct StateKeepArgs { x_off: u32, ldx: u32, cs_off: u32, nch: u32, t: u32, snap_off: u32, stride: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct KvPrepArgs {
    t: u32, p0: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, cs_off: u32, q_off: u32,
    k_off: u32, v_off: u32, r_off: u32, kdyn_off: i32, vdyn_off: i32, qp_off: u32, rp_off: u32, qhalf: u32,
    kconv_w: u32, vconv_w: u32, qclip_w: u32, relclip_w: u32, log_alpha: f32, log_floor: f32, is_full: u32, kpos0: i32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct AttnBArgs { t: u32, p0: u32, ring: u32, cap: u32, k_base: u32, v_base: u32, window: u32, qp_off: u32,
                   rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32, relbias_w: u32, ghalf: u32, kpos0: i32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct KvCopyArgs { n: u32, pos0: u32, ring_k: u32, ring_v: u32, cap: u32, sk_off: u32, sv_off: u32, scap: u32, kpos0: i32, to_ring: u32, pad: [u32; 2] }

/// batch (prefill) arena: per-token activation arrays [T][dim], f32 offsets into the same arena buffer
#[derive(Clone, Copy, Default, Debug)]
pub struct BArena {
    /// logits in `b.lg`: one vocabulary row per span for a prefill (`lg_rows` of them, at most `MAX_SPANS`).
    /// A verify span uses `1 + k` of those rows, one per position. This is not one row per token.
    lg: u32,
    lg_rows: u32,
    vec0: u32, q: u32, gate: u32, k: u32, v: u32, rel: u32, sink: u32, kdyn: u32, vdyn: u32, qp: u32, rp: u32, attn_g: u32,
    inp: [u32; 3], msg: [u32; 4], cin: [u32; 3], fin: u32, finc: u32, ctrlg: [u32; 4],
    cat_u: [u32; 4], cat_m: [u32; 4], gain_o: [u32; 4], forget_o: [u32; 4],
    xh: [u32; 3], agh: u32, xgh: u32, acth: u32, cath: [u32; 4], gh: [u32; 4],
    tl: [u32; NTC], ind: u32, aw: u32, fh: u32, sk: u32, sv: u32,
    wo_y: u32, ffn_y: u32, rlog: u32, sel: u32, perm: u32, inv: u32, tiles: u32, ntiles: u32, act: u32, yd: u32, end: u32,
    /// tokens this layout was sized for (0: no batch arena allocated)
    cap: usize,
}
impl BArena {
    /// offsets for a batch of up to `t` tokens. `lean`: the fp16 path, which never touches the f32 intermediates of the
    /// fallback kernels and keeps q / gate / the per-assignment outputs as half.
    fn new(start: u32, t: usize, lean: bool, kvb: usize, vocab_pad: usize) -> BArena {
        let mut o = start;
        let mut take = |n: usize| { let r = o; o += ((n + 63) / 64 * 64) as u32; r };
        let fb = |n: usize| if lean { 0 } else { n };      // only the fallback paths use it
        let hf = |n: usize| if lean { n / 2 } else { n };  // half in the fp16 path
        let mut b = BArena { cap: t, ..Default::default() };
        b.vec0 = take(B_NVEC * t * E);
        b.q = take(hf(t * NH * HD)); b.gate = take(hf(t * NH * HD)); b.k = take(t * NKV * HD); b.v = take(t * NKV * HD);
        b.rel = take(t * NH * NREL); b.sink = take(t * NH); b.kdyn = take(t * 32); b.vdyn = take(t * 32);
        b.qp = take(hf(t * NH * HD)); b.rp = take(t * NH * NREL); b.attn_g = take(fb(t * NH * HD));
        for i in 0..3 { b.inp[i] = take(t * E); }
        for i in 0..4 { b.msg[i] = take(t * E); }
        for i in 0..3 { b.cin[i] = take(fb(t * E)); }
        b.fin = take(t * E); b.finc = take(fb(t * E));
        for i in 0..4 { b.cat_u[i] = take(fb(t * E)); b.cat_m[i] = take(fb(t * E)); b.gain_o[i] = take(t * E); b.forget_o[i] = take(fb(t * E)); }
        for i in 0..3 { b.xh[i] = take(t * E / 2); }
        b.agh = take(t * NH * HD / 2);
        b.xgh = take(t * TOPK * E / 2);
        b.acth = take(t * TOPK * FF / 2);
        for i in 0..4 { b.cath[i] = take(t * 2 * E / 2); b.gh[i] = take(t * 64 / 2); }
        for i in 0..4 { b.ctrlg[i] = take(t * 64); }
        b.wo_y = take(t * E); b.ffn_y = take(t * E);
        b.rlog = take(t * NEXP); b.sel = take(t * 16);
        b.tl = [take((t * TOPK / 128 + NEXP + 1) * 4), take((NEXP + 1) * 4)];
        b.ind = take(64); b.aw = take(t * TOPK); b.fh = take(t * E / 2);
        // sliding-window K/V scratch of the batch: [guard | window history | batch keys | guard] per group, half
        b.sk = take(NKV * (EXT + t + 2 * KGUARD) * HD * kvb / 4); b.sv = take(NKV * (EXT + t + 2 * KGUARD) * HD * kvb / 4);
        // Not `t` rows. At chunk 4096 that would be 32× this buffer (128 rows today). A verify span reuses them.
        b.lg_rows = t.min(MAX_SPANS) as u32;
        b.lg = take(b.lg_rows as usize * vocab_pad);
        b.perm = take(t * TOPK); b.inv = take(t * TOPK); b.tiles = take(fb((t * TOPK / 32 + NEXP + 1) * 4)); b.ntiles = take(64);
        b.act = take(fb(t * TOPK * FF)); b.yd = take(hf(t * TOPK * E));
        b.end = o;
        b
    }
    fn vec(&self, i: usize) -> u32 { self.vec0 + (i * self.cap * E) as u32 }
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ArgmaxArgs { logits_off: u32, n: u32, part_off: u32, nparts: u32, out_index: u32, checked: u32, pad: [u32; 2] }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct SampArgs { logits_off: u32, n: u32, seed_lo: u32, seed_hi: u32, pos: u32, ban_tok: u32, temp: f32, softcap: f32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct LoopCdArgs { logits_off: u32, n: u32, stride: u32, pad0: u32, softcap: f32, pad1: u32, pad2: u32, pad3: u32 }

// ----------------------------------------------------------------------------- weights
#[derive(Clone, Copy, Default, Debug)]
pub struct QW { pub off: u64, pub row_bytes: u32, pub ty: u32, pub rows: u32, pub k: u32, pub expert_bytes: u64, pub gs: u64 }

fn qw(g: &Gguf, name: &str, shape: &[usize], types: &[u32]) -> io::Result<QW> {
    let t = g.tensor(name, shape, types)?;
    let rb = t.row_bytes;
    // All callers pass the target model's fixed geometry. The shaders use 32-bit strides.
    let expert_bytes = u32::try_from(rb * t.ne[1])
        .map_err(|_| invalid_model(format_args!("tensor {name} exceeds the GPU stride limit")))?;
    let gs = if t.ty == T_NV4 {
        let gname = name.replace(".weight", ".gscale");
        g.tensor(&gname, &[NEXP], &[T_F32])?.file_off as u64
    } else {
        0
    };
    Ok(QW {
        off: t.file_off as u64,
        row_bytes: rb as u32,
        ty: t.ty,
        rows: t.ne[1] as u32,
        k: t.ne[0] as u32,
        expert_bytes: u64::from(expert_bytes),
        gs,
    })
}

fn validate_vocab(g: &Gguf, unpadded: u64, padded: usize) -> io::Result<usize> {
    let token_count = match g.kv.get("tokenizer.ggml.tokens") {
        Some(Value::AS(tokens)) => tokens.len(),
        _ => return Err(invalid_model("missing GGUF tokenizer tokens")),
    };
    let type_count = match g.kv.get("tokenizer.ggml.token_type") {
        Some(Value::AI(types)) => types.len(),
        _ => return Err(invalid_model("missing GGUF tokenizer token types")),
    };
    let eos = g.uint("tokenizer.ggml.eos_token_id")?;
    check_vocab_lengths(unpadded, padded, token_count, type_count, eos)
}

fn check_vocab_lengths(unpadded: u64, padded: usize, tokens: usize, types: usize, eos: u64) -> io::Result<usize> {
    if unpadded == 0 || unpadded > padded as u64 || unpadded != tokens as u64 || tokens != types {
        return Err(invalid_model(format_args!(
            "vocabulary length {unpadded} must fit {padded} rows and match the tokenizer's {tokens} tokens and {types} token types",
        )));
    }
    if eos >= unpadded {
        return Err(invalid_model(format_args!("EOS token ID {eos} is outside the vocabulary")));
    }
    Ok(tokens)
}

pub struct Layer {
    pub swa: bool,
    pub wq: QW, pub wgate: QW, pub wk: QW, pub wv: QW, pub wo: QW, pub wrel: QW, pub wsink: QW,
    pub kconvp: Option<QW>, pub vconvp: Option<QW>,
    pub attn_norm: i32, pub q_inp_norm: i32, pub k_inp_norm: i32, pub v_inp_norm: i32,
    pub attn_res_query: i32, pub n_attn_q: u32,
    pub ffn_res_query: u32, pub ffn_norm: u32,
    pub q_clip: u32, pub rel_clip: u32, pub rel_bias: u32, pub k_conv: u32, pub v_conv: u32,
    pub attn_post_conv: u32, pub ffn_post_conv: u32,
    pub router: u32, pub exp_bias: u32,
    pub gu: QW, pub dn: QW,
}

pub struct LoopW { pub msg_query: u32, pub down: [u32; 4], pub gain: [u32; 4], pub forget: [u32; 4] }

struct Pb { data: Vec<u8> }
impl Pb {
    fn align(&mut self) { while self.data.len() % 256 != 0 { self.data.push(0); } }
    fn f32(&mut self, g: &Gguf, name: &str, shape: &[usize]) -> io::Result<u32> {
        let t = g.tensor(name, shape, &[T_F32])?;
        self.align();
        let off = self.data.len() as u32;
        self.data.extend_from_slice(g.bytes(t));
        Ok(off)
    }
    /// F32 -> BF16 (round to nearest even). Lossless for this model's bf16-origin matrices.
    fn bf16(&mut self, g: &Gguf, name: &str, shape: &[usize]) -> io::Result<u32> {
        let t = g.tensor(name, shape, &[T_F32])?;
        self.align();
        let off = self.data.len() as u32;
        for c in g.bytes(t).chunks_exact(4) {
            let u = u32::from_le_bytes(c.try_into().unwrap());
            let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1)) >> 16;
            self.data.extend_from_slice(&(r as u16).to_le_bytes());
        }
        Ok(off)
    }
}

// ----------------------------------------------------------------------------- activation arena (f32 offsets)
#[derive(Clone, Copy)]
#[derive(Debug)]
pub struct Arena {
    pub q: u32, pub gate: u32, pub k: u32, pub v: u32, pub rel: u32, pub sink: u32, pub kdyn: u32, pub vdyn: u32,
    pub attn_g: u32, pub inp: [u32; 3], pub msg: [u32; 4], pub cin: [u32; 3], pub ctrlg: [u32; 4], pub ssp: [u32; 4],
    pub cat: [u32; 4], pub fin: u32, pub finc: u32, pub fout: u32, pub rlog: u32, pub sel: u32, pub sel_log: u32, pub act: u32, pub part: u32, pub logits: u32, pub amax: u32, pub total: u32,
}
const N_VEC: usize = 16;
/// hidden-state vectors of the batch arena: at most 4 block states + the current one + 3 pinned + the one being written
const B_NVEC: usize = 10;

impl Arena {
    fn new(max_split: usize, vocab_pad: usize) -> Arena {
        let mut o = (N_VEC * E) as u32;
        let mut take = |n: usize| { let r = o; o += ((n + 63) / 64 * 64) as u32; r };
        let q = take(NH * HD); let gate = take(NH * HD); let k = take(NKV * HD); let v = take(NKV * HD);
        let rel = take(NH * NREL); let sink = take(NH); let kdyn = take(32); let vdyn = take(32);
        let attn_g = take(NH * HD);
        let inp = [take(E), take(E), take(E)];
        let msg = [take(E), take(E), take(E), take(E)];
        let cin = [take(E), take(E), take(E)];
        let ctrlg = [take(64), take(64), take(64), take(64)];
        let ssp = [take(128), take(128), take(128), take(128)];
        let cat = [take(2 * E), take(2 * E), take(2 * E), take(2 * E)];
        let fin = take(E); let finc = take(E); let fout = take(E);
        let rlog = take(NEXP); let sel = take(2 * TOPK); let sel_log = take(64 * 16); let act = take(TOPK * FF);
        let part = take(NH * max_split * (HD + 2));
        let logits = take(vocab_pad);
        let amax = take(2 * 256);
        Arena { q, gate, k, v, rel, sink, kdyn, vdyn, attn_g, inp, msg, cin, ctrlg, ssp, cat, fin, finc, fout, rlog, sel, sel_log, act, part, logits, amax, total: o }
    }
    pub fn vec(i: usize) -> u32 { (i * E) as u32 }
    /// the same layout starting `b` floats further into the activation buffer (a lane's own copy in a decode batch)
    fn shift(&self, b: u32) -> Arena {
        let mut a = *self;
        for f in [&mut a.q, &mut a.gate, &mut a.k, &mut a.v, &mut a.rel, &mut a.sink, &mut a.kdyn, &mut a.vdyn, &mut a.attn_g, &mut a.fin, &mut a.finc,
                  &mut a.fout, &mut a.rlog, &mut a.sel, &mut a.sel_log, &mut a.act, &mut a.part, &mut a.logits, &mut a.amax] { *f += b; }
        for f in a.inp.iter_mut().chain(a.msg.iter_mut()).chain(a.cin.iter_mut()).chain(a.ctrlg.iter_mut()).chain(a.ssp.iter_mut()).chain(a.cat.iter_mut()) { *f += b; }
        a
    }
}

// round-robin vector allocator that never hands out a live vector
#[derive(Clone)]
struct VecAlloc { next: usize, n: usize }
impl VecAlloc {
    fn alloc(&mut self, live: &[usize]) -> usize {
        for _ in 0..self.n {
            let c = self.next;
            self.next = (self.next + 1) % self.n;
            if c != 0 && !live.contains(&c) { return c; }
        }
        panic!("vector pool exhausted");
    }
}

pub struct Engine {
    // Drop all Metal buffers (including the no-copy weights buffer `w`) before `gpu` and `g` (which unmaps the file).
    w: Buf,
    pub pb: Buf, ctr: Buf,
    /// activations: the small decode arena, followed by the batch (prefill) arena while a prefill needs one
    act: std::cell::RefCell<Buf>,
    /// per slot: (is sliding-window, index among the slots of its kind)
    slot_kind: Vec<(bool, usize)>,
    /// dev: capture of the full-attention layers' prepared queries and sinks during a prefill (per slot: n x 24 x 128 fp16, n x 24 f32)
    qcap: RefCell<Option<(Buf, usize)>>,
    pub layers: Vec<Layer>,
    loops: Vec<LoopW>,
    tok_embd: QW, tok_norm: u32, output: QW, output_norm: u32, output_res_query: u32,
    pub ar: Arena,
    /// where the decode path's arena starts in the activation buffer: 0, or a lane's copy during a decode batch
    abase: Cell<u32>,
    /// first float of the decode batch's lane arenas (see `ensure_dbatch`)
    dstart: Cell<u32>,
    pub kv8: bool, kvb: usize, ba: std::cell::Cell<BArena>, pub chunk: usize, pub tensor: bool, pub hgemm: bool, sep_comb: bool, sep_topk: bool,
    /// last-layer tail mode of the prefill (dev tools switch it off to profile full layers)
    pub tail: std::cell::Cell<bool>,
    pub skip: std::cell::Cell<u32>,
    pub stop: std::cell::Cell<usize>, done: std::cell::Cell<usize>, pub last_out: std::cell::Cell<usize>, pub last_mid: std::cell::Cell<usize>,
    pub max_ctx: usize,
    pub n_vocab: usize,
    eps: f32, log_alpha: f32, log_floor: f32, situ_beta: f32, situ_lbeta: f32, wscale: f32, pub softcap: f32,
    /// Convolution images for the verify span in flight, one per position, plus the K/V bytes that span overwrote.
    /// Not `cs_at_prompt`: a rejected tail must not become the next turn's resume point.
    verify_keep: RefCell<Option<VerifyKeep>>,
    /// Set only while `encode_verify`'s layers run, so a later prefill does not write into the images.
    verify_record: Cell<bool>,
    /// Layer 3 pools, one per kind. One contiguous GPU slab per kind, sized by `kvb`,
    /// splitting into at most four slabs at `maxBufferLength`. Attention does not read them yet. The table lives on the lane.
    /// `RefCell` because a step mutates the pool through `&Engine`.
    block_pool: std::cell::RefCell<crate::blocks::BlockPools>,
    /// Shared GPU slab for all Layer 3 blocked lanes' `cs` and 64-bit GPU block tables (`full` + `win`),
    /// with free-list slot recycling on `Drop` of `Lane`.
    blocked_slab: Rc<RefCell<BlockedSlab>>,
    /// Maximum adaptive LoopCD contrast weight (`0.0` = disabled, `0.0..=4.0`).
    loopcd_omega: Cell<f32>,
    /// One bit for each output row whose logits have already been softcapped by LoopCD.
    guided_logit_rows: Cell<u128>,
    /// Lazy-allocated GPU buffer for weak-pass logits `z_1` (`n_lanes * vocab_pad * 4` bytes).
    lg_weak: RefCell<Option<Buf>>,
    /// Force `count = 1` groups in Layer 3 `k_attn_grp_dec` (`INK_NO_GRP_DEC=1` for A/B parity testing).
    no_grp_dec: Cell<bool>,
    // Last fields so Metal device/pipelines and the mmapped GGUF outlive every Metal buffer on Engine.
    pub gpu: Gpu,
    pub g: Gguf,
}

/// Shared GPU slab backing all Layer 3 blocked lanes' convolution states (`cs_buf`) and 64-bit
/// block tables (`bt_buf`, storing `[full | win]` per slot, each 256-byte aligned).
struct BlockedSlab {
    cs_buf: Option<Buf>,
    bt_buf: Option<Buf>,
    cap: usize,
    used: usize,
    bt_stride_u64: usize,
    free_slots: Vec<usize>,
}

impl BlockedSlab {
    fn new(default_blocks: usize) -> BlockedSlab {
        let bt_stride_u64 = default_blocks.max(1).div_ceil(BT_STRIDE_ALIGN_U64) * BT_STRIDE_ALIGN_U64;
        BlockedSlab {
            cs_buf: None,
            bt_buf: None,
            cap: 0,
            used: 0,
            bt_stride_u64,
            free_slots: Vec::new(),
        }
    }

    fn next_cap(need: usize) -> usize {
        if need <= 1 {
            1
        } else if need <= 8 {
            need.next_power_of_two()
        } else {
            need.div_ceil(8) * 8
        }
    }

    #[inline]
    fn cs_f32_base(&self, slot_idx: usize) -> u32 {
        u32::try_from(slot_idx * CS_STRIDE_F32).expect("blocked cs offset fits u32")
    }

    #[inline]
    fn bt_u64_base(&self, slot_idx: usize, swa: bool) -> usize {
        (slot_idx * 2 + usize::from(swa)) * self.bt_stride_u64
    }

    fn alloc_slot(&mut self, gpu: &Gpu, need_blocks: usize) -> usize {
        let slot_idx = if let Some(idx) = self.free_slots.pop() {
            idx
        } else {
            let idx = self.used;
            self.used += 1;
            idx
        };
        let need_stride = need_blocks.max(1).div_ceil(BT_STRIDE_ALIGN_U64) * BT_STRIDE_ALIGN_U64;
        let new_cap = if self.used > self.cap { Self::next_cap(self.used) } else { self.cap };
        let stride_grew = need_stride > self.bt_stride_u64;
        if new_cap > self.cap {
            let nb = gpu.buffer(new_cap * CS_STRIDE_BYTES);
            unsafe {
                let dst = nb.contents().as_ptr() as *mut u8;
                let old_bytes = self.cap * CS_STRIDE_BYTES;
                if let Some(old) = &self.cs_buf {
                    std::ptr::copy_nonoverlapping(old.contents().as_ptr() as *const u8, dst, old_bytes);
                }
                std::ptr::write_bytes(dst.add(old_bytes), 0, new_cap * CS_STRIDE_BYTES - old_bytes);
            }
            self.cs_buf = Some(nb);
        }
        if new_cap > self.cap || stride_grew {
            let old_stride = self.bt_stride_u64;
            let new_stride = old_stride.max(need_stride);
            let new_bytes = new_cap * 2 * new_stride * 8;
            let nb = gpu.buffer(new_bytes);
            unsafe {
                let dst_u8 = nb.contents().as_ptr() as *mut u8;
                std::ptr::write_bytes(dst_u8, 0, new_bytes);
                if let Some(old) = &self.bt_buf {
                    if stride_grew {
                        let src_u64 = old.contents().as_ptr() as *const u64;
                        let dst_u64 = dst_u8 as *mut u64;
                        for s in 0..self.cap {
                            std::ptr::copy_nonoverlapping(
                                src_u64.add(s * 2 * old_stride),
                                dst_u64.add(s * 2 * new_stride),
                                old_stride,
                            );
                            std::ptr::copy_nonoverlapping(
                                src_u64.add((s * 2 + 1) * old_stride),
                                dst_u64.add((s * 2 + 1) * new_stride),
                                old_stride,
                            );
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(
                            old.contents().as_ptr() as *const u8,
                            dst_u8,
                            self.cap * 2 * old_stride * 8,
                        );
                    }
                }
            }
            self.bt_buf = Some(nb);
            self.bt_stride_u64 = new_stride;
        }
        self.cap = new_cap;
        // Zero-fill this slot's cs and both block tables (required when recycling a slot).
        unsafe {
            let cs_ptr = (self.cs_buf.as_ref().unwrap().contents().as_ptr() as *mut u8).add(slot_idx * CS_STRIDE_BYTES);
            std::ptr::write_bytes(cs_ptr, 0, CS_STRIDE_BYTES);
            let bt_ptr = (self.bt_buf.as_ref().unwrap().contents().as_ptr() as *mut u8).add(slot_idx * 2 * self.bt_stride_u64 * 8);
            std::ptr::write_bytes(bt_ptr, 0, 2 * self.bt_stride_u64 * 8);
        }
        slot_idx
    }
}

struct BlockedSlot {
    slab: Rc<RefCell<BlockedSlab>>,
    slot_idx: usize,
    bt_blocks: usize,
}

impl Drop for BlockedSlot {
    fn drop(&mut self) {
        self.slab.borrow_mut().free_slots.push(self.slot_idx);
    }
}

/// One stream's state: everything a token's computation reads or writes besides the weights and the scratch arenas.
/// Two lanes on one engine are independent streams; the engine works on one lane at a time (`Engine::on`).
pub struct Lane {
    /// K/V cache (fp16). Sliding-window layers keep exactly their window (a ring of EXT keys per slot), full-attention
    /// layers every key. Both buffers are sized to the context and grow with it.
    kv_swa: RefCell<Buf>, kv_full: RefCell<Buf>, swa_cap: Cell<usize>, full_cap: Cell<usize>,
    /// causal-convolution rows of every slot for Layer 2 lanes (Layer 3 blocked lanes sub-allocate from `Engine::blocked_slab`)
    cs_own: Option<Buf>,
    /// the stream's tokens by position (greedy decoding writes each step's argmax here, GPU to GPU)
    pub toks: Buf,
    tier: Option<Tier>,
    pub prune: RefCell<Option<Prune>>,
    /// `RefCell` because a step mutates the table through `&Lane`.
    block_table: std::cell::RefCell<crate::blocks::BlockTable>,
    /// True when this lane reads/writes K/V through `Engine::block_pool` and `Engine::blocked_slab`.
    use_blocks: Cell<bool>,
    /// Sub-allocated slot inside `Engine::blocked_slab` for Layer 3 blocked lanes (recycled on `Drop`).
    blocked_slot: Option<BlockedSlot>,
}

/// One lane's GPU Gumbel-max sampling request for [`Engine::encode_gumbel_spans`].
#[allow(dead_code)] // wired by serve.rs after Track 1 + Track 2 merge
#[derive(Clone, Copy)]
pub struct GumbelSpan<'a> {
    pub span_idx: usize,
    pub lane: &'a Lane,
    pub out_index: usize,
    pub seed: u64,
    pub pos: usize,
    pub temp: f32,
    pub ban_tok: Option<u32>,
}

#[derive(Clone, Copy)]
struct DecBatchMeta {
    desc_off: u32,
    kv_cur_off: u32,
    swa_solo_off: u32,
    swa_shared_off: u32,
    n_swa_solo: usize,
    n_swa_shared: usize,
    full_solo_off: u32,
    full_shared_off: u32,
    n_full_solo: usize,
    n_full_shared: usize,
    swa_kps: u32,
    full_kps: u32,
}

/// The engine working on one lane. Methods that touch a stream's state are on `Run`; the rest are the engine's, which
/// `Run` reaches through Deref.
#[derive(Clone, Copy)]
pub struct Run<'a> { pub eng: &'a Engine, pub lane: &'a Lane }
impl std::ops::Deref for Run<'_> { type Target = Engine; fn deref(&self) -> &Engine { self.eng } }

/// One lane's part of a step: positions [p0, p0 + t) of that lane.
#[derive(Clone, Copy)]
pub struct Span<'a> { pub lane: &'a Lane, pub p0: usize, pub t: usize }

/// K/V rows one verify position overwrote, copied before the span ran.
#[derive(Clone)]
struct KvUndo { pos: usize, rows: Vec<u8> }
struct VerifySpan {
    lane: *const Lane,
    p0: usize,
    kv: Vec<KvUndo>,
    /// `p0 + n_keep` after rollback. Absent until then. Not a seal of the span's end, and not `cs_at_prompt`.
    accepted_end: Option<usize>,
}
struct VerifyKeep {
    /// `spans * (1+k)` convolution images, span-major, each `cs`'s layout. Transient.
    cs: Buf,
    stride: usize,
    t: usize,
    spans: Vec<VerifySpan>,
}

static LANE_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Lane {
    /// A fresh, empty stream for `eng`: a small cache that grows with the context, zeroed convolution rows.
    pub fn new(eng: &Engine) -> Lane {
        let n_swa = eng.slot_kind.iter().filter(|k| k.0).count();
        let n_full = N_SLOT - n_swa;
        let cap0 = KV_CAP0.min(eng.max_ctx);
        let gpu = &eng.gpu;
        let tier = if KV_DISK.load(std::sync::atomic::Ordering::Relaxed) {
            assert!(!eng.kv8, "--kv-disk keeps the exact fp16 cache; it cannot be combined with --kv8");
            let path = std::env::temp_dir().join(format!("inky-kv-{}-{}.bin", std::process::id(), LANE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
            let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&path).expect("create the K/V file");
            // uncached I/O: the point of the file is that its contents are NOT in memory
            unsafe { libc::fcntl(std::os::fd::AsRawFd::as_raw_fd(&file), libc::F_NOCACHE, 1); }
            let _ = std::fs::remove_file(&path);       // it lives as long as the lane
            Some(Tier { file, on: Cell::new(false), lo: Default::default(), cold: Default::default(), filed: Default::default(), next: Cell::new(0),
                        min_n: std::env::var("INK_TIER_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(512),
                        stg: RefCell::new(gpu.buffer(16384)), half: Cell::new(0), pending: RefCell::new([None, None]),
                        plan: RefCell::new(Vec::new()), execs: RefCell::new(Vec::new()), loaded: Cell::new(usize::MAX),
                        read_bytes: Cell::new(0), read_time: Cell::new(Default::default()), unaligned: Cell::new(false) })
        } else { None };
        Lane {
            kv_swa: RefCell::new(gpu.buffer(n_swa * 2 * NKV * cap0 * HD * eng.kvb)),
            kv_full: RefCell::new(gpu.buffer(n_full * 2 * NKV * cap0 * HD * eng.kvb)),
            swa_cap: Cell::new(cap0), full_cap: Cell::new(cap0),
            cs_own: Some(gpu.buffer(CS_STRIDE_BYTES)),
            toks: gpu.buffer((eng.max_ctx + 8) * 4),
            tier, prune: RefCell::new(None),
            block_table: std::cell::RefCell::new(crate::blocks::BlockTable::empty()),
            use_blocks: Cell::new(false),
            blocked_slot: None,
        }
    }

    /// A Layer 3 block-addressed lane: no contiguous Layer 2 KV cache buffers; K/V lives in
    /// `Engine::block_pool`, while `cs` and 64-bit block tables are sub-allocated from `Engine::blocked_slab`
    /// and recycled on `Drop`.
    pub fn new_blocked(eng: &Engine, max_pos: usize) -> Lane {
        let gpu = &eng.gpu;
        let blocks = max_pos.div_ceil(crate::blocks::BLOCK).max(1);
        let blocks_u32 = u32::try_from(blocks).expect("block count fits u32");
        let slot_idx = eng.blocked_slab.borrow_mut().alloc_slot(gpu, blocks);
        Lane {
            kv_swa: RefCell::new(gpu.buffer(16)),
            kv_full: RefCell::new(gpu.buffer(16)),
            swa_cap: Cell::new(0), full_cap: Cell::new(0),
            cs_own: None,
            toks: gpu.buffer((eng.max_ctx + 8) * 4),
            tier: None, prune: RefCell::new(None),
            block_table: RefCell::new(crate::blocks::BlockTable::new(blocks_u32).expect("block table")),
            use_blocks: Cell::new(true),
            blocked_slot: Some(BlockedSlot {
                slab: eng.blocked_slab.clone(),
                slot_idx,
                bt_blocks: blocks,
            }),
        }
    }

    /// Access `(buffer, byte_offset, byte_len)` for this lane's convolution state.
    #[inline]
    fn with_cs_buf<R>(&self, f: impl FnOnce(&Buf, usize, usize) -> R) -> R {
        if let Some(slot) = &self.blocked_slot {
            let slab = slot.slab.borrow();
            let buf = slab.cs_buf.as_ref().expect("blocked slab missing cs_buf");
            f(buf, slot.slot_idx * CS_STRIDE_BYTES, CS_STRIDE_BYTES)
        } else {
            let buf = self.cs_own.as_ref().expect("Layer 2 lane missing cs buffer");
            f(buf, 0, buf.length())
        }
    }

    /// Access `(buffer, byte_offset, n_blocks)` for this blocked lane's 64-bit GPU block table.
    #[inline]
    fn with_bt_buf<R>(&self, swa: bool, f: impl FnOnce(&Buf, usize, usize) -> R) -> R {
        let slot = self.blocked_slot.as_ref().expect("blocked lane missing slab slot");
        let slab = slot.slab.borrow();
        let buf = slab.bt_buf.as_ref().expect("blocked slab missing bt_buf");
        let base_u64 = slab.bt_u64_base(slot.slot_idx, swa);
        f(buf, base_u64 * 8, slot.bt_blocks)
    }

    #[allow(dead_code)]
    pub fn is_blocked(&self) -> bool { self.use_blocks.get() }

    /// The lane's block table. `fill_block` writes it; `attn_key_offset` reads it.
    /// Interior mutability: callers borrow through the `RefCell`.
    #[allow(dead_code)] // the GPU kernels still address the layer-2 buffers; the pools are not allocated
    pub fn block_table(&self) -> &std::cell::RefCell<crate::blocks::BlockTable> { &self.block_table }

    /// Install one block. The entry's offset is `byte_offset` for `slot` in `kind`'s pool
    /// at that pool's `kvb`. The generation must match the pool record. An empty pool
    /// is `NoSlots`. Does not allocate a GPU buffer.
    #[allow(dead_code)]
    pub fn fill_block(&self, pools: &mut crate::blocks::BlockPools, kind: crate::blocks::BlockKind, block: crate::blocks::BlockIndex, slot: crate::blocks::Slot, generation: u32) -> Result<(), crate::blocks::BlockError> {
        self.block_table.borrow_mut().fill(pools.of_mut(kind), block, slot, generation)
    }

    /// Byte offset of one key row at `pos` for this pass, K or V, and head.
    /// The slot base comes from the lane table. The width is the pool's `kvb`.
    /// Not `kv_swa` / `kv_full` indexed by `pos` or `pos % cap`.
    #[allow(dead_code)]
    pub fn attn_key_offset(&self, pools: &crate::blocks::BlockPools, kind: crate::blocks::BlockKind, pos: usize, pass: u32, kv: crate::blocks::Kv, head: u32) -> Result<u64, crate::blocks::BlockError> {
        self.block_table.borrow().read_key(pools.of(kind), pos)?.row(pass, kv, head)
    }
}

fn slot_of(pass: usize, ip: usize) -> usize {
    if pass == 0 || !(LOOP_START..LOOP_END).contains(&ip) { ip } else { N_PHYS + (pass - 1) * (LOOP_END - LOOP_START) + (ip - LOOP_START) }
}

impl Engine {
    /// the vocabulary as the arena stores it (rounded up to 64)
    fn vocab_pad(&self) -> usize { (self.ar.amax - self.ar.logits) as usize }
    /// Both block pools. One GPU slab per kind, sized at load to `kvb` and the initial cache block.
    /// Interior mutability: callers borrow through the `RefCell`.
    #[allow(dead_code)] // published for the layer-3 workers; nothing in this commit calls it
    pub fn block_pool(&self) -> &std::cell::RefCell<crate::blocks::BlockPools> { &self.block_pool }

    /// One convolution image: `N_SLOT × NCH × 3` floats. The page's 3.66 MiB.
    pub fn cs_image_bytes() -> usize { N_SLOT * NCH * 3 * 4 }
    /// dev: a hash of a lane's cache (both buffers) and convolution rows, to compare two runs bit for bit
    pub fn lane_hash(&self, lane: &Lane) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        let mut eat_slice = |b: &Buf, off: usize, len: usize| {
            let p = unsafe { (b.contents().as_ptr() as *const u8).add(off) as *const u64 };
            for i in 0..len / 8 { h = (h ^ unsafe { *p.add(i) }).wrapping_mul(0x100000001b3); }
        };
        { let b = lane.kv_swa.borrow(); eat_slice(&b, 0, b.length()); }
        { let b = lane.kv_full.borrow(); eat_slice(&b, 0, b.length()); }
        lane.with_cs_buf(|b, off, len| eat_slice(b, off, len));
        h
    }
    /// dev: make `dst` a copy of `src` (cache and convolution rows); `dst` must have been reserved to the same capacity
    pub fn copy_lane(&self, src: &Lane, dst: &Lane) {
        let cp = |a: &Buf, b: &Buf| { assert_eq!(a.length(), b.length()); unsafe { std::ptr::copy_nonoverlapping(a.contents().as_ptr() as *const u8, b.contents().as_ptr() as *mut u8, a.length()); } };
        cp(&src.kv_swa.borrow(), &dst.kv_swa.borrow()); cp(&src.kv_full.borrow(), &dst.kv_full.borrow());
        src.with_cs_buf(|sb, soff, slen| dst.with_cs_buf(|db, doff, dlen| {
            assert_eq!(slen, dlen);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (sb.contents().as_ptr() as *const u8).add(soff),
                    (db.contents().as_ptr() as *mut u8).add(doff),
                    slen,
                );
            }
        }));
    }
    /// dev: make `dst` hold `src`'s first `n` positions (cache rows of every slot, convolution rows); the two may have
    /// different capacities
    pub fn copy_lane_prefix(&self, src: &Lane, dst: &Lane, n: usize) {
        for (sb, db, sc, dc, nk) in [(&src.kv_swa, &dst.kv_swa, src.swa_cap.get(), dst.swa_cap.get(), n.min(EXT)), (&src.kv_full, &dst.kv_full, src.full_cap.get(), dst.full_cap.get(), n)] {
            assert!(nk <= sc && nk <= dc);
            let (sp, dp) = (sb.borrow().contents().as_ptr() as *const u8, db.borrow().contents().as_ptr() as *mut u8);
            let rows = sb.borrow().length() / (sc * HD * self.kvb);
            assert_eq!(rows, db.borrow().length() / (dc * HD * self.kvb));
            for r in 0..rows { unsafe { std::ptr::copy_nonoverlapping(sp.add(r * sc * HD * self.kvb), dp.add(r * dc * HD * self.kvb), nk * HD * self.kvb); } }
        }
        src.with_cs_buf(|sb, soff, slen| dst.with_cs_buf(|db, doff, dlen| {
            assert_eq!(slen, dlen);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (sb.contents().as_ptr() as *const u8).add(soff),
                    (db.contents().as_ptr() as *mut u8).add(doff),
                    slen,
                );
            }
        }));
    }
    /// dev, the poisoned-pool gate (src/poolgate.rs): CPU views of a layer-2 lane's cache: its window buffer and
    /// its full buffer. Valid until the lane's next `reserve`; nothing may be in flight on the GPU.
    pub fn lane_regions(&self, lane: &Lane) -> [(*mut u8, usize); 2] {
        let r = |b: &Buf| (b.contents().as_ptr() as *mut u8, b.length());
        [r(&lane.kv_swa.borrow()), r(&lane.kv_full.borrow())]
    }
    /// GPU memory the device has allocated (bytes) and the working set it recommends
    pub fn gpu_mem(&self) -> (usize, usize) {
        (self.gpu.current_allocated_size(), self.gpu.recommended_max_working_set_size() as usize)
    }
    /// Exact Metal buffer bytes currently allocated for model weights and shared scratch arenas.
    /// Called after `ensure_batch` and before lanes are created; never undercounts Metal's own
    /// reported device allocation.
    pub fn model_and_scratch_bytes(&self) -> u64 {
        let buffers = self.w.length()
            .saturating_add(self.pb.length())
            .saturating_add(self.ctr.length())
            .saturating_add(self.act.borrow().length())
            .saturating_add(self.lg_weak.borrow().as_ref().map_or(0, |b| b.length()));
        buffers.max(self.gpu.current_allocated_size()) as u64
    }
    /// Pure worst-case per-lane GPU memory formula at `max_ctx`: full KV + sliding-window KV +
    /// convolution/recurrent state + per-lane token buffer + per-lane decode-batch arena slice.
    pub fn lane_worst_case_bytes_for(n_swa: usize, n_full: usize, max_ctx: usize, kvb: usize, arena_floats: usize) -> u64 {
        let kv_row = 2usize.saturating_mul(NKV).saturating_mul(HD).saturating_mul(kvb);
        let swa_kv = n_swa.saturating_mul(max_ctx.min(EXT)).saturating_mul(kv_row);
        let full_kv = n_full.saturating_mul(max_ctx).saturating_mul(kv_row);
        let cs = Self::cs_image_bytes();
        let toks = max_ctx.saturating_add(8).saturating_mul(4);
        let dbatch_arena = arena_floats.saturating_mul(4);
        swa_kv.saturating_add(full_kv).saturating_add(cs).saturating_add(toks).saturating_add(dbatch_arena) as u64
    }
    /// Worst-case GPU memory required by a single lane at `self.max_ctx`.
    #[allow(dead_code)]
    pub fn per_lane_worst_case_bytes(&self) -> u64 {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        let n_full = N_SLOT - n_swa;
        Self::lane_worst_case_bytes_for(n_swa, n_full, self.max_ctx, self.kvb, self.ar.total as usize)
    }
    /// Realistic paged per-lane working reserve for Layer 3 `serve`: KV blocks are paged from
    /// shared `BlockPools` rather than pre-allocated to `max_ctx`, so admission sizes lanes using
    /// `min(max_ctx, 32768)` KV tokens plus full `max_ctx` token/block-table metadata.
    pub fn per_lane_paged_reserve_bytes(&self) -> u64 {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        let n_full = N_SLOT - n_swa;
        let working_ctx = self.max_ctx.min(32768);
        let base = Self::lane_worst_case_bytes_for(n_swa, n_full, working_ctx, self.kvb, self.ar.total as usize);
        let extra_toks = self.max_ctx.saturating_sub(working_ctx).saturating_mul(4) as u64;
        let bt_bytes = self.max_ctx.div_ceil(crate::blocks::BLOCK).saturating_mul(16) as u64;
        base.saturating_add(extra_toks).saturating_add(bt_bytes)
    }
    /// Pure Layer 3 block-pool sizing from `kv_budget_bytes`: allocates ~15% to `Window` slots
    /// and ~85% to `Full` slots while ensuring at least `max_lanes * span_bound(chunk) + 256`
    /// window slots when budget permits (capped to 45% of `kv_budget_bytes` so `Full` is never starved,
    /// with a hard floor of `span_bound(chunk)`), bounded by `MAX_SLABS` per kind.
    pub fn serve_block_pool_counts(
        kv_budget_bytes: u64,
        kvb: u32,
        chunk: usize,
        max_lanes: usize,
        max_slab_bytes: u64,
    ) -> (usize, usize) {
        use crate::blocks::{span_bound, stride_at, BlockKind, MAX_SLABS};
        let win_stride = stride_at(BlockKind::Window, kvb).expect("valid kvb");
        let full_stride = stride_at(BlockKind::Full, kvb).expect("valid kvb");
        let max_win_slab = MAX_SLABS.saturating_mul((max_slab_bytes / win_stride) as usize).max(1);
        let max_full_slab = MAX_SLABS.saturating_mul((max_slab_bytes / full_stride) as usize).max(1);
        let total_win_cap = ((kv_budget_bytes / win_stride) as usize).max(1);
        let lanes = max_lanes.max(1);
        let target_win = ((kv_budget_bytes.saturating_mul(15) / 100) / win_stride) as usize;
        let desired_min_win = lanes.saturating_mul(span_bound(chunk)).saturating_add(256);
        let hard_min_win = span_bound(chunk).max(32);
        let budget_win_cap = ((kv_budget_bytes.saturating_mul(45) / 100) / win_stride) as usize;
        let floor_win = desired_min_win.min(budget_win_cap.max(hard_min_win));
        let n_window = target_win.max(floor_win).clamp(1, total_win_cap.min(max_win_slab));
        let win_bytes = (n_window as u64).saturating_mul(win_stride);
        let rem_bytes = kv_budget_bytes.saturating_sub(win_bytes);
        let n_full = ((rem_bytes / full_stride) as usize).clamp(1, max_full_slab);
        (n_window, n_full)
    }
    /// Bound `kv_budget_bytes` to what `max_lanes` at `max_ctx` can usefully address (including
    /// retained prefix checkpoints), so small-context servers (such as `tests/serve.sh` at
    /// `-c 6144 --lanes 1..4`) allocate a few GiB rather than wiring the entire machine's working
    /// set, while production `-c 65536..131072` servers still use the full `kv_budget_bytes`.
    pub fn capped_serve_kv_budget(
        kv_budget_bytes: u64,
        kvb: u32,
        chunk: usize,
        max_ctx: usize,
        max_lanes: usize,
    ) -> u64 {
        use crate::blocks::{span_bound, stride_at, BlockKind, BLOCK, WINDOW};
        let win_stride = stride_at(BlockKind::Window, kvb).expect("valid kvb");
        let full_stride = stride_at(BlockKind::Full, kvb).expect("valid kvb");
        let lanes = max_lanes.max(1) as u64;
        let blocks_per_lane = max_ctx.div_ceil(BLOCK).max(1) as u64;
        let win_per_lane = ((WINDOW / BLOCK) + 2) as u64;
        let active_span_win = span_bound(chunk.min(max_ctx.max(BLOCK))) as u64;
        let useful_win = lanes
            .saturating_mul(win_per_lane)
            .saturating_mul(2)
            .saturating_add(active_span_win)
            .saturating_add(32);
        let useful_full = lanes
            .saturating_mul(blocks_per_lane)
            .saturating_mul(2)
            .saturating_add(64);
        let useful_bytes = useful_win
            .saturating_mul(win_stride)
            .saturating_add(useful_full.saturating_mul(full_stride));
        kv_budget_bytes.min(useful_bytes)
    }
    /// Size and allocate GPU Layer 3 `BlockPools` for `serve` from `kv_budget_bytes` and `max_lanes`.
    /// Returns `(n_window, n_full)`.
    pub fn init_serve_block_pools(&self, kv_budget_bytes: u64, max_lanes: usize) -> (usize, usize) {
        let max_slab = self.gpu.dev.maxBufferLength() as u64;
        let capped_budget = Self::capped_serve_kv_budget(
            kv_budget_bytes,
            self.kvb as u32,
            self.chunk,
            self.max_ctx,
            max_lanes,
        );
        let (n_window, n_full) = Self::serve_block_pool_counts(
            capped_budget,
            self.kvb as u32,
            self.chunk,
            max_lanes,
            max_slab,
        );
        let pools = crate::blocks::BlockPools::gpu(&self.gpu.dev, n_window, n_full, self.kvb as u32);
        *self.block_pool.borrow_mut() = pools;
        (n_window, n_full)
    }
    /// Vocabulary row `i` in `b.lg` from the last step that wrote it. A prefill with several spans stores one
    /// row per span, so `i` is the span. A verify span stores `1 + k` rows, so `i` is the position in that span
    /// (span `s`, position `r`, at `s * (1 + k) + r`).
    pub fn span_logits(&self, i: usize) -> &[f32] {
        let b = self.ba.get();
        assert!(i < b.lg_rows as usize, "span_logits: row {i} but b.lg has {} rows", b.lg_rows);
        let off = b.lg as usize + i * self.vocab_pad();
        unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const f32).add(off), self.n_vocab) }
    }
    /// Configure maximum adaptive LoopCD-Logits contrast weight (`0.0` disables LoopCD).
    pub fn set_loopcd_omega(&self, omega: f32) {
        assert!(omega.is_finite() && (0.0..=4.0).contains(&omega), "loopcd_omega must be finite in [0, 4]");
        self.loopcd_omega.set(omega);
    }
    /// Current default adaptive LoopCD-Logits contrast weight (`0.0` = disabled).
    pub fn loopcd_omega(&self) -> f32 { self.loopcd_omega.get() }
    /// The sampler applies the model softcap only to raw rows of the last forward step.
    pub fn effective_sampler_softcap(&self, row: usize) -> f32 {
        if self.guided_logit_rows.get() & (1u128 << row) == 0 { self.softcap } else { 0.0 }
    }
    fn ensure_lg_weak(&self, n_rows: usize) -> Buf {
        let need_bytes = n_rows.max(1) * self.vocab_pad() * 4;
        let mut slot = self.lg_weak.borrow_mut();
        if slot.as_ref().is_none_or(|b| b.length() < need_bytes) {
            *slot = Some(self.gpu.buffer(need_bytes));
        }
        slot.as_ref().unwrap().clone()
    }
    fn encode_gumbel_at(&self, cb: &CmdBuf, logits_off: u32, row: usize, lane: &Lane, out_index: usize, seed: u64, pos: usize, temp: f32, ban_tok: Option<u32>) {
        assert!(out_index * 4 < lane.toks.length(), "encode_gumbel: out_index {out_index} out of bounds");
        let sa = SampArgs {
            logits_off,
            n: self.n_vocab as u32,
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            pos: pos as u32,
            ban_tok: ban_tok.unwrap_or(u32::MAX),
            temp,
            softcap: self.effective_sampler_softcap(row),
        };
        let nparts = self.n_vocab.div_ceil(1024).min(256) as u32;
        let am = ArgmaxArgs { logits_off, n: self.n_vocab as u32, part_off: self.ar.part, nparts, out_index: out_index as u32, checked: 1, pad: [0; 2] };
        let e = Enc::new(cb);
        self.bind_eng(&e);
        self.bind_lane(&e, lane);
        self.go(&e, C_ARGMAX, "k_gumbel", &sa, (self.n_vocab.div_ceil(256), 1, 1), 256);
        self.go(&e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
        self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        e.end();
    }
    /// GPU Gumbel-max sampling for a batch span's logits row (`top_k == 0 && top_p >= 1.0`, chadlyb/erisinky):
    /// transforms logits in-place with softcap + Gumbel noise (and optional `ban_tok` mask) and writes the drawn
    /// token ID into `lane.toks[out_index]`.
    pub fn encode_gumbel_span(&self, cb: &CmdBuf, span_idx: usize, lane: &Lane, out_index: usize, seed: u64, pos: usize, temp: f32, ban_tok: Option<u32>) {
        let b = self.ba.get();
        assert!(span_idx < b.lg_rows as usize, "encode_gumbel_span: row {span_idx} >= {}", b.lg_rows);
        let logits_off = b.lg + (span_idx * self.vocab_pad()) as u32;
        self.encode_gumbel_at(cb, logits_off, span_idx, lane, out_index, seed, pos, temp, ban_tok);
    }
    /// Batched GPU Gumbel-max sampling for multiple batch spans inside a SINGLE `MTLComputeCommandEncoder`:
    /// dispatches all `k_gumbel`s concurrently, one barrier, all `k_argmax1`s concurrently into disjoint
    /// partial slices of `ar.part`, one barrier, then all `k_argmax2`s concurrently into each lane's `toks`.
    /// Bit-identical to calling [`Engine::encode_gumbel_span`] per item.
    #[allow(dead_code)] // wired by serve.rs after Track 1 + Track 2 merge
    pub fn encode_gumbel_spans(&self, cb: &CmdBuf, items: &[GumbelSpan<'_>]) {
        if items.is_empty() { return; }
        let b = self.ba.get();
        let vp = self.vocab_pad();
        let nparts = self.n_vocab.div_ceil(1024).min(256) as u32;
        const PART_STRIDE: usize = 512; // 2 * 256 floats per span partial buffer
        assert!(
            items.len() * PART_STRIDE <= NH * MAX_SPLIT * (HD + 2),
            "encode_gumbel_spans: {} spans exceed ar.part scratch capacity",
            items.len()
        );
        for item in items {
            assert!(item.span_idx < b.lg_rows as usize, "encode_gumbel_spans: row {} >= {}", item.span_idx, b.lg_rows);
            assert!(item.out_index * 4 < item.lane.toks.length(), "encode_gumbel_spans: out_index {} out of bounds", item.out_index);
        }
        let e = Enc::new(cb);
        self.bind_eng(&e);
        let gumbel_grid = (self.n_vocab.div_ceil(256), 1, 1);
        for item in items {
            let logits_off = b.lg + (item.span_idx * vp) as u32;
            let sa = SampArgs {
                logits_off,
                n: self.n_vocab as u32,
                seed_lo: item.seed as u32,
                seed_hi: (item.seed >> 32) as u32,
                pos: item.pos as u32,
                ban_tok: item.ban_tok.unwrap_or(u32::MAX),
                temp: item.temp,
                softcap: self.effective_sampler_softcap(item.span_idx),
            };
            self.go_nb(&e, C_ARGMAX, "k_gumbel", &sa, gumbel_grid, 256);
        }
        e.barrier();
        for (idx, item) in items.iter().enumerate() {
            let logits_off = b.lg + (item.span_idx * vp) as u32;
            let am = ArgmaxArgs {
                logits_off,
                n: self.n_vocab as u32,
                part_off: self.ar.part + (idx * PART_STRIDE) as u32,
                nparts,
                out_index: item.out_index as u32,
                checked: 1,
                pad: [0; 2],
            };
            self.go_nb(&e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
        }
        e.barrier();
        for (idx, item) in items.iter().enumerate() {
            let logits_off = b.lg + (item.span_idx * vp) as u32;
            let am = ArgmaxArgs {
                logits_off,
                n: self.n_vocab as u32,
                part_off: self.ar.part + (idx * PART_STRIDE) as u32,
                nparts,
                out_index: item.out_index as u32,
                checked: 1,
                pad: [0; 2],
            };
            e.buf(&item.lane.toks, 0, 6);
            self.go_nb(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        }
        e.end();
    }
    /// GPU Gumbel-max sampling for a single-span prefill whose logits live in `ar.logits`.
    /// Ensures solo (`spans.len() == 1`) and batched (`spans.len() > 1`) prefills draw from
    /// the exact same RNG/sampling kernel for bit-identical `temperature = 1.0` equivalence.
    pub fn encode_gumbel_solo(&self, cb: &CmdBuf, lane: &Lane, out_index: usize, seed: u64, pos: usize, temp: f32, ban_tok: Option<u32>) {
        self.encode_gumbel_at(cb, self.ar.logits, 0, lane, out_index, seed, pos, temp, ban_tok);
    }
    /// the decode path's arena at the current base
    fn ar_at(&self) -> Arena { self.ar.shift(self.abase.get()) }
    /// offset of residual vector `i` of the decode path at the current base
    fn v(&self, i: usize) -> u32 { self.abase.get() + Arena::vec(i) as u32 }
    fn decode_span_limit(&self) -> usize { MAX_SPANS.min(self.chunk).min(self.ba.get().cap / 2) }

    /// The server builds a batch from every ready decoding lane. Check the eventual batch size and GPU path
    /// after the prefill arena is allocated and the device's fp16 layout has been probed.
    pub fn check_serve_decode(&self, lanes: usize) -> Result<(), String> {
        if !self.hgemm { return Err("serve needs the fp16 tensor path for batched decode, which is unavailable".into()); }
        if !self.sep_comb { return Err("serve needs separate split attention combines; unset INK_FUSE_COMB".into()); }
        let limit = self.decode_span_limit();
        if lanes > limit {
            return Err(format!("--lanes {lanes} exceeds the decode batch capacity of {limit} for the selected context and prefill chunk"));
        }
        Ok(())
    }

    /// Toggle grouped shared-KV decode (`true` forces `count = 1` groups, matching `INK_NO_GRP_DEC=1`).
    pub fn set_no_grp_dec(&self, no_grp: bool) { self.no_grp_dec.set(no_grp); }
    /// Whether grouped shared-KV decode is currently disabled (`count = 1` forced).
    #[allow(dead_code)]
    pub fn no_grp_dec(&self) -> bool { self.no_grp_dec.get() }

    /// Room for a decode batch of `n` lanes: a copy of the decode arena per lane, followed by Layer 3
    /// batched-decode scratch (`DecodeSpanDesc`, `kv_cur`, and `DecAttnGroup` lists), after everything else
    /// in the activation buffer. Call after `ensure_batch` (a later batch arena may overlap it). Nothing may
    /// be in flight on the GPU.
    pub fn ensure_dbatch(&self, n: usize) {
        let start = (self.ar.total).max(self.ba.get().end);
        let need = (start as usize + n * self.ar.total as usize + decode_batch_extra_floats(n)) * 4;
        self.dstart.set(start);
        if self.act.borrow().length() < need {
            let nb = self.gpu.buffer(need);
            unsafe { std::ptr::copy_nonoverlapping(self.act.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8, self.act.borrow().length()); }
            *self.act.borrow_mut() = nb;
        }
    }
    /// the logits lane `j` of the last decode batch produced
    pub fn dbatch_logits(&self, j: usize) -> &[f32] {
        let off = (self.dstart.get() + j as u32 * self.ar.total + self.ar.logits) as usize;
        unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const f32).add(off), self.n_vocab) }
    }
    /// the engine working on `lane`
    pub fn on<'a>(&'a self, lane: &'a Lane) -> Run<'a> { Run { eng: self, lane } }
    pub fn load(path: &str, max_ctx: usize) -> Engine {
        let chunk = configured_prefill_chunk(std::env::var("INK_CHUNK"), max_ctx).unwrap_or_else(|err| {
            eprintln!("[inky] {err}");
            std::process::exit(2);
        });
        Self::load_with(path, max_ctx, KV8.load(std::sync::atomic::Ordering::Relaxed), chunk)
            .unwrap_or_else(|err| {
                eprintln!("[inky] could not load {path}: {err}");
                std::process::exit(1);
            })
    }
    /// `kv8`: keep the K/V cache as 8-bit integers with a per-vector scale (half the memory, not exact) instead of fp16
    fn load_with(path: &str, max_ctx: usize, kv8: bool, chunk: usize) -> io::Result<Engine> {
        let t0 = std::time::Instant::now();
        let g = Gguf::open(path)?;
        if !matches!(g.kv.get("general.architecture"), Some(Value::S(s)) if s == "inkling2_loop") {
            return Err(invalid_model("expected inkling2_loop architecture"));
        }
        let a = |k: &str| format!("inkling2_loop.{k}");
        for (name, expected) in [
            ("embedding_length", E), ("block_count", N_PHYS), ("expert_count", NEXP),
            ("expert_used_count", TOPK), ("loop.start", LOOP_START), ("loop.end", LOOP_END),
            ("loop.count", N_PASS), ("attn_res.block_size", BLOCK), ("attention.sliding_window", EXT),
            ("attention.relative_extent", EXT), ("vocab_size", N_VOCAB_PAD),
        ] {
            let value = g.uint(&a(name))?;
            if value != expected as u64 {
                return Err(invalid_model(format_args!("{} is {value}, expected {expected}", a(name))));
            }
        }
        let swa_pat = match g.kv.get(&a("attention.sliding_window_pattern")) {
            Some(Value::AB(v)) if v.len() == N_PHYS
                && v.iter().copied().eq((0..N_PHYS).map(|i| (i + 1) % BLOCK != 0)) => v,
            _ => return Err(invalid_model("unexpected sliding-window attention pattern")),
        };

        let n_vocab = validate_vocab(&g, g.uint(&a("vocab_size_unpadded"))?, N_VOCAB_PAD)?;
        let embed_shape = [E, N_VOCAB_PAD];
        let tok_embd = qw(&g, "token_embd.weight", &embed_shape, &[T_Q4_K, T_Q8_0])?;
        let output = qw(&g, "output.weight", &embed_shape, &[T_Q6_K, T_Q8_0])?;

        let eps = g.f(&a("attention.layer_norm_rms_epsilon"))?;
        let log_alpha = g.f(&a("attention.log_scale_alpha"))?;
        let log_floor = g.f(&a("attention.log_scale_floor"))?;
        let situ_beta = g.f(&a("activation.situ_beta"))?;
        let situ_lbeta = g.f(&a("activation.situ_linear_beta"))?;
        let wscale = g.f(&a("expert_weights_scale"))?;
        let softcap = g.f(&a("final_logit_softcapping"))?;

        let mut pb = Pb { data: Vec::new() };
        let mut layers = Vec::new();
        for (i, &swa) in swa_pat.iter().enumerate() {
            let n = |s: &str| format!("blk.{i}.{s}");
            let (attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, attn_res_query, n_attn_q, kconvp, vconvp) =
                if swa {
                    let query = n("attn_res_query.weight");
                    let res_query = match g.opt(&query) {
                        Some(t) => {
                            let shape: &[usize] = if t.ne.len() == 2 { &[E, 1] } else { &[E] };
                            pb.f32(&g, &query, shape)? as i32
                        }
                        None => -1,
                    };
                    (pb.f32(&g, &n("attn_norm.weight"), &[E])? as i32, -1, -1, -1, res_query, 1, None, None)
                } else {
                    (
                        -1,
                        pb.f32(&g, &n("attn_q_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_k_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_v_inp_norm.weight"), &[E])? as i32,
                        pb.f32(&g, &n("attn_res_query.weight"), &[E, 3])? as i32,
                        3,
                        Some(qw(&g, &n("attn_k_conv_proj.weight"), &[E, NKV * CONV_TAPS], &[T_Q4_K, T_Q6_K, T_Q8_0])?),
                        Some(qw(&g, &n("attn_v_conv_proj.weight"), &[E, NKV * CONV_TAPS], &[T_Q4_K, T_Q6_K, T_Q8_0])?),
                    )
                };
            layers.push(Layer {
                swa,
                wq: qw(&g, &n("attn_q.weight"), &[E, NH * HD], &[T_Q4_K, T_Q8_0])?,
                wgate: qw(&g, &n("attn_gate.weight"), &[E, NH * HD], &[T_Q4_K, T_Q8_0])?,
                wk: qw(&g, &n("attn_k.weight"), &[E, NKV * HD], &[T_Q4_K, T_Q8_0])?,
                wv: qw(&g, &n("attn_v.weight"), &[E, NKV * HD], &[T_Q4_K, T_Q6_K, T_Q8_0])?,
                wo: qw(&g, &n("attn_output.weight"), &[NH * HD, E], &[T_Q4_K, T_Q8_0])?,
                wrel: qw(&g, &n("attn_rel.weight"), &[E, NH * NREL], &[T_Q4_K, T_Q8_0])?,
                wsink: qw(&g, &n("attn_sink_proj.weight"), &[E, NH], &[T_Q4_K, T_Q8_0])?,
                kconvp, vconvp,
                attn_norm, q_inp_norm, k_inp_norm, v_inp_norm, attn_res_query, n_attn_q,
                ffn_res_query: pb.f32(&g, &n("ffn_res_query.weight"), &[E])?,
                ffn_norm: pb.f32(&g, &n("ffn_norm.weight"), &[E])?,
                q_clip: pb.f32(&g, &n("attn_q_clip.weight"), &[HD, NH])?,
                rel_clip: pb.f32(&g, &n("attn_rel_clip.weight"), &[NREL, NH])?,
                rel_bias: pb.f32(&g, &n("attn_rel_bias.weight"), &[NREL, EXT])?,
                k_conv: pb.f32(&g, &n("attn_k_conv.weight"), &[CONV_TAPS, NKV * HD])?,
                v_conv: pb.f32(&g, &n("attn_v_conv.weight"), &[CONV_TAPS, NKV * HD])?,
                attn_post_conv: pb.f32(&g, &n("attn_post_conv.weight"), &[CONV_TAPS, E])?,
                ffn_post_conv: pb.f32(&g, &n("ffn_post_conv.weight"), &[CONV_TAPS, E])?,
                router: pb.bf16(&g, &n("ffn_gate_inp.weight"), &[E, NEXP])?,
                exp_bias: pb.f32(&g, &n("exp_probs_b.bias"), &[NEXP])?,
                gu: qw(&g, &n("ffn_gate_up_exps.weight"), &[E, 2 * FF, NEXP], &[T_Q4_K, T_Q8_0, T_NV4, T_MX8])?,
                dn: qw(&g, &n("ffn_down_exps.weight"), &[FF, E, NEXP], &[T_Q8_0, T_Q5_0, T_NV4, T_MX8])?,
            });
        }
        let mut loops = Vec::new();
        for i in LOOP_START..LOOP_END {
            let n = |s: &str| format!("blk.{i}.{s}");
            let mut lw = LoopW {
                msg_query: pb.f32(&g, &n("loop_msg_query.weight"), &[E, 4])?,
                down: [0; 4], gain: [0; 4], forget: [0; 4],
            };
            for (c, name) in ["q", "k", "v", "ffn"].into_iter().enumerate() {
                lw.down[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_down.weight")), &[2 * E, CTRL_RANK])?;
                lw.gain[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_gain.weight")), &[CTRL_RANK, E])?;
                lw.forget[c] = pb.bf16(&g, &n(&format!("loop_{name}_ctrl_forget.weight")), &[CTRL_RANK, E])?;
            }
            loops.push(lw);
        }
        let tok_norm = pb.f32(&g, "token_embd_norm.weight", &[E])?;
        let output_norm = pb.f32(&g, "output_norm.weight", &[E])?;
        let output_res_query = pb.f32(&g, "output_res_query.weight", &[E])?;

        let src = format!("{}{}\n{}\n{}", if kv8 { "#define KV8 1\n" } else { "" }, include_str!("kernels.metal"), include_str!("prefill.metal"), include_str!("bench_kernels.metal"));
        let kvb = if kv8 { 1 } else { 2 };
        let gpu = Gpu::new(&src, KERNELS);
        if g.map_len() > gpu.dev.maxBufferLength() {
            return Err(invalid_model("GGUF exceeds the Metal buffer size limit"));
        }
        let w = gpu.buffer_nocopy(g.ptr(), g.map_len());
        pb.data.extend(std::iter::repeat(0u8).take(128 * NREL * 4));   // k_attn_h reads up to 95 rows past a relative-bias table
        let pbuf = gpu.buffer_with(&pb.data);

        let max_split = MAX_SPLIT;
        let ar = Arena::new(max_split, N_VOCAB_PAD);
        assert!(max_ctx % 64 == 0, "context size must be a multiple of 64");
        // the batch arena is allocated when a prefill needs it and released when generation starts
        let act = gpu.buffer(ar.total as usize * 4);
        // K/V cache layout: per slot [K: NKV x cap x HD][V: NKV x cap x HD] halfs, sliding-window and full slots in separate
        // buffers that both grow with the context (Metal commits a buffer when it is created, so capacity is real memory)
        let mut slot_kind = Vec::new();
        let (mut n_swa, mut n_full) = (0usize, 0usize);
        for s in 0..N_SLOT {
            let phys = if s < N_PHYS { s } else { LOOP_START + (s - N_PHYS) % (LOOP_END - LOOP_START) };
            if layers[phys].swa { slot_kind.push((true, n_swa)); n_swa += 1; } else { slot_kind.push((false, n_full)); n_full += 1; }
        }
        assert!(n_full * 2 * NKV * max_ctx * HD < u32::MAX as usize, "KV cache too large for 32-bit offsets");
        let ctr = gpu.buffer(4096 * 4);
        eprintln!("[inky] loaded {} ({:.1} GiB mapped, {:.1} MiB params) in {:.2?}",
            path.rsplit('/').next().unwrap_or(path), g.file_len() as f64 / (1u64 << 30) as f64, pb.data.len() as f64 / (1 << 20) as f64, t0.elapsed());
        // KV_CAP0 positions is one block. Further slots come from `BlockPool::grow`.
        let block_pool = std::cell::RefCell::new(crate::blocks::BlockPools::gpu(&gpu.dev, 1, 1, kvb as u32));
        let blocked_slab = Rc::new(RefCell::new(BlockedSlab::new(max_ctx.div_ceil(crate::blocks::BLOCK))));
        let eng = Engine {
            eps, log_alpha, log_floor, situ_beta, situ_lbeta, wscale, softcap,
            g, gpu, w, pb: pbuf, act: std::cell::RefCell::new(act),
            slot_kind, qcap: RefCell::new(None), ctr, layers, loops, tok_embd, tok_norm, output, output_norm, output_res_query,
            ar, abase: Cell::new(0), dstart: Cell::new(0), kv8, kvb, ba: Default::default(), chunk, tensor: std::env::var("INK_NO_TENSOR").is_err(),
            hgemm: std::env::var("INK_NO_HGEMM").is_err() && std::env::var("INK_NO_TENSOR").is_err(),
            sep_comb: std::env::var("INK_FUSE_COMB").is_err(), sep_topk: std::env::var("INK_SEP_TOPK").is_ok(),
            tail: std::cell::Cell::new(std::env::var("INK_NO_TAIL").is_err()),
            skip: Default::default(), stop: std::cell::Cell::new(usize::MAX), done: std::cell::Cell::new(0),
            last_out: Default::default(), last_mid: Default::default(),
            max_ctx, n_vocab,
            verify_keep: RefCell::new(None), verify_record: Cell::new(false),
            block_pool,
            blocked_slab,
            loopcd_omega: Cell::new(0.0),
            guided_logit_rows: Cell::new(0),
            lg_weak: RefCell::new(None),
            no_grp_dec: Cell::new(std::env::var("INK_NO_GRP_DEC").is_ok_and(|v| !v.is_empty() && v != "0")),
        };
        let mut eng = eng;
        eng.ensure_batch(64);
        let ok = !eng.hgemm || eng.layout_ok();
        eng.release_batch();
        if !ok {
            eprintln!("[inky] unexpected tensor accumulator layout on this GPU/OS: using the slower prefill path");
            eng.hgemm = false;
        }
        if std::env::var("INK_NO_WARM").is_err() { eng.warm(); }
        Ok(eng)
    }

    /// Allocate (or enlarge) the batch arena for a prefill of `t` tokens. Nothing may be in flight on the GPU.
    pub fn ensure_batch(&self, t: usize) {
        let t = t.div_ceil(64) * 64;
        if t <= self.ba.get().cap { return; }
        let nb = BArena::new(self.ar.total, t, self.hgemm, self.kvb, self.vocab_pad());
        self.realloc_act(nb.end as usize * 4);
        self.ba.set(nb);
    }
    /// Give the batch arena back (the decode arena and the last logits stay). Nothing may be in flight on the GPU.
    pub fn release_batch(&self) {
        if self.ba.get().cap == 0 { return; }
        self.realloc_act(self.ar.total as usize * 4);
        self.ba.set(BArena::default());
    }
    fn realloc_act(&self, bytes: usize) {
        let nb = self.gpu.buffer(bytes);
        unsafe { std::ptr::copy_nonoverlapping(self.act.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8, self.ar.total as usize * 4); }
        *self.act.borrow_mut() = nb;
    }
    // ======================================================================== dev: key eviction experiments
    pub fn qcap_alloc(&self, n: usize) {
        let nf = self.slot_kind.iter().filter(|k| !k.0).count();
        *self.qcap.borrow_mut() = Some((self.gpu.buffer(nf * n * (NH * HD / 2 + NH) * 4), n));
    }
    pub fn arena(&self, off: u32, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const f32).add(off as usize), n).to_vec() }
    }
    /// expert ids chosen by each of the 48 layer executions of the last decode step
    pub fn routes(&self) -> Vec<[u8; TOPK]> {
        let v = self.arena(self.ar.sel_log, 48 * 16);
        (0..48).map(|l| { let mut r = [0u8; TOPK]; for k in 0..TOPK { r[k] = v[l * 16 + k] as u8; } r }).collect()
    }
    /// dev: MoE tiles of the last prefilled layer: per width class (128, 64, 32) the per-tile token counts
    pub fn moe_tiles(&self) -> [Vec<u32>; NTC] {
        let a = self.act.borrow().contents().as_ptr() as *const u32;
        let cls = |c: usize| unsafe {
            let n = *a.add(self.ba.get().ind as usize + c * 4) as usize;
            (0..n).map(|i| *a.add(self.ba.get().tl[c] as usize + i * 4 + 2)).collect::<Vec<u32>>()
        };
        [cls(0), cls(1)]
    }
    /// The register epilogues of the fp16 kernels assume how the tensor ops spread their accumulators over threads
    /// (see the comment above k_gemm_sg in prefill.metal).
    fn layout_ok(&self) -> bool {
        {
            let out_off = self.ba.get().yd;
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            e.pso(&self.gpu.pso("k_layout_sg")); e.bytes(&out_off, 5);
            e.dispatch((1, 1, 1), 32);
            e.end(); cb.commit(); cb.waitUntilCompleted();
            let o = unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const i32).add(out_off as usize), 32 * 64) };
            for t in 0..32 {
                if o[t * 64 + 63] != 32 { return false; }
                let rt = |i: usize| (o[t * 64 + i] >> 16, o[t * 64 + i] & 0xffff);
                for q in 0..8 {
                    let (r0, t0) = rt(4 * q);
                    if o[t * 64 + 4 * q] < 0 || r0 % 4 != 0 || t0 >= 16 { return false; }
                    for e in 1..4 { if rt(4 * q + e) != (r0 + e as i32, t0) { return false; } }
                    if q < 4 && (r0 >= 32 || rt(4 * q + 16) != (r0 + 32, t0)) { return false; }
                }
            }
        }
        // attention P.V accumulators: 16 per thread, quads of 4 consecutive dims of one query row
        {
            let out_off = self.ba.get().yd;
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            e.pso(&self.gpu.pso("k_layout_att")); e.bytes(&out_off, 5);
            e.dispatch((1, 1, 1), 256);
            e.end(); cb.commit(); cb.waitUntilCompleted();
            let o = unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const i32).add(out_off as usize), 256 * 64) };
            for t in 0..256 {
                if o[t * 64 + 63] != 16 { return false; }
                for q in 0..4 {
                    let (d0, r0) = (o[t * 64 + 4 * q] >> 16, o[t * 64 + 4 * q] & 0xffff);
                    if o[t * 64 + 4 * q] < 0 || d0 % 4 != 0 { return false; }
                    for e in 1..4 { if o[t * 64 + 4 * q + e] != ((d0 + e as i32) << 16 | r0) { return false; } }
                }
            }
        }
        true
    }
    pub fn logits(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts((self.act.borrow().contents().as_ptr() as *const f32).add(self.ar.logits as usize), self.n_vocab) }
    }

    /// micro-benchmark a Q4_K matvec variant over the gate_up expert weights of every layer (no cache reuse)
    pub fn kbench(&self, name: &str, nr: usize, nsg: usize) -> f64 {
        #[repr(C)] #[derive(Clone, Copy)]
        struct BenchArgs { w_lo: u32, w_hi: u32, row_bytes: u32, nrows: u32, k: u32, x_off: u32, out_off: u32, pad: u32 }
        let rows = 8 * E; // 8 experts' worth of gate_up rows
        let mut best = f64::MAX;
        for rep in 0..4 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            e.pso(&self.gpu.pso(name));
            let mut bytes = 0usize;
            for l in 0..N_PHYS {
                for chunk in 0..4 {
                    let gu = &self.layers[l].gu;
                    let off = gu.off + ((rep * 4 + chunk) % 16) as u64 * 8 * gu.expert_bytes;
                    let a = BenchArgs { w_lo: off as u32, w_hi: (off >> 32) as u32, row_bytes: gu.row_bytes, nrows: rows as u32, k: E as u32,
                                        x_off: Arena::vec(1), out_off: self.ar.logits, pad: 0 };
                    e.bytes(&a, 5);
                    e.dispatch((rows / (nr * nsg), 1, 1), 32 * nsg);
                    e.barrier();
                    bytes += rows * gu.row_bytes as usize;
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            let t = cb.GPUEndTime() - cb.GPUStartTime();
            if rep > 0 { best = best.min(bytes as f64 / t / 1e9); }
        }
        best
    }

    /// micro-benchmark a MoE-down variant over all layers' down experts (random-ish expert picks)
    pub fn kbench_down(&self, name: &str, rows_per_tg: usize, tg_threads: usize, only_type: u32) -> f64 {
        let ar = self.ar;
        // synthetic act + selection
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        unsafe {
            for i in 0..TOPK * FF { *a.add(ar.act as usize + i) = ((i * 7919) % 1000) as f32 / 1000.0 - 0.5; }
            for k in 0..TOPK { *a.add(ar.sel as usize + TOPK + k) = 0.125; }
        }
        let mut best = f64::MAX;
        for rep in 0..4 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            let mut bytes = 0usize;
            let mut sels = Vec::new();
            for l in 0..N_PHYS {
                if only_type != 0 && self.layers[l].dn.ty != only_type { continue; }
                for it in 0..4 {
                    let dn = &self.layers[l].dn;
                    let ma = MoeArgs { sel_off: ar.logits + 100000 + ((l * 4 + it) as u32 % 32) * 16, act_off: ar.act, out_off: Arena::vec(2), res_off: Arena::vec(3) as i32,
                                       dn_lo: dn.off as u32, dn_hi: (dn.off >> 32) as u32, dn_expert_bytes: dn.expert_bytes as u32,
                                       dn_row_bytes: dn.row_bytes, dn_type: dn.ty,
                                       dn_gs_lo: dn.gs as u32, dn_gs_hi: (dn.gs >> 32) as u32, ..Default::default() };
                    sels.push(ma.sel_off);
                    e.pso(&self.gpu.pso(name));
                    e.bytes(&ma, 5);
                    e.dispatch((E / rows_per_tg, 1, 1), tg_threads);
                    e.barrier();
                    bytes += TOPK * E * dn.row_bytes as usize;
                }
            }
            unsafe {
                for (j, &so) in sels.iter().enumerate() {
                    for k in 0..TOPK { *a.add(so as usize + k) = (((j * 37 + k * 13 + rep * 5) % 128) as f32).max(0.0); *a.add(so as usize + TOPK + k) = 0.125; }
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            let t = cb.GPUEndTime() - cb.GPUStartTime();
            if rep > 0 { best = best.min(bytes as f64 / t / 1e9); }
        }
        best
    }

    /// touch every page of the mapped model from the GPU once (wires the GPU page mappings: avoids first-use
    /// page-fault stalls when a rarely used expert is hit for the first time mid-generation)
    pub fn warm(&self) {
        #[repr(C)] #[derive(Clone, Copy)]
        struct TouchArgs { lo: u32, hi: u32, n16: u32, pad: u32 }
        let t0 = std::time::Instant::now();
        let chunk = 1usize << 30;
        let mut off = 0usize;
        let cb = self.gpu.cmd();
        let e = Enc::new(&cb); self.bind_eng(&e);
        e.pso(&self.gpu.pso("k_touch"));
        while off < self.g.file_len() {
            let n = chunk.min(self.g.file_len() - off) / 16;
            let ta = TouchArgs { lo: off as u32, hi: (off >> 32) as u32, n16: n as u32, pad: 0 };
            e.bytes(&ta, 5); e.dispatch((4096, 1, 1), 256);
            off += chunk;
        }
        e.end(); cb.commit(); cb.waitUntilCompleted();
        eprintln!("[inky] warmed {:.1} GiB of weights in {:.0?}", self.g.file_len() as f64 / (1u64 << 30) as f64, t0.elapsed());
    }

    /// raw GPU read bandwidth over the mapped weights
    pub fn bw_weights(&self) {
        #[repr(C)] #[derive(Clone, Copy)]
        struct TouchArgs { lo: u32, hi: u32, n16: u32, pad: u32 }
        for (gb, tgs) in [(1usize, 1024usize), (2, 2048), (4, 4096), (8, 8192)] {
            let mut best = 0.0f64;
            for rep in 0..3 {
                let off = (1u64 << 30) + rep as u64 * (1u64 << 28);
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb); self.bind_eng(&e);
                let ta = TouchArgs { lo: off as u32, hi: (off >> 32) as u32, n16: ((gb << 30) / 16) as u32, pad: 0 };
                e.pso(&self.gpu.pso("k_touch")); e.bytes(&ta, 5); e.dispatch((tgs, 1, 1), 256);
                e.end(); cb.commit(); cb.waitUntilCompleted();
                best = best.max((gb << 30) as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e9);
            }
            println!("weights read {gb} GB: {best:.1} GB/s");
        }
    }

    /// benchmark the production MoE kernels over every layer with random expert picks (GB/s of expert weights)
    pub fn kbench_moe(&self, which: &str, grid: (usize, usize), tgs: usize) -> f64 {
        let ar = self.ar;
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        let mut best = 0.0f64;
        for rep in 0..5 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            let mut bytes = 0usize;
            for l in 0..N_PHYS {
                for it in 0..2 {
                    let sel = ar.logits + 100000 + ((l * 2 + it) as u32) * 16;
                    unsafe { for k in 0..TOPK {
                        *a.add(sel as usize + k) = (((l * 2 + it) * 53 + k * 17 + rep * 29) % 128) as f32;
                        *a.add(sel as usize + TOPK + k) = 0.125; } }
                    let ld = &self.layers[l];
                    let ma = MoeArgs { sel_off: sel, x_off: Arena::vec(1), act_off: ar.act, out_off: Arena::vec(2), res_off: Arena::vec(3) as i32,
                        gu_lo: ld.gu.off as u32, gu_hi: (ld.gu.off >> 32) as u32, gu_expert_bytes: ld.gu.expert_bytes as u32, gu_row_bytes: ld.gu.row_bytes,
                        gu_type: ld.gu.ty, gu_gs_lo: ld.gu.gs as u32, gu_gs_hi: (ld.gu.gs >> 32) as u32,
                        dn_lo: ld.dn.off as u32, dn_hi: (ld.dn.off >> 32) as u32, dn_expert_bytes: ld.dn.expert_bytes as u32, dn_row_bytes: ld.dn.row_bytes,
                        dn_type: ld.dn.ty, dn_gs_lo: ld.dn.gs as u32, dn_gs_hi: (ld.dn.gs >> 32) as u32,
                        beta: 4.0, lbeta: 25.0, cs_off: 0, convw_off: ld.ffn_post_conv, ..Default::default() };
                    e.pso(&self.gpu.pso(which));
                    e.bytes(&ma, 5);
                    e.dispatch((grid.0, grid.1, 1), tgs);
                    e.barrier();
                    bytes += TOPK * if which.contains("gu") { ld.gu.expert_bytes as usize } else { ld.dn.expert_bytes as usize };
                }
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            if rep > 0 { best = best.max(bytes as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e9); }
        }
        best
    }

    /// dense GEMM tiling benchmark: QKV-sized Q4_K matrices of all layers, T tokens of random activations
    pub fn kbench_gemm(&self, name: &str, bm: usize, bn: usize, t: usize) -> f64 {
        self.ensure_batch(self.chunk);
        let b = self.ba.get();
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        let half_x = name.ends_with("_3") || name.starts_with("kgb") || name.starts_with("kgs");
        let nsg: usize = if name.starts_with("kgb") || name.starts_with("kgs") { name.rsplit('_').next().unwrap().parse().unwrap() } else { 4 };
        let swap = name.starts_with("kgs");
        unsafe {
            for i in 0..t * E {
                let v = ((i * 7919) % 1000) as f32 / 500.0 - 1.0;
                if half_x {
                    // f32 -> f16 bits (simple round-to-nearest for normal range)
                    let bits = v.to_bits(); let sign = (bits >> 16) & 0x8000; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
                    let h: u16 = if v == 0.0 || exp <= 0 { sign as u16 } else { (sign | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16 };
                    *(a.add(b.inp[0] as usize) as *mut u16).add(i) = h;
                } else { *a.add(b.inp[0] as usize + i) = v; }
            }
        }
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            let tt = t as u32;
            let mut flops = 0.0;
            for l in &self.layers {
                let w = &l.wq;
                let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                in_off: b.inp[0], out_off: b.gain_o[0], ldo: w.rows, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                {
                if swap { e.dispatch((t.div_ceil(bn), (w.rows as usize).div_ceil(bm), 1), 32 * nsg); }
                else { e.dispatch(((w.rows as usize).div_ceil(bm), t.div_ceil(bn), 1), 32 * nsg); }
                e.barrier();
                flops += 2.0 * t as f64 * w.rows as f64 * w.k as f64;
                }
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }

    /// dev: after a prefill of `t` tokens, re-run only the MoE GEMM kernels of every layer on the buffers of the last layer
    /// (real routing, gathered inputs already in place): (gate/up TFLOPs, down TFLOPs)
    pub fn kbench_moe_pf(&self, t: usize) -> (f64, f64) {
        let b = self.ba.get();
        let mut res = [0.0f64; 2];
        for which in 0..2 {
            for _ in 0..3 {
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb);
                self.bind_eng(&e);
                for l in &self.layers {
                    for (c, (name, _, tg)) in MOE_CLS.iter().enumerate() {
                        let g = if which == 0 { MoeGemmHArgs {
                            w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                            wtype: l.gu.ty, nrows: (2 * FF) as u32, k: E as u32, xh_off: b.xgh, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                            out_off: b.acth, ldo: FF as u32, situ: 1, aw_off: b.aw, beta: self.situ_beta, lbeta: self.situ_lbeta,
                            gs_lo: l.gu.gs as u32, gs_hi: (l.gu.gs >> 32) as u32, pad0: 0, pad1: 0,
                        } } else { MoeGemmHArgs {
                            w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                            wtype: l.dn.ty, nrows: E as u32, k: FF as u32, xh_off: b.acth, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                            out_off: b.yd, ldo: E as u32, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                            gs_lo: l.dn.gs as u32, gs_hi: (l.dn.gs >> 32) as u32, pad0: 0, pad1: 0,
                        } };
                        e.pso(&self.gpu.pso(name)); e.bytes(&g, 5);
                        e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + which * 4 * NTC + c * 4) * 4, *tg);
                    }
                    e.barrier();
                }
                e.end(); cb.commit(); cb.waitUntilCompleted();
                let flops = self.layers.len() as f64 * (t * TOPK) as f64 * 2.0 * (E * FF) as f64 * if which == 0 { 2.0 } else { 1.0 };
                res[which] = res[which].max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
            }
        }
        (res[0], res[1])
    }

    /// per-simdgroup tile benchmark: `nact` of `nsg` simdgroups per tile do the matmul. Returns TFLOPs over the ACTIVE tokens.
    /// tensor-unit ceiling probe (fp16 x fp16 from device memory, no weight staging): TFLOPs
    pub fn kbench_peak(&self, name: &str, nsg: usize, t: usize) -> f64 {
        self.ensure_batch(self.chunk);
        let b = self.ba.get();
        let h = self.act.borrow().contents().as_ptr() as *mut u16;
        unsafe { for i in 0..t * E { *h.add(b.inp[0] as usize * 2 + i) = 0x3800 + ((i * 7919) % 512) as u16; } }
        let rows = 3072usize;
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            let tt = t as u32;
            for _ in 0..8 {
                let sg = GSeg { k: E as u32, in_off: b.inp[0], out_off: b.gain_o[0], ldo: rows as u32, nrows: rows as u32, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                e.dispatch((t / (16 * nsg), rows / 64, 1), 32 * nsg);
                e.barrier();
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(8.0 * 2.0 * (t * rows * E) as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }
    /// precision-mode probe: TFLOPs of a tile-op variant and its error against the exact-accumulation variant (layer 0 wq, realistic x)
    pub fn kbench_mode(&self, name: &str, t: usize) -> (f64, f64, f64) {
        self.ensure_batch(self.chunk);
        let b = self.ba.get();
        let a = self.act.borrow().contents().as_ptr() as *mut f32;
        let h = self.act.borrow().contents().as_ptr() as *mut u16;
        // x ~ unit-variance pseudo-normal values, as fp16
        let f16 = |v: f32| -> u16 { let bits = v.to_bits(); let sign = (bits >> 16) & 0x8000; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign as u16 } else { (sign | ((exp as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16 } };
        let mut st = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 40) as f32 / (1u64 << 24) as f32 };
        unsafe { for i in 0..t * E { let v = rnd() + rnd() + rnd() + rnd() - 2.0; *h.add(b.inp[0] as usize * 2 + i) = f16(v * 1.7); } }
        let w = &self.layers[0].wq;
        let run = |nm: &str, out: u32| -> f64 {
            let mut best = 0.0f64;
            for _ in 0..3 {
                let cb = self.gpu.cmd();
                let e = Enc::new(&cb);
                self.bind_eng(&e);
                let tt = t as u32;
                for _ in 0..8 {
                    let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                    in_off: b.inp[0], out_off: out, ldo: w.rows, ..Default::default() };
                    e.pso(&self.gpu.pso(nm)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                    e.dispatch((t / 128, w.rows as usize / 64, 1), 256);
                    e.barrier();
                }
                e.end(); cb.commit(); cb.waitUntilCompleted();
                best = best.max(8.0 * 2.0 * t as f64 * w.rows as f64 * w.k as f64 / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
            }
            best
        };
        run("kgmode_0_0", b.q);
        let tf = run(name, b.gate);
        let n = t * w.rows as usize;
        let (mut se, mut sr, mut mx) = (0.0f64, 0.0f64, 0.0f64);
        unsafe { for i in 0..n { let (x, y) = (*a.add(b.gate as usize + i) as f64, *a.add(b.q as usize + i) as f64); se += (x - y) * (x - y); sr += y * y; mx = mx.max((x - y).abs()); } }
        (tf, (se / sr).sqrt(), mx / (sr / n as f64).sqrt())
    }
    pub fn kbench_sgn(&self, name: &str, nsg: usize, nact: usize, t: usize) -> f64 {
        self.ensure_batch(self.chunk);
        let b = self.ba.get();
        let a = self.act.borrow().contents().as_ptr() as *mut u16;
        unsafe { for i in 0..t * E { *a.add(b.inp[0] as usize * 2 + i) = 0x3800 + ((i * 7919) % 512) as u16; } }
        let mut best = 0.0f64;
        for _ in 0..3 {
            let cb = self.gpu.cmd();
            let e = Enc::new(&cb);
            self.bind_eng(&e);
            let tt = (t | nact << 16) as u32;
            let mut flops = 0.0;
            for l in &self.layers {
                let w = &l.wq;
                let sg = GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
                                in_off: b.inp[0], out_off: b.gain_o[0], ldo: w.rows, ..Default::default() };
                e.pso(&self.gpu.pso(name)); e.bytes(&sg, 5); e.bytes(&tt, 9);
                e.dispatch((t / (16 * nsg), w.rows as usize / 64, 1), 32 * nsg);
                e.barrier();
                flops += 2.0 * (t * nact / nsg) as f64 * w.rows as f64 * w.k as f64;
            }
            e.end(); cb.commit(); cb.waitUntilCompleted();
            best = best.max(flops / (cb.GPUEndTime() - cb.GPUStartTime()) / 1e12);
        }
        best
    }

    /// dev: overwrite the scratch arenas (decode and batch) with NaN in both f32 and fp16 readings, so a result that still
    /// depends on what a previous step left there shows. The counters are not touched: their kernels leave them at zero.
    pub fn arena_map(&self) -> String { format!("{:?}\nbatch {:?}", self.ar, self.ba.get()) }
    pub fn poison_scratch(&self) {
        let a = self.act.borrow();
        let p = a.contents().as_ptr() as *mut u32;
        // dev: INK_POISON_RANGE=lo,hi poisons only words [lo, hi), to find a region that is read before it is written
        let (lo, hi) = std::env::var("INK_POISON_RANGE").ok().and_then(|v| v.split_once(',').map(|(x, y)| (x.parse().unwrap(), y.parse().unwrap())))
            .unwrap_or((0usize, usize::MAX));
        for i in lo..hi.min(a.length() / 4) { unsafe { *p.add(i) = 0x7fc0_7fc0; } }
    }
    /// bind a lane's cache (sliding-window buffer), convolution rows and tokens
    fn bind_lane(&self, e: &Enc, lane: &Lane) {
        e.buf(&lane.kv_swa.borrow(), 0, 3);
        lane.with_cs_buf(|b, off, _| e.buf(b, off, 4));
        e.buf(&lane.toks, 0, 6);
    }
    /// Bind the 1..=4 GPU slab buffers of `kind` (`Window` if `swa`, else `Full`) at buffer slots 3, 10, 11, 12.
    fn bind_pool_slabs(&self, e: &Enc, swa: bool) {
        let pools = self.block_pool.borrow();
        let pool = if swa { &pools.window } else { &pools.full };
        let slabs = pool.gpu_slabs();
        assert!(!slabs.is_empty(), "Layer 3 block pool ({}) has no GPU slabs", if swa { "window" } else { "full" });
        e.buf(slabs[0], 0, 3);
        e.buf(slabs.get(1).copied().unwrap_or(slabs[0]), 0, 10);
        e.buf(slabs.get(2).copied().unwrap_or(slabs[0]), 0, 11);
        e.buf(slabs.get(3).copied().unwrap_or(slabs[0]), 0, 12);
    }
    /// Bind a blocked lane's 64-bit GPU block table (`Window` if `swa`, else `Full`) at buffer slot 8.
    fn bind_lane_bt(e: &Enc, lane: &Lane, swa: bool) {
        lane.with_bt_buf(swa, |b, off, _| e.buf(b, off, 8));
    }
    /// Validate required blocks for span `[p0, p0 + t)` and incrementally upload ONLY the lane's dirty
    /// 64-bit block-table range `[dirty_lo..=dirty_hi]` to GPU memory (0 bytes written on clean decode steps).
    pub fn sync_lane_block_table(&self, lane: &Lane, p0: usize, t: usize) {
        use crate::blocks::{BlockIndex, BlockKind, BLOCK as KV_BLOCK};
        assert!(lane.use_blocks.get(), "sync_lane_block_table called on a Layer 2 lane");
        let mut table = lane.block_table.borrow_mut();
        let pools = self.block_pool.borrow();
        let n_blk = table.blocks_per_kind() as usize;
        assert!(t >= 1 && p0 + t <= n_blk * KV_BLOCK, "Layer 3 span [{p0}, {}) exceeds block table ({} blocks)", p0 + t, n_blk);
        let lo_span_blk = p0 / KV_BLOCK;
        let hi_blk = (p0 + t - 1) / KV_BLOCK;
        let lo_win_blk = p0.saturating_sub(EXT - 1) / KV_BLOCK;
        for (kind, lo) in [
            (BlockKind::Full, 0usize),
            (BlockKind::Window, lo_win_blk),
        ] {
            let pool = pools.of(kind);
            // O(1) boundary validation in release builds; full `[lo..=hi_blk]` verification in debug builds.
            for check_b in [lo, lo_span_blk, hi_blk] {
                let bi = BlockIndex::new(check_b as u32);
                let entry = table.get(kind, bi).expect("block index in range").unwrap_or_else(|| {
                    panic!("Layer 3 span [{p0}, {}) missing required {kind:?} block {check_b}", p0 + t)
                });
                let _ = pool.gpu_table_entry(entry).unwrap_or_else(|err| {
                    panic!(
                        "Layer 3 {kind:?} block {check_b} entry invalid for slot {} gen {}: {err}",
                        entry.slot().get(),
                        entry.generation()
                    )
                });
            }
            #[cfg(debug_assertions)]
            for b in lo..=hi_blk {
                let bi = BlockIndex::new(b as u32);
                let entry = table.get(kind, bi).expect("block index in range").unwrap_or_else(|| {
                    panic!("Layer 3 span [{p0}, {}) missing required {kind:?} block {b}", p0 + t)
                });
                let _ = pool.gpu_table_entry(entry).unwrap_or_else(|err| {
                    panic!(
                        "Layer 3 {kind:?} block {b} entry invalid for slot {} gen {}: {err}",
                        entry.slot().get(),
                        entry.generation()
                    )
                });
            }
            if table.dirty_range(kind).is_some() {
                lane.with_bt_buf(kind == BlockKind::Window, |bt_buf, byte_off, cap_blk| {
                    assert!(n_blk <= cap_blk, "blocked lane GPU table smaller than BlockTable");
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(
                            (bt_buf.contents().as_ptr() as *mut u8).add(byte_off) as *mut u64,
                            n_blk,
                        )
                    };
                    table.sync_dirty_to_slice(kind, pool, dst).unwrap_or_else(|err| {
                        panic!("Layer 3 {kind:?} dirty block-table sync failed: {err}")
                    });
                });
            }
        }
    }
    /// bind the engine's buffers: weights, parameters, scratch arenas, counters
    fn bind_eng(&self, e: &Enc) {
        e.buf(&self.w, 0, 0); e.buf(&self.pb, 0, 1); e.buf(&self.act.borrow(), 0, 2); e.buf(&self.ctr, 0, 7);
    }

    fn seg(w: &QW, in_off: u32, out_off: u32) -> Seg {
        debug_assert!(matches!(w.ty, T_Q4_K | T_Q6_K | T_Q8_0));
        Seg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
              in_off, in2_off: -1, norm_off: -1, ss_off: -1, ss_n: 0, out_off, ..Default::default() }
    }
    fn seg_pb(off: u32, rows: u32, k: u32, in_off: u32, out_off: u32) -> Seg {
        Seg { w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, in_off, in2_off: -1, norm_off: -1, ss_off: -1, ss_n: 0, out_off, ..Default::default() }
    }
    #[inline]
    /// `go` without the barrier: for dispatches that do not depend on each other (one per lane), followed by one barrier
    fn go_nb<T>(&self, e: &Enc, cls: u32, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.gpu.pso(pso));
        e.bytes(args, 5);
        e.dispatch(grid, tg);
    }
    fn go<T>(&self, e: &Enc, cls: u32, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.gpu.pso(pso));
        e.bytes(args, 5);
        e.dispatch(grid, tg);
        e.barrier();
    }
    fn mv(&self, e: &Enc, cls: u32, segs: &[Seg]) { self.mv_post(e, cls, segs, MvArgs::default()); }
    fn mv_post(&self, e: &Enc, cls: u32, segs: &[Seg], post: MvArgs) {
        let mut a = MvArgs { nseg: segs.len() as u32, eps: self.eps, ..post };
        let mut tg = 0u32;
        for (i, s) in segs.iter().enumerate() {
            a.seg[i] = *s;
            a.seg[i].tg_start = tg;
            tg += s.nrows.div_ceil(8);
        }
        self.go(e, cls, "k_mv", &a, (tg as usize, 1, 1), 128);
    }
    /// k_mv for the `n` lanes of a decode batch (segments at lane 0's offsets; lane j's are `j * ar.total` further)
    fn mvb(&self, e: &Enc, cls: u32, segs: &[Seg], post: MvArgs, n: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        let mut a = MvArgs { nseg: segs.len() as u32, eps: self.eps, ..post };
        let mut tg = 0u32;
        for (i, s) in segs.iter().enumerate() { assert!(s.epi == 0); a.seg[i] = *s; a.seg[i].tg_start = tg; tg += s.nrows.div_ceil(8); }
        e.pso(&self.gpu.pso("k_mvb"));
        e.bytes(&a, 5);
        e.bytes(&[n as u32, self.ar.total], 8);
        e.dispatch((tg as usize, 1, 1), 128);
        e.barrier();
    }
    /// decode: one token, the decode arena's layout
    fn mix(&self, e: &Enc, cls: u32, jobs: &[MixJob]) { self.mix_impl(e, cls, jobs, 1, false); }
    /// prefill: `t` tokens in the batch arena's layout. A batch of one token is still a batch: its jobs carry the batch
    /// layout (token stride, fp16 outputs), which the decode kernel does not read.
    fn mix_t(&self, e: &Enc, cls: u32, jobs: &[MixJob], t: usize) { self.mix_impl(e, cls, jobs, t, true); }
    fn mix_impl(&self, e: &Enc, cls: u32, jobs: &[MixJob], t: usize, batch: bool) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        if batch && std::env::var("INK_OLD_MIXB").is_err() {
            // independent jobs run concurrently (no barrier between them)
            for j in jobs {
                let j = MixJob { ntok: t as u32, ..*j };
                let name = if j.q_off < 0 { "k_mix_b_id".to_string() } else { format!("k_mix_b_{}_{}", j.n_s, j.n_q) };
                e.pso(&self.gpu.pso(&name));
                e.bytes(&j, 5);
                e.dispatch((t.div_ceil(4), 1, 1), 128);
            }
            e.barrier();
            return;
        }
        e.pso(&self.gpu.pso("k_mix"));
        e.bytes_slice(jobs, 5);
        e.dispatch((jobs.len(), t, 1), 768);
        e.barrier();
    }
    fn mix_job(&self, states: &[usize], q_off: i32, n_q: u32, outs: &[u32], norms: &[i32]) -> MixJob {
        let offs: Vec<u32> = states.iter().map(|&s| self.v(s)).collect();
        self.mix_job_off(&offs, q_off, n_q, outs, norms, 0)
    }
    fn mix_job_off(&self, states: &[u32], q_off: i32, n_q: u32, outs: &[u32], norms: &[i32], tstride: u32) -> MixJob {
        let mut j = MixJob { n_s: states.len() as u32, n_q, q_off, eps: self.eps, tstride, ..Default::default() };
        assert!(states.len() <= 6);
        for (i, &s) in states.iter().enumerate() { j.s_off[i] = s; }
        for i in 0..4 { j.norm_off[i] = -1; j.ncopy[i] = [-1; 3]; j.hcopy[i] = [-1; 3]; j.hout[i] = -1; }
        j.hstride = (2 * E) as u32;
        for (i, &o) in outs.iter().enumerate() { j.out_off[i] = o; }
        for (i, &n) in norms.iter().enumerate() { j.norm_off[i] = n; }
        j
    }
    /// loop controllers, then RMSNorm of each controlled vector (norm weight w, written to out)
    /// decode loop controllers: g = D.cat[j] (bf16 matvec on the mix kernel's normalized copies), then gain/forget apply
    /// with the RMSNorm fused into the last threadgroup of each controller
    fn ctrl_dec(&self, e: &Enc, jobs: &[CtrlJob]) {
        let ar = self.ar_at();
        let segs: Vec<Seg> = jobs.iter().map(|j| {
            let c = ar.ctrlg.iter().position(|&o| o == j.out_off).unwrap();
            Self::seg_pb(j.d_off, 64, (2 * E) as u32, ar.cat[c], j.out_off)
        }).collect();
        self.mv(e, C_CTRLD, &segs);
        let mut a = CtrlArgs { n_c: jobs.len() as u32, ..Default::default() };
        for (i, j) in jobs.iter().enumerate() { a.job[i] = *j; }
        self.go(e, C_CTRLA, "k_ctrl_apply", &a, (E / 32, jobs.len(), 1), 256);
    }

    fn gemm(&self, e: &Enc, cls: u32, segs: &[GSeg], t: usize) {
        let mut a = GemmArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() { a.seg[i] = *sg; a.seg[i].tile_start = tiles; tiles += sg.nrows.div_ceil(64); }
        self.go(e, cls, if self.tensor { "k_gemm_t" } else { "k_gemm" }, &a, (tiles as usize, t.div_ceil(32), 1), 128);
    }
    /// batched loop controllers as GEMMs: g = D.[u^;m^] (split-input GEMM over the mix kernel's normalized copies),
    /// gain/forget pre-activations = G.g, F.g (K=64 GEMM), then a fused apply + RMSNorm per token
    /// jobs: (controller index c, loop weights, u_off, m_off, out_off, norm weight)
    /// fp16 path: the mix kernels already wrote [u^ | m^] as half into cath; `hout[i]` receives job i's output as half
    /// (`nof32[i]`: only as half)
    fn ctrl_b(&self, e: &Enc, lw: &LoopW, jobs: &[(usize, u32, u32, u32, u32)], t: usize, hout: &[(u32, bool)]) {
        let b = self.ba.get();
        let pb = |off: u32, rows: u32, k: u32, in_off: u32, in2: u32, out: u32| GSeg {
            w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, in_off, out_off: out, ldo: rows, wsrc: 1, in2, ..Default::default() };
        if self.hgemm {
            let fuse_down_half = ctrl_down_direct_half(self.skip.get());
            let down: Vec<HSeg> = jobs.iter().map(|&(c, _, _, _, _)| {
                let mut sg = Self::hseg_pb(
                    lw.down[c],
                    64,
                    (2 * E) as u32,
                    b.cath[c],
                    if fuse_down_half { b.gh[c] } else { b.ctrlg[c] },
                );
                sg.hout = fuse_down_half as u32;
                sg
            }).collect();
            self.gemm_h(e, C_CTRLD, &down, t);
            if !fuse_down_half {
                for &(c, _, _, _, _) in jobs { self.tohalf(e, C_CTRLA, b.ctrlg[c], 64, b.gh[c], 64, 64, t, None); }
                e.barrier();
            }
            // gain/forget GEMM with the apply fused into its epilogue: raw (un-normalized) outputs land in gain_o
            let mut ga = CtrlGfArgs { t: t as u32, njobs: jobs.len() as u32, ..Default::default() };
            for (i, &(c, u, m, _, _)) in jobs.iter().enumerate() {
                ga.job[i] = CtrlGfJob { gain_w: lw.gain[c], forget_w: lw.forget[c], gh_off: b.gh[c], u_off: u, m_off: m, raw_off: b.gain_o[c], pad: [0; 2] };
            }
            self.go(e, C_CTRLA, "k_ctrl_gf", &ga, (t.div_ceil(128), jobs.len() * (E / 32), 1), 256);
        } else {
            let down: Vec<GSeg> = jobs.iter().map(|&(c, _, _, _, _)| pb(lw.down[c], 64, (2 * E) as u32, b.cat_u[c], b.cat_m[c], b.ctrlg[c])).collect();
            self.gemm(e, C_CTRLD, &down, t);
            let mut gf = Vec::new();
            for &(c, _, _, _, _) in jobs {
                gf.push(pb(lw.gain[c], E as u32, 64, b.ctrlg[c], 0, b.gain_o[c]));
                gf.push(pb(lw.forget[c], E as u32, 64, b.ctrlg[c], 0, b.forget_o[c]));
            }
            self.gemm(e, C_CTRLA, &gf, t);
        }
        let mut a = CtrlBArgs { n: jobs.len() as u32, ..Default::default() };
        for (i, &(c, u, m, out, nw)) in jobs.iter().enumerate() {
            let (h, nof) = hout.get(i).map(|&(h, n)| (h as i32, n as u32)).unwrap_or((-1, 0));
            a.job[i] = if self.hgemm { CtrlBJob { u_off: b.gain_o[c], m_off: u32::MAX, g_off: 0, f_off: 0, out_off: out, norm_w: nw, h_off: h, nof32: nof } }
                       else { CtrlBJob { u_off: u, m_off: m, g_off: b.gain_o[c], f_off: b.forget_o[c], out_off: out, norm_w: nw, h_off: h, nof32: nof } };
        }
        self.go(e, C_CTRLN, "k_ctrl_apply_b", &a, (t, jobs.len(), 1), 256);
    }

    /// f32 [rows][k] (row stride src_stride, optional row gather) -> half [rows][dst_stride] at dst
    #[allow(clippy::too_many_arguments)]
    fn tohalf(&self, e: &Enc, cls: u32, src: u32, src_stride: usize, dst: u32, dst_stride: usize, k: usize, rows: usize, perm: Option<u32>) {
        self.tohalf_x(e, cls, src, src_stride, dst, dst_stride, k, rows, perm, false)
    }
    /// `half_src`: the source is already half (row stride in halves): a plain (gathered) copy
    #[allow(clippy::too_many_arguments)]
    fn tohalf_x(&self, e: &Enc, cls: u32, src: u32, src_stride: usize, dst: u32, dst_stride: usize, k: usize, rows: usize, perm: Option<u32>, half_src: bool) {
        let a = ToHalfArgs { src_off: src, dst_off: dst, k: k as u32, rows: rows as u32, perm_off: perm.unwrap_or(0), gather: perm.is_some() as u32 | (half_src as u32) << 1,
                             src_stride: src_stride as u32, dst_stride: dst_stride as u32 };
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.gpu.pso("k_tohalf")); e.bytes(&a, 5); e.dispatch((k.div_ceil(1024), rows, 1), 256);
    }
    /// fp16-activation tensor GEMM over segments (weight, wsrc, half input, output, ldo); barrier after
    fn gemm_h(&self, e: &Enc, cls: u32, segs: &[HSeg], t: usize) {
        let mut a = GemmHArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() { a.seg[i] = *sg; a.seg[i].tile_start = tiles; tiles += sg.nrows.div_ceil(64); }
        if t > 64 { self.go(e, cls, "k_gemm_sg8", &a, (t.div_ceil(128), tiles as usize, 1), 256); }
        else { self.go(e, cls, "k_gemm_sg4", &a, (1, tiles as usize, 1), 128); }
    }
    fn hseg(w: &QW, xh: u32, out: u32) -> HSeg {
        HSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k, xh_off: xh, out_off: out, ldo: w.rows, ..Default::default() }
    }
    fn hseg_pb(off: u32, rows: u32, k: u32, xh: u32, out: u32) -> HSeg {
        HSeg { w_lo: off, w_hi: 0, row_bytes: k * 2, wtype: T_BF16, nrows: rows, k, xh_off: xh, out_off: out, ldo: rows, wsrc: 1, ..Default::default() }
    }

    fn gseg(w: &QW, in_off: u32, out_off: u32) -> GSeg {
        GSeg { w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty, nrows: w.rows, k: w.k,
               in_off, out_off, ldo: w.rows, ..Default::default() }
    }


    /// Encode one decode step for each of `lanes` (lane, position of the token it decodes) in one command buffer, through
    /// the decode path: every lane runs today's single-stream kernels on its own copy of the decode arena with its own
    /// buffers bound, layer by layer, so each lane's numbers are exactly what it gets decoding alone. Logits for each lane
    /// land in its arena (`dbatch_logits`). Needs `ensure_dbatch(lanes.len())`; no disk tier, no pruning.
    /// This is the decode-path reference the `dbatch` harness checks (`INK_DB_OLD`), not the serving path:
    /// `inky serve` decodes through `encode_decode`.
    pub fn encode_dbatch(&self, cb: &CmdBuf, lanes: &[(&Lane, usize)], logits: bool) {
        self.guided_logit_rows.set(0);
        let n = lanes.len();
        let (start, tot) = (self.dstart.get(), self.ar.total);
        assert!(n >= 1 && start >= self.ar.total && start >= self.ba.get().end && self.act.borrow().length() >= (start + n as u32 * tot) as usize * 4,
                "decode batch arenas: call ensure_dbatch() after ensure_batch()");
        for (l, pos) in lanes { assert!(l.tier.is_none() && l.prune.borrow().is_none() && *pos < self.on(l).kv_capacity(), "K/V capacity: call reserve() first"); }
        let base = |j: usize| start + j as u32 * tot;
        let mut cx = Cx { e: Enc::new(cb), cb: None, prior: Vec::new() };
        self.bind_eng(&cx.e);
        for (j, (l, pos)) in lanes.iter().enumerate() {
            self.abase.set(base(j));
            self.bind_lane(&cx.e, l);
            let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                                 norm_off: self.tok_norm, out0: self.v(0), out1: self.v(0), tok_index: *pos as u32, eps: self.eps };
            self.go(&cx.e, C_EMBED, "k_embed", &ea, (1, 1, 1), 256);
        }
        // the residual bookkeeping is the same for every lane: run it once per layer, and every lane's layer from it
        let (mut cur, mut blocks, mut va, mut exec) = (0usize, Vec::<usize>::new(), VecAlloc { next: 1, n: N_VEC }, 0usize);
        let stage_a = std::env::var("INK_DBATCH_A").is_ok();
        let mut layer_all = |cx: &mut Cx, cur: &mut usize, blocks: &mut Vec<usize>, va: &mut VecAlloc, slot: usize, ip: usize, msg: Option<(usize, usize)>, pinned: &[usize]| {
            if !stage_a {
                *cur = self.layer_db(&cx.e, lanes, start, exec, slot, ip, *cur, blocks, msg, va, pinned);
                exec += 1;
                return;
            }
            let mut after = None;
            for (j, (l, pos)) in lanes.iter().enumerate() {
                self.abase.set(base(j));
                self.bind_lane(&cx.e, l);
                self.done.set(exec);
                let (mut b2, mut v2) = (blocks.clone(), va.clone());
                let c2 = self.on(l).layer(cx, *pos, slot, ip, *cur, &mut b2, msg, &mut v2, pinned);
                after = Some((c2, b2, v2));
            }
            let (c2, b2, v2) = after.unwrap();
            *cur = c2; *blocks = b2; *va = v2; exec += 1;
        };
        for ip in 0..LOOP_START { layer_all(&mut cx, &mut cur, &mut blocks, &mut va, ip, ip, None, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END { layer_all(&mut cx, &mut cur, &mut blocks, &mut va, slot_of(pass, ip), ip, msg, &pinned); }
            bank = (*blocks.get(prelude_blocks.len() + 1).unwrap_or(&0), cur);
        }
        for ip in LOOP_END..N_PHYS { layer_all(&mut cx, &mut cur, &mut blocks, &mut va, ip, ip, None, &[]); }
        if logits {
            self.abase.set(base(0));
            let ar = self.ar_at();
            let mut st = blocks.clone(); st.push(cur);
            let mut jb = self.mix_job(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32]);
            jb.tstride = tot;
            self.mix_impl(&cx.e, C_OMIX, &[jb], n, false);
            self.mvb(&cx.e, C_LMHEAD, &[Engine::seg(&self.output, ar.fout, ar.logits)], MvArgs::default(), n);
        }
        self.abase.set(0);
        cx.e.end();
    }

    /// One layer execution of a decode batch: the decode path's layer for every lane, with the row-wise work (mixes,
    /// projections, output projection, router) run once over all lanes and the rest per lane. Each lane's arithmetic is
    /// the decode path's, so its numbers are exactly what it gets decoding alone.
    #[allow(clippy::too_many_arguments)]
    fn layer_db(&self, e: &Enc, lanes: &[(&Lane, usize)], start: u32, exec: usize, slot: usize, ip: usize, cur: usize,
                blocks: &mut Vec<usize>, msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]) -> usize {
        let n = lanes.len();
        let tot = self.ar.total;
        let at = |j: usize| self.abase.set(start + j as u32 * tot);
        let l = &self.layers[ip];
        let ctl = msg.map(|_| &self.loops[ip - LOOP_START]);
        let cs_slot = (slot * NCH * 3) as u32;
        let ts = |mut j: MixJob| { j.tstride = tot; j };
        at(0);
        let ar = self.ar_at();
        // ---- attention inputs: depth mix (+ loop control) -> normalized q/k/v inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st = blocks.clone(); st.push(cur);
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        let xin: [u32; 3];
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &[]);
            if l.swa { ja.ncopy[0] = [ar.cat[0] as i32, ar.cat[1] as i32, ar.cat[2] as i32]; }
            else { for j in 0..3 { ja.ncopy[j][0] = ar.cat[j] as i32; } }
            let mut jm = self.mix_job(&[b0, b1], lw.msg_query as i32, 4, &ar.msg, &[]);
            for j in 0..4 { jm.ncopy[j][0] = (ar.cat[j] + E as u32) as i32; }
            self.mix_impl(e, C_MIXA, &[ts(ja), ts(jm)], n, false);
            for jl in 0..n {
                at(jl);
                let ar = self.ar_at();
                let mut cj = Vec::new();
                for j in 0..3 {
                    let u = if l.swa { ar.inp[0] } else { ar.inp[j] };
                    cj.push(CtrlJob { u_off: u, m_off: ar.msg[j], d_off: lw.down[j], out_off: ar.ctrlg[j], gain_off: lw.gain[j],
                                      forget_off: lw.forget[j], res_out: ar.cin[j], ss_off: ar.ssp[j], norm_w: norms[j], ctr: 8 + j as u32 });
                }
                self.ctrl_dec(e, &cj);
            }
            at(0);
            xin = ar.cin;
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            self.mix_impl(e, C_MIXA, &[ts(self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &nn))], n, false);
            xin = if l.swa { [ar.inp[0]; 3] } else { [ar.inp[0], ar.inp[1], ar.inp[2]] };
        }
        // ---- fused projections, once for every lane
        let mut segs = vec![Engine::seg(&l.wq, xin[0], ar.q), Engine::seg(&l.wgate, xin[0], ar.gate), Engine::seg(&l.wrel, xin[0], ar.rel),
                            Engine::seg(&l.wsink, xin[0], ar.sink), Engine::seg(&l.wk, xin[1], ar.k), Engine::seg(&l.wv, xin[2], ar.v)];
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { segs.push(Engine::seg(kp, xin[1], ar.kdyn)); segs.push(Engine::seg(vp, xin[2], ar.vdyn)); }
        self.mvb(e, C_QKV, &segs, MvArgs::default(), n);
        // ---- attention (split-K, fused K/V conv and cache write) per lane, on the lane's cache and state
        for (jl, (lane, pos)) in lanes.iter().enumerate() {
            at(jl);
            let ar = self.ar_at();
            let run = self.on(lane);
            self.bind_lane(e, lane);
            let pos = *pos;
            let (kb, vb, kcap) = run.kv_slot(slot);
            let tau = if l.swa { 1.0 } else { 1.0 + self.log_alpha * (((pos + 1) as f32 / self.log_floor).max(1.0)).ln() };
            let base = AttnArgs {
                pos: pos as u32, cap: kcap, k_base: kb, v_base: vb, cs_off: cs_slot,
                q_off: ar.q, k_off: ar.k, v_off: ar.v, r_off: ar.rel, sink_off: ar.sink,
                kdyn_off: if l.swa { -1 } else { ar.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { ar.vdyn as i32 }, part_off: ar.part,
                kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias, tau,
                ctr: 0, gate_off: ar.gate, out_off: ar.attn_g, gstr: kcap * HD as u32, istr: HD as u32, ..Default::default()
            };
            run.bind_kv(e, l.swa);
            let (first_pos, kps, _sp_lo, nsplit) = decode_split_grid(l.swa, pos, self.max_ctx);
            let nkeys = pos + 1 - first_pos;
            let aa = AttnArgs { nkeys: nkeys as u32, first_pos: first_pos as u32, ring: l.swa as u32, kps: kps as u32, nsplit: nsplit as u32,
                                pad: self.sep_comb as u32, ..base };
            self.go(e, C_ATTN, "k_attn", &aa, (nsplit, NKV, 1), 128);
            if self.sep_comb {
                let ca = CombineArgs { part_off: ar.part, nsplit: nsplit as u32, gate_off: ar.gate, out_off: ar.attn_g, hout: 0, pad: [0; 3] };
                self.go(e, C_COMB, "k_attn_combine", &ca, (NH, 1, 1), 128);
            }
        }
        at(0);
        // ---- output projection once for every lane, then each lane's post-conv (+ residual) on its own state
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        self.mvb(e, C_WO, &[Engine::seg(&l.wo, ar.attn_g, self.v(mid))], MvArgs::default(), n);
        for (jl, (lane, _)) in lanes.iter().enumerate() {
            at(jl);
            self.bind_lane(e, lane);
            let ea = EpiArgs { out_off: self.v(mid), res_off: if block_start { -1 } else { self.v(cur) as i32 }, cs_off: cs_slot + (2 * NKV * HD * 3) as u32,
                               convw_off: l.attn_post_conv, n: E as u32, pad: [0; 3] };
            self.go(e, C_WO, "k_epi", &ea, (E.div_ceil(256), 1, 1), 256);
        }
        at(0);
        if block_start { blocks.push(cur); }
        // ---- FFN input: depth mix (+ control) -> normalized
        let mut st = blocks.clone(); st.push(mid);
        let x_off = if let Some(lw) = ctl {
            let mut jf = self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[]);
            jf.ncopy[0][0] = ar.cat[3] as i32;
            self.mix_impl(e, C_MIXF, &[ts(jf)], n, false);
            for jl in 0..n {
                at(jl);
                let ar = self.ar_at();
                self.ctrl_dec(e, &[CtrlJob { u_off: ar.fin, m_off: ar.msg[3], d_off: lw.down[3], out_off: ar.ctrlg[3], gain_off: lw.gain[3],
                                             forget_off: lw.forget[3], res_out: ar.finc, ss_off: ar.ssp[3], norm_w: l.ffn_norm as i32, ctr: 11 }]);
            }
            at(0);
            ar.finc
        } else {
            self.mix_impl(e, C_MIXF, &[ts(self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[l.ffn_norm as i32]))], n, false);
            ar.fin
        };
        // ---- router with fused top-k, once for every lane
        assert!(!self.sep_topk);
        let sel = ar.sel_log + (exec * 16) as u32;
        let rpost = MvArgs { post: 1, ctr: 12, bias_off: l.exp_bias, sel_off: sel, wscale: self.wscale, ..Default::default() };
        self.mvb(e, C_ROUTER, &[Engine::seg_pb(l.router, NEXP as u32, E as u32, x_off, ar.rlog)], rpost, n);
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        // ---- experts per lane (the down projection's epilogue updates the lane's state)
        for (jl, (lane, _)) in lanes.iter().enumerate() {
            at(jl);
            let ar = self.ar_at();
            self.bind_lane(e, lane);
            let sel = ar.sel_log + (exec * 16) as u32;
            let x = if ctl.is_some() { ar.finc } else { ar.fin };
            let ma = MoeArgs {
                logits_off: ar.rlog, bias_off: l.exp_bias, sel_off: sel, x_off: x,
                norm_off: -1, ss_off: -1, ss_n: 0, eps: self.eps,
                gu_lo: l.gu.off as u32, gu_hi: (l.gu.off >> 32) as u32, gu_expert_bytes: l.gu.expert_bytes as u32, gu_row_bytes: l.gu.row_bytes,
                act_off: ar.act, wscale: self.wscale, beta: self.situ_beta, lbeta: self.situ_lbeta,
                dn_lo: l.dn.off as u32, dn_hi: (l.dn.off >> 32) as u32, dn_expert_bytes: l.dn.expert_bytes as u32, dn_row_bytes: l.dn.row_bytes,
                dn_type: l.dn.ty, out_off: self.v(out), res_off: self.v(mid) as i32, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
                convw_off: l.ffn_post_conv, gu_type: l.gu.ty, gu_gs_lo: l.gu.gs as u32, gu_gs_hi: (l.gu.gs >> 32) as u32,
                dn_gs_lo: l.dn.gs as u32, dn_gs_hi: (l.dn.gs >> 32) as u32, no_state_write: 0, pad1: 0,
            };
            self.go(e, C_GU, "k_moe_gu", &ma, (FF / 4, TOPK, 1), 128);
            self.go(e, C_DOWN, "k_moe_down", &ma, (E / 4, 1, 1), 64);
        }
        at(0);
        out
    }

    /// Encode one step over `spans`, each lane's positions [p0, p0 + t), as one batch: row-wise work (mixes, controllers,
    /// projections, router, experts) runs once over every span's rows; what couples a row to its lane (embedding, K/V
    /// preparation and cache writes, attention, the convolutions and their state) runs span by span with that lane's
    /// buffers bound. Optionally computes each span's logits (and argmax, written to the lane's tokens) for its last row:
    /// in the decode arena for a single span (`logits()`), in the batch arena for several (`span_logits`).
    pub fn encode_spans(&self, cb: &CmdBuf, spans: &[Span], logits: bool, argmax: bool) { self.encode_spans_x(cb, spans, logits, argmax, false, None, None); }
    /// Verify step. Each span has length `1 + k` (the last committed token again, then `k` drafts). `k` is the
    /// argument, not a fixed 8. Writes one vocabulary row per position into `b.lg`; `span_logits` names the row.
    /// Argmax is off: that flag writes `toks[p0 + t]`, where the host has just placed the draft tokens.
    /// Long prefills stay on `encode_spans` (one row per span, the last). This does not grow `lg` to one row per token.
    pub fn encode_verify(&self, cb: &CmdBuf, spans: &[Span], k: usize) {
        assert!(!spans.is_empty() && spans.iter().all(|s| s.t == 1 + k), "a verify span has length 1 + k");
        // A --kv-disk lane stages cache from the single-token step path. This span does not, and it must not run there.
        assert!(spans.iter().all(|s| s.lane.tier.is_none()), "verify does not run on a --kv-disk lane");
        let t = 1 + k;
        let stride = N_SLOT * NCH * 3;
        let cs = self.gpu.buffer(spans.len() * t * stride * 4);
        unsafe { std::ptr::write_bytes(cs.contents().as_ptr() as *mut u8, 0, spans.len() * t * stride * 4); }
        let kept = spans.iter().map(|sp| VerifySpan {
            lane: sp.lane as *const Lane,
            p0: sp.p0,
            kv: self.snapshot_kv_positions(sp.lane, sp.p0, sp.t),
            accepted_end: None,
        }).collect();
        *self.verify_keep.borrow_mut() = Some(VerifyKeep { cs, stride, t, spans: kept });
        self.encode_spans_x(cb, spans, true, false, false, Some(k), None);
    }
    /// Put span `si` of the last `encode_verify` back to its first `n_keep` positions (1..=1+k).
    /// The convolution image at that position replaces `cs`. K/V rows past it are put back to the bytes from before
    /// the span, so a rejected draft and anything after it do not stay. The recorded length is `p0 + n_keep`, never
    /// a seal of the span's end. The images live on `verify_keep`, not in serve's `cs_at_prompt`. Nothing may be in
    /// flight on the GPU. The correction token is not written here: its K and V land when that token is encoded.
    pub fn rollback_verify(&self, lane: &Lane, si: usize, n_keep: usize) {
        assert!(lane.tier.is_none(), "verify does not run on a --kv-disk lane");
        let tail = {
            let mut slot = self.verify_keep.borrow_mut();
            let g = slot.as_mut().expect("rollback_verify without encode_verify");
            assert!(si < g.spans.len(), "verify span {si} is out of range");
            let t = g.t;
            let stride = g.stride;
            let cs = g.cs.clone();
            assert!(n_keep >= 1 && n_keep <= t, "n_keep {n_keep} is outside the verify span of {t}");
            let rec = &mut g.spans[si];
            assert!(std::ptr::eq(rec.lane, lane), "rollback lane is not the lane encode_verify ran");
            let end = rec.p0 + n_keep;
            // Layer 2 has no block table. The only length this step records is the accepted prefix.
            rec.accepted_end = Some(end);
            assert!(end <= rec.p0 + t);
            if n_keep < t { assert!(end < rec.p0 + t, "a rejected tail is not sealed"); }
            let bytes = stride * 4;
            lane.with_cs_buf(|dst_buf, dst_off, dst_len| {
                assert_eq!(bytes, dst_len);
                unsafe {
                    let src = (cs.contents().as_ptr() as *const u8).add((si * t + (n_keep - 1)) * bytes);
                    std::ptr::copy_nonoverlapping(src, (dst_buf.contents().as_ptr() as *mut u8).add(dst_off), bytes);
                }
            });
            rec.kv.iter().skip(n_keep).cloned().collect::<Vec<_>>()
        };
        for undo in &tail { self.restore_kv_pos(lane, undo); }
    }
    /// Bytes of every slot's K and V row at positions [p0, p0+t), before the span writes them.
    fn snapshot_kv_positions(&self, lane: &Lane, p0: usize, t: usize) -> Vec<KvUndo> {
        let eb = self.kvb;
        let row = HD * eb;
        let mut out = Vec::with_capacity(t);
        for d in 0..t {
            let pos = p0 + d;
            let mut rows = Vec::new();
            for slot in 0..N_SLOT {
                let (swa, i) = self.slot_kind[slot];
                let cap = if swa { lane.swa_cap.get() } else { lane.full_cap.get() };
                assert!(swa || pos < cap, "full-attention row {pos} is past the cache");
                let idx = if swa { pos % cap } else { pos };
                let buf = if swa { lane.kv_swa.borrow() } else { lane.kv_full.borrow() };
                let base = buf.contents().as_ptr() as *const u8;
                let k0 = i * 2 * NKV * cap * HD * eb;
                for which in 0..2 {
                    let b0 = k0 + which * NKV * cap * HD * eb;
                    for g in 0..NKV {
                        let off = b0 + (g * cap + idx) * row;
                        rows.extend_from_slice(unsafe { std::slice::from_raw_parts(base.add(off), row) });
                    }
                }
            }
            out.push(KvUndo { pos, rows });
        }
        out
    }
    fn restore_kv_pos(&self, lane: &Lane, undo: &KvUndo) {
        let eb = self.kvb;
        let row = HD * eb;
        let pos = undo.pos;
        let mut at = 0usize;
        for slot in 0..N_SLOT {
            let (swa, i) = self.slot_kind[slot];
            let cap = if swa { lane.swa_cap.get() } else { lane.full_cap.get() };
            let idx = if swa { pos % cap } else { pos };
            let buf = if swa { lane.kv_swa.borrow() } else { lane.kv_full.borrow() };
            let base = buf.contents().as_ptr() as *mut u8;
            let k0 = i * 2 * NKV * cap * HD * eb;
            for which in 0..2 {
                let b0 = k0 + which * NKV * cap * HD * eb;
                for g in 0..NKV {
                    let off = b0 + (g * cap + idx) * row;
                    unsafe { std::ptr::copy_nonoverlapping(undo.rows.as_ptr().add(at), base.add(off), row); }
                    at += row;
                }
            }
        }
        assert_eq!(at, undo.rows.len());
    }
    /// Write this conv site's state after every prefix of the span. Runs only while a verify span is recording,
    /// and only before `k_state_upd` overwrites `cs` with the state at the end.
    fn keep_state(&self, e: &Enc, si: usize, su: &StateArgs, cls: u32, grid: (usize, usize, usize)) {
        if !self.verify_record.get() { return; }
        let (sk, buf) = {
            let slot = self.verify_keep.borrow();
            let g = slot.as_ref().expect("verify recording without convolution images");
            let sk = StateKeepArgs {
                x_off: su.x_off, ldx: su.ldx, cs_off: su.cs_off, nch: su.nch, t: su.t,
                snap_off: (si * g.t * g.stride) as u32, stride: g.stride as u32, pad: 0,
            };
            (sk, g.cs.clone())
        };
        e.buf(&buf, 0, 9);
        self.go(e, cls, "k_state_keep", &sk, grid, 256);
    }
    /// Encode one decode step: one token for each lane in `spans` (every span has t = 1). Row-wise work runs through the
    /// batched tensor path as in `encode_spans`; each lane's attention is the decode path's fused kernel (K/V conv, cache
    /// write, split-K attention over its own cache). A decode step's arithmetic is its own: a token's numbers depend
    /// neither on the other lanes in the step nor on how many there are, but differ from a prefill row's, so cache
    /// blocks written by decode steps are never shared (docs/DESIGN.md).
    pub fn encode_decode(&self, cb: &CmdBuf, spans: &[Span], logits: bool, argmax: bool) {
        assert!(spans.iter().all(|s| s.t == 1));
        self.encode_spans_x(cb, spans, logits, argmax, true, None, None);
    }
    /// Encode one decode step with explicit per-span `omega_max` values for adaptive `LoopCD-Logits`
    /// (`0.0` disables LoopCD for that span). When every entry is `0.0`, behaves identically to `encode_decode`
    /// with zero extra buffers or dispatches.
    pub fn encode_decode_with_omegas(&self, cb: &CmdBuf, spans: &[Span], logits: bool, argmax: bool, omegas: &[f32]) {
        assert!(spans.iter().all(|s| s.t == 1));
        assert_eq!(omegas.len(), spans.len(), "encode_decode_with_omegas: one omega per span");
        self.encode_spans_x(cb, spans, logits, argmax, true, None, Some(omegas));
    }
    fn encode_spans_x(&self, cb: &CmdBuf, spans: &[Span], logits: bool, argmax: bool, dec: bool, verify_k: Option<usize>, omegas: Option<&[f32]>) {
        self.guided_logit_rows.set(0);
        assert!(!spans.is_empty() && spans.len() <= MAX_SPANS);
        let lane_omegas: Vec<f32> = if dec && logits {
            match omegas {
                Some(ws) => ws.to_vec(),
                None => vec![self.loopcd_omega.get(); spans.len()],
            }
        } else {
            Vec::new()
        };
        let run_loopcd = dec && logits && lane_omegas.iter().any(|&w| w > 0.0);
        let t: usize = spans.iter().map(|s| s.t).sum();
        let blocked = spans[0].lane.use_blocks.get();
        assert!(spans.iter().all(|s| s.lane.use_blocks.get() == blocked), "cannot mix Layer 2 and Layer 3 lanes in one step");
        if blocked {
            for sp in spans { self.sync_lane_block_table(sp.lane, sp.p0, sp.t); }
        } else {
            for sp in spans { assert!(sp.t >= 1 && sp.p0 + sp.t <= self.on(sp.lane).kv_capacity(), "K/V capacity: call reserve() first"); }
        }
        assert!(spans.len() == 1 || self.hgemm, "several spans need the fp16 path");
        assert!(!dec || (self.hgemm && spans.len() <= self.decode_span_limit()), "a decode step needs the fp16 path and room for f32 queries");
        assert!(!dec || (self.dstart.get() >= self.ba.get().end && self.act.borrow().length() >= (self.dstart.get() as usize + spans.len() * self.ar.total as usize + decode_batch_extra_floats(spans.len())) * 4),
                "a decode step keeps each lane's split partials and descriptors in the decode-batch region: call ensure_dbatch(n) after ensure_batch()");
        assert!(t <= self.chunk && t <= self.ba.get().cap, "batch arena: call ensure_batch() first");
        if let Some(k) = verify_k {
            assert!(!dec && !argmax, "verify is not a decode step, and argmax would overwrite toks[p0 + t]");
            let need = spans.len() * (1 + k);
            assert!(need <= self.ba.get().lg_rows as usize,
                    "verify needs {need} vocabulary rows and this batch has {}; lg is not grown to one row per token",
                    self.ba.get().lg_rows);
        }
        let mut row0 = Vec::with_capacity(spans.len());
        { let mut r = 0; for sp in spans { row0.push(r); r += sp.t; } }
        let e = Enc::new(cb);
        self.bind_eng(&e);
        let b = self.ba.get();
        let dec_meta = if dec && blocked {
            let n = spans.len();
            let align64 = |x: usize| x.div_ceil(64) * 64;
            let grp_floats = std::mem::size_of::<DecAttnGroup>() / 4;
            let desc_off = self.dstart.get() + n as u32 * self.ar.total;
            let kv_cur_off = desc_off + align64(n * (std::mem::size_of::<DecodeSpanDesc>() / 4)) as u32;
            let swa_grp_off = kv_cur_off + align64(n * 2 * NKV * HD) as u32;
            let full_grp_off = swa_grp_off + align64(n * MAX_SWA_SPLIT * grp_floats) as u32;
            let swa_kps = decode_split_kps(true, self.max_ctx);
            let full_kps = decode_split_kps(false, self.max_ctx);
            let mut swa_ranges = Vec::with_capacity(n);
            let mut full_ranges = Vec::with_capacity(n);
            let slab = self.blocked_slab.borrow();
            {
                let act = self.act.borrow();
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        (act.contents().as_ptr() as *mut f32).add(desc_off as usize) as *mut DecodeSpanDesc,
                        n,
                    )
                };
                for (si, sp) in spans.iter().enumerate() {
                    let slot = sp.lane.blocked_slot.as_ref().expect("blocked lane missing slab slot");
                    assert!(Rc::ptr_eq(&slot.slab, &self.blocked_slab), "blocked lane belongs to a different Engine");
                    let pos = sp.p0;
                    assert!(pos * 4 < sp.lane.toks.length(), "decode pos {pos} out of bounds for lane.toks");
                    let tok_id = unsafe { *(sp.lane.toks.contents().as_ptr() as *const i32).add(pos) };
                    let (swa_first_pos, _sk, _swa_sp_lo, swa_nsplit) = decode_split_grid(true, pos, self.max_ctx);
                    let swa_nkeys = pos + 1 - swa_first_pos;
                    let (_full_first_pos, _fk, _full_sp_lo, full_nsplit) = decode_split_grid(false, pos, self.max_ctx);
                    let full_nkeys = pos + 1;
                    let full_tau = 1.0 + self.log_alpha * (((pos + 1) as f32 / self.log_floor).max(1.0)).ln();
                    swa_ranges.push((swa_first_pos, pos));
                    full_ranges.push((0usize, pos));
                    dst[si] = DecodeSpanDesc {
                        tok_id,
                        pos: pos as u32,
                        cs_base: slab.cs_f32_base(slot.slot_idx),
                        bt_full_base: u32::try_from(slab.bt_u64_base(slot.slot_idx, false)).expect("bt_full_base fits u32"),
                        bt_win_base: u32::try_from(slab.bt_u64_base(slot.slot_idx, true)).expect("bt_win_base fits u32"),
                        part_off: self.dstart.get() + si as u32 * self.ar.total + self.ar.part,
                        swa_nkeys: swa_nkeys as u32,
                        swa_first_pos: swa_first_pos as u32,
                        swa_kps: swa_kps as u32,
                        swa_nsplit: swa_nsplit as u32,
                        full_nkeys: full_nkeys as u32,
                        full_kps: full_kps as u32,
                        full_nsplit: full_nsplit as u32,
                        full_tau,
                        swa_tau: 1.0,
                        pad1: 0,
                    };
                }
            }
            // Build grouped shared-KV decode descriptors ONCE for all 40 SWA passes and all 8 full passes.
            let no_grp = self.no_grp_dec.get();
            let tables: Vec<std::cell::Ref<'_, crate::blocks::BlockTable>> =
                spans.iter().map(|sp| sp.lane.block_table.borrow()).collect();
            let swa_groups = build_decode_groups(&swa_ranges, swa_kps, no_grp, |si, blk| {
                tables[si]
                    .get(crate::blocks::BlockKind::Window, crate::blocks::BlockIndex::new(blk as u32))
                    .ok()
                    .flatten()
                    .map(|e| (e.slot().get(), e.generation()))
            });
            let full_groups = build_decode_groups(&full_ranges, full_kps, no_grp, |si, blk| {
                tables[si]
                    .get(crate::blocks::BlockKind::Full, crate::blocks::BlockIndex::new(blk as u32))
                    .ok()
                    .flatten()
                    .map(|e| (e.slot().get(), e.generation()))
            });
            debug_assert!(swa_groups.len() <= n * MAX_SWA_SPLIT);
            debug_assert!(full_groups.len() <= n * MAX_SPLIT);
            let partition_copy = |groups: &[DecAttnGroup], base_off: u32| -> (u32, usize, u32, usize) {
                let mut n_solo = 0usize;
                for g in groups { if g.count == 1 { n_solo += 1; } }
                let n_shared = groups.len() - n_solo;
                let shared_off = base_off + (n_solo * grp_floats) as u32;
                let act = self.act.borrow();
                unsafe {
                    let dst_solo = (act.contents().as_ptr() as *mut f32).add(base_off as usize) as *mut DecAttnGroup;
                    let dst_shared = (act.contents().as_ptr() as *mut f32).add(shared_off as usize) as *mut DecAttnGroup;
                    let (mut is, mut ig) = (0usize, 0usize);
                    for &g in groups {
                        if g.count == 1 {
                            *dst_solo.add(is) = g;
                            is += 1;
                        } else {
                            *dst_shared.add(ig) = g;
                            ig += 1;
                        }
                    }
                }
                (base_off, n_solo, shared_off, n_shared)
            };
            let (swa_solo_off, n_swa_solo, swa_shared_off, n_swa_shared) = partition_copy(&swa_groups, swa_grp_off);
            let (full_solo_off, n_full_solo, full_shared_off, n_full_shared) = partition_copy(&full_groups, full_grp_off);
            drop(tables);
            // Bind shared blocked `CS` (buffer 4) and `BT` (buffer 8) ONCE for the entire decode step.
            e.buf(slab.cs_buf.as_ref().expect("blocked slab missing cs_buf"), 0, 4);
            e.buf(slab.bt_buf.as_ref().expect("blocked slab missing bt_buf"), 0, 8);
            let ea = EmbedDecArgs {
                emb_lo: self.tok_embd.off as u32,
                emb_hi: (self.tok_embd.off >> 32) as u32,
                row_bytes: self.tok_embd.row_bytes,
                norm_off: self.tok_norm,
                out_off: b.vec(0),
                desc_off,
                eps: self.eps,
                pad: 0,
            };
            self.go(&e, C_EMBED, "k_embed_dec", &ea, (n, 1, 1), 256);
            Some(DecBatchMeta {
                desc_off,
                kv_cur_off,
                swa_solo_off,
                swa_shared_off,
                n_swa_solo,
                n_swa_shared,
                full_solo_off,
                full_shared_off,
                n_full_solo,
                n_full_shared,
                swa_kps: swa_kps as u32,
                full_kps: full_kps as u32,
            })
        } else {
            self.bind_lane(&e, spans[0].lane);
            // the window K/V scratch (sk, then sv) starts every prefill at zero: see k_zero (a decode step or Layer 3 block path does not use it)
            if !dec && !blocked {
                let bz = self.ba.get();
                let zr = [bz.sk, bz.perm - bz.sk];
                e.pso(&self.gpu.pso("k_zero")); e.bytes(&zr, 5); e.dispatch(((zr[1] as usize).div_ceil(1024), 1, 1), 256);
                e.barrier();
            }
            for (si, sp) in spans.iter().enumerate() {
                let r0 = (row0[si] * E) as u32;
                let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                                     norm_off: self.tok_norm, out0: b.vec(0) + r0, out1: b.vec(0) + r0, tok_index: sp.p0 as u32, eps: self.eps };
                e.buf(&sp.lane.toks, 0, 6);
                self.go_nb(&e, C_EMBED, "k_embed", &ea, (sp.t, 1, 1), 256);
            }
            e.barrier();
            None
        };
        // Last layer: nothing after the batch reads its outputs except the final token (logits) and the conv states (last 3
        // rows of the two branch outputs). It is a full-attention layer, so K/V still cover every token, but queries, attention,
        // output projection and FFN only run from row t-8: the conv/residual chain makes rows >= t-2 of the result exact.
        let mut trim = [0usize; N_PHYS];
        // Verify needs every row's logits, so it does not take the last-layer tail trim (single span, t > 16).
        if verify_k.is_none() && spans.len() == 1 && self.hgemm && self.tail.get() && t > 16 { trim[N_PHYS - 1] = t - 8; }
        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        if verify_k.is_some() { self.verify_record.set(true); }
        for ip in 0..LOOP_START { cur = self.layer_b_ex(&e, spans, &row0, t, dec, dec_meta, ip, ip, cur, &mut blocks, None, &mut va, &[], trim[ip], false); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        let mut lg_weak_buf: Option<Buf> = None;
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END { cur = self.layer_b_ex(&e, spans, &row0, t, dec, dec_meta, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned, 0, false); }
            bank = (blocks[prelude_blocks.len() + 1], cur);
            if pass == 0 && run_loopcd {
                // Exact dry-state weak reference pass (z_1) immediately after pass 0, while CS[18..24] and
                // KV[18..24] still hold the untouched pos - 1 state.
                let mut cur_w = cur;
                let mut blocks_w = blocks.clone();
                let pinned_w = [prelude, bank.0, bank.1];
                for ip in LOOP_END..N_PHYS {
                    cur_w = self.layer_b_ex(&e, spans, &row0, t, dec, dec_meta, ip, ip, cur_w, &mut blocks_w, None, &mut va, &pinned_w, 0, true);
                }
                let n = spans.len();
                let st_w: Vec<u32> = blocks_w.iter().map(|&v| b.vec(v)).chain(std::iter::once(b.vec(cur_w))).collect();
                let mut j = self.mix_job_off(&st_w, self.output_res_query as i32, 1, &[b.fin], &[self.output_norm as i32], E as u32);
                j.hout[0] = b.xh[0] as i32; j.nof32 = 1;
                self.mix_t(&e, C_OMIX, &[j], n);
                let mut hs = Engine::hseg(&self.output, b.xh[0], b.lg);
                hs.ldo = self.vocab_pad() as u32;
                self.gemm_h(&e, C_LMHEAD, &[hs], n);
                let weak_buf = self.ensure_lg_weak(n);
                let n_words = n * self.vocab_pad();
                let dc = DevCopyArgs { src: b.lg, dst: 0, n: n_words as u32, pad: 0 };
                e.buf(&weak_buf, 0, 3);
                self.go(&e, C_LMHEAD, "k_devcopy", &dc, (n_words.div_ceil(1024), 1, 1), 1024);
                e.buf(&self.act.borrow(), 0, 3);
                lg_weak_buf = Some(weak_buf);
            }
        }
        for ip in LOOP_END..N_PHYS { cur = self.layer_b_ex(&e, spans, &row0, t, dec, dec_meta, ip, ip, cur, &mut blocks, None, &mut va, &[], trim[ip], false); }
        self.verify_record.set(false);
        // Three logit destinations stay as they are. A single-span prefill reads ar.logits. A multi-span prefill
        // reads b.lg, one row per span. Decode writes gemm_h into b.lg with ldo = vocab_pad. Verify is a fourth
        // use of b.lg, for that span only: 1+k rows, one per position, argmax off.
        if let Some(k) = verify_k {
            let ar = self.ar;
            let rows = 1 + k;
            let vp = self.vocab_pad();
            for (si, sp) in spans.iter().enumerate() {
                for r in 0..sp.t {
                    let row = ((row0[si] + r) * E) as u32;
                    let lg = b.lg + ((si * rows + r) * vp) as u32;
                    let mut st: Vec<u32> = blocks.iter().map(|&v| b.vec(v) + row).collect();
                    st.push(b.vec(cur) + row);
                    let j = self.mix_job_off(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32], 0);
                    self.mix(&e, C_OMIX, &[j]);
                    self.mv(&e, C_LMHEAD, &[Engine::seg(&self.output, ar.fout, lg)]);
                }
            }
        } else if logits && dec {
            // a decode step: every lane's last row through the output mix and one tensor GEMM over the output matrix (read
            // once for all lanes, whatever their number); logits in the batch arena (`span_logits`)
            let ar = self.ar;
            let n = spans.len();
            let st: Vec<u32> = blocks.iter().map(|&v| b.vec(v)).chain(std::iter::once(b.vec(cur))).collect();
            let mut j = self.mix_job_off(&st, self.output_res_query as i32, 1, &[b.fin], &[self.output_norm as i32], E as u32);
            j.hout[0] = b.xh[0] as i32; j.nof32 = 1;
            self.mix_t(&e, C_OMIX, &[j], n);
            let mut hs = Engine::hseg(&self.output, b.xh[0], b.lg);
            hs.ldo = self.vocab_pad() as u32;
            self.gemm_h(&e, C_LMHEAD, &[hs], n);
            if let Some(weak_buf) = lg_weak_buf.as_ref() {
                let la = LoopCdArgs {
                    logits_off: b.lg,
                    n: self.n_vocab as u32,
                    stride: self.vocab_pad() as u32,
                    pad0: 0,
                    softcap: self.softcap,
                    pad1: 0,
                    pad2: 0,
                    pad3: 0,
                };
                e.bytes_slice(&lane_omegas, 8);
                e.buf(weak_buf, 0, 9);
                self.go(&e, C_LMHEAD, "k_loopcd_blend", &la, (1, n, 1), 1024);
                self.guided_logit_rows.set(guided_rows(&lane_omegas));
            }
            if argmax {
                for (si, sp) in spans.iter().enumerate() {
                    let nparts = 128u32;
                    let am = ArgmaxArgs { logits_off: b.lg + (si * self.vocab_pad()) as u32, n: self.n_vocab as u32, part_off: ar.amax, nparts, out_index: (sp.p0 + 1) as u32, checked: 0, pad: [0; 2] };
                    e.buf(&sp.lane.toks, 0, 6);
                    self.go(&e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
                    self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
                }
            }
        } else if logits {
            let ar = self.ar;
            for (si, sp) in spans.iter().enumerate() {
                let last = ((row0[si] + sp.t - 1) * E) as u32;
                let lg = if spans.len() == 1 { ar.logits } else { b.lg + (si * self.vocab_pad()) as u32 };
                let mut st: Vec<u32> = blocks.iter().map(|&v| b.vec(v) + last).collect();
                st.push(b.vec(cur) + last);
                let j = self.mix_job_off(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32], 0);
                self.mix(&e, C_OMIX, &[j]);
                self.mv(&e, C_LMHEAD, &[Engine::seg(&self.output, ar.fout, lg)]);
                if argmax {
                    let nparts = 128u32;
                    let am = ArgmaxArgs { logits_off: lg, n: self.n_vocab as u32, part_off: ar.amax, nparts, out_index: (sp.p0 + sp.t) as u32, checked: 0, pad: [0; 2] };
                    e.buf(&sp.lane.toks, 0, 6);
                    self.go(&e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
                    self.go(&e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
                }
            }
        }
        e.end();
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_b_ex(&self, e: &Enc, spans: &[Span], row0: &[usize], t: usize, dec: bool, dec_meta: Option<DecBatchMeta>, slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
                  msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize], ts: usize, dry_state: bool) -> usize {
        let l = &self.layers[ip];
        let b = self.ba.get();
        let vo = |i: usize| b.vec(i);
        let ctl = msg.map(|_| &self.loops[ip - LOOP_START]);
        let cs_slot = (slot * NCH * 3) as u32;
        let tt = t as u32;
        let es = E as u32;
        let hatt = self.hgemm;      // fp16 path: k_attn_h, half q / gate, router on the fp16 FFN input
        let (qgh, rth) = (hatt, hatt);
        // the query side (queries, attention, output projection, FFN) covers rows [ts, t); the key side always all rows
        assert!(ts == 0 || (ctl.is_none() && hatt));
        let n = t - ts;
        let nn_ = n as u32;
        let sh = |off: u32, dim: usize| off + (ts * dim) as u32;
        // ---- attention inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st: Vec<u32> = blocks.iter().map(|&v| vo(v)).collect(); st.push(vo(cur));
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        let xin: [u32; 3];
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = self.mix_job_off(&st, l.attn_res_query, nq, &b.inp[..nq as usize], &[], es);
            let mut jm = self.mix_job_off(&[vo(b0), vo(b1)], lw.msg_query as i32, 4, &b.msg, &[], es);
            if self.hgemm {
                // normalized copies straight into the controllers' fp16 inputs [u^ | m^]
                if l.swa { ja.hcopy[0] = [b.cath[0] as i32, b.cath[1] as i32, b.cath[2] as i32]; }
                else { for j in 0..3 { ja.hcopy[j][0] = b.cath[j] as i32; } }
                for j in 0..4 { jm.hcopy[j][0] = (b.cath[j] + (E / 2) as u32) as i32; }
            } else {
                if l.swa { ja.ncopy[0] = [b.cat_u[0] as i32, b.cat_u[1] as i32, b.cat_u[2] as i32]; }
                else { for j in 0..3 { ja.ncopy[j][0] = b.cat_u[j] as i32; } }
                for j in 0..4 { jm.ncopy[j][0] = b.cat_m[j] as i32; }
            }
            self.mix_t(e, C_MIXA, &[ja, jm], t);
            let jobs: Vec<(usize, u32, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { b.inp[0] } else { b.inp[j] }, b.msg[j], b.cin[j], norms[j] as u32)).collect();
            let ho: Vec<(u32, bool)> = if self.hgemm { (0..3).map(|j| (b.xh[j], true)).collect() } else { vec![] };
            self.ctrl_b(e, lw, &jobs, t, &ho);
            xin = b.cin;
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            let mut ja = self.mix_job_off(&st, l.attn_res_query, nq, &b.inp[..nq as usize], &nn, es);
            if self.hgemm { for q in 0..nq as usize { ja.hout[q] = b.xh[q] as i32; } ja.nof32 = 0xF; }
            self.mix_t(e, C_MIXA, &[ja], t);
            xin = if l.swa { [b.inp[0]; 3] } else { [b.inp[0], b.inp[1], b.inp[2]] };
        }
        // ---- projections (one batched GEMM over all segments)
        let mut segs = vec![Engine::gseg(&l.wq, xin[0], b.q), Engine::gseg(&l.wgate, xin[0], b.gate), Engine::gseg(&l.wrel, xin[0], b.rel),
                            Engine::gseg(&l.wsink, xin[0], b.sink), Engine::gseg(&l.wk, xin[1], b.k), Engine::gseg(&l.wv, xin[2], b.v)];
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { segs.push(Engine::gseg(kp, xin[1], b.kdyn)); segs.push(Engine::gseg(vp, xin[2], b.vdyn)); }
        if self.hgemm {
            // the fp16 inputs b.xh[..] were written by the mix / controller kernels: one shared input in an uncontrolled
            // sliding-window layer, separate q / k / v inputs otherwise
            let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
            let mut hq = vec![Engine::hseg(&l.wq, b.xh[xi[0]], b.q), Engine::hseg(&l.wgate, b.xh[xi[0]], b.gate), Engine::hseg(&l.wrel, b.xh[xi[0]], b.rel),
                              Engine::hseg(&l.wsink, b.xh[xi[0]], b.sink)];
            // q and the output gate are only read once more (K/V prep, attention epilogue): write them as half
            if hatt && qgh && !dec { hq[0].hout = 1; hq[1].hout = 1; }     // a decode step keeps them f32 for k_attn
            let mut hs = vec![Engine::hseg(&l.wk, b.xh[xi[1]], b.k), Engine::hseg(&l.wv, b.xh[xi[2]], b.v)];
            if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { hs.push(Engine::hseg(kp, b.xh[xi[1]], b.kdyn)); hs.push(Engine::hseg(vp, b.xh[xi[2]], b.vdyn)); }
            if ts == 0 { hq.extend_from_slice(&hs); self.gemm_h(e, C_QKV, &hq, t); }
            else {
                // query-side projections for the tail rows only (the two GEMMs are independent)
                for sg in hq.iter_mut() { sg.xh_off = sh(sg.xh_off, E / 2); sg.out_off = sh(sg.out_off, sg.nrows as usize / (1 + sg.hout as usize)); }
                self.gemm_h(e, C_QKV, &hs, t);
                self.gemm_h(e, C_QKV, &hq, n);
            }
        } else {
            self.gemm(e, C_QKV, &segs, t);
        }
        // ---- K/V conv + cache write, Q/R prep, attention: span by span, each with its lane's cache and convolution rows.
        // Sliding-window layers: a span attends through a linear scratch [window history | span keys] in the arena (reused by
        // the next span: every dispatch ends with a barrier), and the ring (exactly one window) only receives the last window
        // of keys. Full layers address their cache directly.
        let scap = (EXT + b.cap + 2 * KGUARD) as u32;
        let blocked = spans[0].lane.use_blocks.get();
        let pass_idx = self.slot_kind[slot].1 as u32;
        if let Some(dm) = dec_meta {
            // Layer 3 blocked multi-query shared-KV grouped decode:
            //   1. `k_kvprep_dec` once per (kv_head g, span si)
            //   2. `k_attn_grp_dec` once per (DecAttnGroup, kv_head g), staging shared KV tiles once for up to 8 lanes
            //   3. `k_attn_combine_dec` once per (head h, span si)
            // Shared `CS` (buffer 4) stays bound; rebind `BT` (buffer 8) after `mix_t`'s inline buffer(8) constants.
            assert!(self.sep_comb, "a decode step combines the splits in a separate pass");
            self.bind_pool_slabs(e, l.swa);
            e.buf(self.blocked_slab.borrow().bt_buf.as_ref().expect("blocked slab missing bt_buf"), 0, 8);
            let ka = KvPrepDecArgs {
                desc_off: dm.desc_off,
                pass_idx,
                cs_slot,
                swa: (l.swa as u32) | ((dry_state as u32) << 1),
                q_off: b.q,
                k_off: b.k,
                v_off: b.v,
                r_off: b.rel,
                qp_off: b.qp,
                rp_off: b.rp,
                kv_cur_off: dm.kv_cur_off,
                kdyn_off: if l.swa { -1 } else { b.kdyn as i32 },
                vdyn_off: if l.swa { -1 } else { b.vdyn as i32 },
                kconv_w: l.k_conv,
                vconv_w: l.v_conv,
                qclip_w: l.q_clip,
                relclip_w: l.rel_clip,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            self.go(e, C_KVPREP, "k_kvprep_dec", &ka, (NKV, spans.len(), 1), 128);
            let (solo_off, n_solo, shared_off, n_shared, kps) = if l.swa {
                (dm.swa_solo_off, dm.n_swa_solo, dm.swa_shared_off, dm.n_swa_shared, dm.swa_kps)
            } else {
                (dm.full_solo_off, dm.n_full_solo, dm.full_shared_off, dm.n_full_shared, dm.full_kps)
            };
            if n_solo > 0 {
                let ga_solo = AttnGrpDecArgs {
                    desc_off: dm.desc_off,
                    grp_off: solo_off,
                    qp_off: b.qp,
                    rp_off: b.rp,
                    kv_cur_off: dm.kv_cur_off,
                    sink_off: b.sink,
                    relbias_w: l.rel_bias,
                    pass_idx,
                    kps,
                    swa: l.swa as u32,
                    pad0: 0,
                    pad1: 0,
                };
                self.go_nb(e, C_ATTN, "k_attn_dec_solo", &ga_solo, (n_solo, NKV, 1), 128);
            }
            if n_shared > 0 {
                let ga_shared = AttnGrpDecArgs {
                    desc_off: dm.desc_off,
                    grp_off: shared_off,
                    qp_off: b.qp,
                    rp_off: b.rp,
                    kv_cur_off: dm.kv_cur_off,
                    sink_off: b.sink,
                    relbias_w: l.rel_bias,
                    pass_idx,
                    kps,
                    swa: l.swa as u32,
                    pad0: 0,
                    pad1: 0,
                };
                self.go_nb(e, C_ATTN, "k_attn_grp_dec", &ga_shared, (n_shared, NKV, 1), 512);
            }
            e.barrier();
            let direct_half = decode_combine_direct_half(self.hgemm, self.skip.get());
            let ca = CombineDecArgs {
                desc_off: dm.desc_off,
                gate_off: b.gate,
                out_off: if direct_half { b.agh } else { b.qp },
                swa: l.swa as u32,
                hout: direct_half as u32,
                pad: [0; 3],
            };
            self.go(e, C_COMB, "k_attn_combine_dec", &ca, (NH, spans.len(), 1), 128);
            e.buf(&self.act.borrow(), 0, 3);
            if !direct_half {
                // f32 attention outputs -> the half rows the output projection reads
                self.tohalf(e, C_WO, b.qp, NH * HD, b.agh, NH * HD, NH * HD, t, None);
                e.barrier();
            }
        } else if dec {
            // Layer 2 decode step: each lane's token through the decode path's fused attention.
            let direct_half = decode_combine_direct_half(self.hgemm, self.skip.get());
            let mut combs: Vec<CombineArgs> = Vec::with_capacity(spans.len());
            for (si, sp) in spans.iter().enumerate() {
                let run = self.on(sp.lane);
                let (pos, r0) = (sp.p0, row0[si]);
                let tau = if l.swa { 1.0 } else { 1.0 + self.log_alpha * (((pos + 1) as f32 / self.log_floor).max(1.0)).ln() };
                let ar = self.ar;
                self.bind_lane(e, sp.lane);
                run.bind_kv(e, l.swa);
                let (kb, vb, kcap) = run.kv_slot(slot);
                let base = AttnArgs {
                    pos: pos as u32, cap: kcap, k_base: kb, v_base: vb, cs_off: cs_slot,
                    q_off: b.q + (r0 * NH * HD) as u32, k_off: b.k + (r0 * NKV * HD) as u32, v_off: b.v + (r0 * NKV * HD) as u32,
                    r_off: b.rel + (r0 * NH * NREL) as u32, sink_off: b.sink + (r0 * NH) as u32,
                    kdyn_off: if l.swa { -1 } else { (b.kdyn + (r0 * 32) as u32) as i32 }, vdyn_off: if l.swa { -1 } else { (b.vdyn + (r0 * 32) as u32) as i32 },
                    part_off: self.dstart.get() + si as u32 * ar.total + ar.part, kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias, tau,
                    ctr: 0, gate_off: b.gate + (r0 * NH * HD) as u32, out_off: b.qp + (r0 * NH * HD) as u32, gstr: kcap * HD as u32, istr: HD as u32,
                    ..Default::default()
                };
                let (first_pos, kps, _sp_lo, nsplit) = decode_split_grid(l.swa, pos, self.max_ctx);
                let nkeys = pos + 1 - first_pos;
                let aa = AttnArgs { nkeys: nkeys as u32, first_pos: first_pos as u32, ring: l.swa as u32, kps: kps as u32, nsplit: nsplit as u32,
                                    pad: (self.sep_comb as u32) | ((dry_state as u32) << 1), ..base };
                assert!(self.sep_comb, "a decode step combines the splits in a separate pass");
                self.go_nb(e, C_ATTN, "k_attn", &aa, (nsplit, NKV, 1), 128);
                let comb_out = if direct_half { b.agh + (r0 * (NH * HD / 2)) as u32 } else { aa.out_off };
                combs.push(CombineArgs { part_off: aa.part_off, nsplit: nsplit as u32, gate_off: aa.gate_off, out_off: comb_out, hout: direct_half as u32, pad: [0; 3] });
            }
            e.barrier();
            for ca in &combs { self.go_nb(e, C_COMB, "k_attn_combine", ca, (NH, 1, 1), 128); }
            e.barrier();
            e.buf(&self.act.borrow(), 0, 3);
            if !direct_half {
                // f32 attention outputs -> the half rows the output projection reads
                self.tohalf(e, C_WO, b.qp, NH * HD, b.agh, NH * HD, NH * HD, t, None);
                e.barrier();
            }
        }
        if !dec && blocked { self.bind_pool_slabs(e, l.swa); }
        for (si, sp) in spans.iter().enumerate() {
        if dec { break; }
        let run = self.on(sp.lane);
        let (p0, t, r0) = (sp.p0, sp.t, row0[si]);
        let tt = t as u32;
        // a row-major region shifted to this span's first row
        let rs = |off: u32, dim: usize| off + (r0 * dim) as u32;
        let (n, nn_) = (t - ts, (t - ts) as u32);
        let shs = |off: u32, dim: usize| off + ((r0 + ts) * dim) as u32;
        sp.lane.with_cs_buf(|buf, off, _| e.buf(buf, off, 4));
        let (qd, gd) = (if hatt && qgh { NH * HD / 2 } else { NH * HD }, if qgh { NH * HD / 2 } else { NH * HD });
        let qpd = if hatt { NH * HD / 2 } else { NH * HD };
        if blocked {
            Self::bind_lane_bt(e, sp.lane, l.swa);
            let ka = KvPrepArgs {
                t: tt, p0: p0 as u32, ring: 0, cap: 0, k_base: pass_idx, v_base: 0, cs_off: cs_slot, q_off: rs(b.q, qd),
                k_off: rs(b.k, NKV * HD), v_off: rs(b.v, NKV * HD), r_off: rs(b.rel, NH * NREL),
                kdyn_off: if l.swa { -1 } else { rs(b.kdyn, 32) as i32 }, vdyn_off: if l.swa { -1 } else { rs(b.vdyn, 32) as i32 },
                qp_off: rs(b.qp, qpd), rp_off: rs(b.rp, NH * NREL), qhalf: hatt as u32 | ((hatt && qgh) as u32) << 1, kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip,
                log_alpha: self.log_alpha, log_floor: self.log_floor, is_full: (!l.swa) as u32, kpos0: 0,
            };
            self.go(e, C_KVPREP, "k_kvprep_b_blk", &ka, (t, NKV, 1), 128);
            if let (false, Some((qc, ncap))) = (l.swa, self.qcap.borrow().as_ref()) {
                assert!(spans.len() == 1 && hatt && ts == 0 && p0 + t <= *ncap, "query capture needs one span, the fp16 attention path, no tail trimming and room");
                let (fi, per) = (self.slot_kind[slot].1, ncap * (NH * HD / 2 + NH));
                e.buf(qc, 0, 3);
                let a = DevCopyArgs { src: b.qp, dst: (fi * per + p0 * (NH * HD / 2)) as u32, n: (t * NH * HD / 2) as u32, pad: 0 };
                self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH * HD / 2).div_ceil(1024), 1, 1), 1024);
                let a = DevCopyArgs { src: b.sink, dst: (fi * per + ncap * (NH * HD / 2) + p0 * NH) as u32, n: (t * NH) as u32, pad: 0 };
                self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH).div_ceil(1024), 1, 1), 1024);
                self.bind_pool_slabs(e, l.swa);
            }
            let su = StateArgs { x_off: rs(b.k, NKV * HD), ldx: (NKV * HD) as u32, cs_off: cs_slot, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
            self.keep_state(e, si, &su, C_KVPREP, (NKV * HD / 256, 1, 1));
            self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
            let su = StateArgs { x_off: rs(b.v, NKV * HD), ldx: (NKV * HD) as u32, cs_off: cs_slot + (NKV * HD * 3) as u32, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
            self.keep_state(e, si, &su, C_KVPREP, (NKV * HD / 256, 1, 1));
            self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
            let aa = AttnBArgs { t: tt, p0: p0 as u32, ring: 0, cap: 0, k_base: pass_idx, v_base: 0,
                                 window: if l.swa { EXT as u32 } else { 0 }, qp_off: rs(b.qp, qpd), rp_off: rs(b.rp, NH * NREL), sink_off: rs(b.sink, NH), gate_off: rs(b.gate, gd),
                                 out_off: rs(b.attn_g, NH * HD), relbias_w: l.rel_bias, ghalf: 0, kpos0: 0, pad: 0 };
            if hatt {
                let ah = AttnBArgs { t: nn_, p0: (p0 + ts) as u32, qp_off: shs(b.qp, NH * HD / 2), rp_off: shs(b.rp, NH * NREL), sink_off: shs(b.sink, NH),
                                     gate_off: shs(b.gate, gd), out_off: shs(b.agh, NH * HD / 2), ghalf: qgh as u32, ..aa };
                self.go(e, C_ATTN, "k_attn_h_blk", &ah, (n.div_ceil(32), NH, 1), 256);
            }
            else if self.tensor { self.go(e, C_ATTN, "k_attn_bt_blk", &aa, (t.div_ceil(32), NH, 1), 256); }
            else { self.go(e, C_ATTN, "k_attn_b_blk", &aa, (t.div_ceil(8), NKV, 1), 128); }
            continue;
        }
        let (kb, vb, kcap) = run.kv_slot(slot);
        let (pk, pv, cap, kpos0) = if l.swa { (b.sk * (4 / self.kvb) as u32, b.sv * (4 / self.kvb) as u32, scap, p0 as i32 - (EXT + KGUARD) as i32) } else { (kb, vb, kcap, 0) };
        let kcopy = |n: usize, pos0: usize, to_ring: u32| {
            let a = KvCopyArgs { n: n as u32, pos0: pos0 as u32, ring_k: kb, ring_v: vb, cap: kcap, sk_off: b.sk, sv_off: b.sv, scap, kpos0, to_ring, pad: [0; 2] };
            e.buf(&sp.lane.kv_swa.borrow(), 0, 3);
            self.go(e, C_KVPREP, "k_kvcopy", &a, (n, NKV, 1), 32);
            e.buf(&self.act.borrow(), 0, 3);
        };
        if l.swa { if p0 > 0 { kcopy(p0.min(EXT), p0 - p0.min(EXT), 0); } e.buf(&self.act.borrow(), 0, 3); } else { e.buf(&sp.lane.kv_full.borrow(), 0, 3); }
        let ka = KvPrepArgs {
            t: tt, p0: p0 as u32, ring: 0, cap, k_base: pk, v_base: pv, cs_off: cs_slot, q_off: rs(b.q, qd),
            k_off: rs(b.k, NKV * HD), v_off: rs(b.v, NKV * HD), r_off: rs(b.rel, NH * NREL),
            kdyn_off: if l.swa { -1 } else { rs(b.kdyn, 32) as i32 }, vdyn_off: if l.swa { -1 } else { rs(b.vdyn, 32) as i32 },
            qp_off: rs(b.qp, qpd), rp_off: rs(b.rp, NH * NREL), qhalf: hatt as u32 | ((hatt && qgh) as u32) << 1, kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip,
            log_alpha: self.log_alpha, log_floor: self.log_floor, is_full: (!l.swa) as u32, kpos0,
        };
        self.go(e, C_KVPREP, "k_kvprep_b", &ka, (t, NKV, 1), 128);
        if let (false, Some((qc, ncap))) = (l.swa, self.qcap.borrow().as_ref()) {
            assert!(spans.len() == 1 && hatt && ts == 0 && p0 + t <= *ncap, "query capture needs one span, the fp16 attention path, no tail trimming and room");
            let (fi, per) = (self.slot_kind[slot].1, ncap * (NH * HD / 2 + NH));
            e.buf(qc, 0, 3);
            let a = DevCopyArgs { src: b.qp, dst: (fi * per + p0 * (NH * HD / 2)) as u32, n: (t * NH * HD / 2) as u32, pad: 0 };
            self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH * HD / 2).div_ceil(1024), 1, 1), 1024);
            let a = DevCopyArgs { src: b.sink, dst: (fi * per + ncap * (NH * HD / 2) + p0 * NH) as u32, n: (t * NH) as u32, pad: 0 };
            self.go(e, C_KVPREP, "k_devcopy", &a, ((t * NH).div_ceil(1024), 1, 1), 1024);
            e.buf(&sp.lane.kv_full.borrow(), 0, 3);
        }
        if l.swa { kcopy(t.min(EXT), p0 + t - t.min(EXT), 1); }
        // k_attn_h reads keys in whole tiles of ATT_AK positions, so the span's last tile runs past its last key to the
        // next multiple of ATT_AK. Those scratch rows are masked with a zero weight, which does not cancel a NaN or an
        // infinity, and k_zero cleared them only at the start of the step: an earlier span of this layer (or a longer span of the
        // layer before) may have written another lane's keys there since. Zero them for this span.
        // (specs/ScratchTail.tla; one span alone never leaves them dirty)
        if l.swa && spans.len() > 1 {
            let (end, tail) = (p0 + t, (p0 + t).next_multiple_of(ATT_AK) - (p0 + t));
            if tail > 0 { kcopy(tail, end, 2); }
        }
        let su = StateArgs { x_off: rs(b.k, NKV * HD), ldx: (NKV * HD) as u32, cs_off: cs_slot, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
        self.keep_state(e, si, &su, C_KVPREP, (NKV * HD / 256, 1, 1));
        self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
        let su = StateArgs { x_off: rs(b.v, NKV * HD), ldx: (NKV * HD) as u32, cs_off: cs_slot + (NKV * HD * 3) as u32, nch: (NKV * HD) as u32, t: tt, pad: [0; 3] };
        self.keep_state(e, si, &su, C_KVPREP, (NKV * HD / 256, 1, 1));
        self.go(e, C_KVPREP, "k_state_upd", &su, (NKV * HD / 256, 1, 1), 256);
        // ---- attention (causal, query-tiled)
        let aa = AttnBArgs { t: tt, p0: p0 as u32, ring: 0, cap, k_base: pk, v_base: pv,
                             window: if l.swa { EXT as u32 } else { 0 }, qp_off: rs(b.qp, qpd), rp_off: rs(b.rp, NH * NREL), sink_off: rs(b.sink, NH), gate_off: rs(b.gate, gd),
                             out_off: rs(b.attn_g, NH * HD), relbias_w: l.rel_bias, ghalf: 0, kpos0, pad: 0 };
        if hatt {
            let ah = AttnBArgs { t: nn_, p0: (p0 + ts) as u32, qp_off: shs(b.qp, NH * HD / 2), rp_off: shs(b.rp, NH * NREL), sink_off: shs(b.sink, NH),
                                 gate_off: shs(b.gate, gd), out_off: shs(b.agh, NH * HD / 2), ghalf: qgh as u32, ..aa };
            self.go(e, C_ATTN, "k_attn_h", &ah, (n.div_ceil(32), NH, 1), 256);
        }
        else if self.tensor { self.go(e, C_ATTN, "k_attn_bt", &aa, (t.div_ceil(32), NH, 1), 256); }
        else { self.go(e, C_ATTN, "k_attn_b", &aa, (t.div_ceil(8), NKV, 1), 128); }
        }
        // ---- output projection + post-conv (+ residual)
        if self.hgemm {
            if !hatt { self.tohalf(e, C_WO, b.attn_g, NH * HD, b.agh, NH * HD, NH * HD, t, None); e.barrier(); }
            self.gemm_h(e, C_WO, &[Engine::hseg(&l.wo, sh(b.agh, NH * HD / 2), sh(b.wo_y, E))], n);
        } else {
            self.gemm(e, C_WO, &[Engine::gseg(&l.wo, b.attn_g, b.wo_y)], t);
        }
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        // (tail mode: the first 3 tail rows see the conv state instead of their real predecessors; they are never used)
        // post-conv and its state: fused single 2D dispatch for Layer 3 decode, span by span otherwise.
        if let Some(dm) = dec_meta {
            debug_assert!(ts == 0 && !self.verify_record.get());
            let pc = PostConvDecArgs {
                desc_off: dm.desc_off,
                y_off: b.wo_y,
                res_off: vo(cur),
                out_off: vo(mid),
                cs_slot_off: cs_slot + (2 * NKV * HD * 3) as u32,
                convw_off: l.attn_post_conv,
                has_res: (!block_start) as u32,
                no_state_write: dry_state as u32,
            };
            self.go(e, C_WOPOST, "k_postconv_state_dec", &pc, (E / 256, spans.len(), 1), 256);
        } else {
            for (si, sp) in spans.iter().enumerate() {
                let (q0, qn) = (((row0[si] + ts) * E) as u32, sp.t - ts);
                sp.lane.with_cs_buf(|buf, off, _| e.buf(buf, off, 4));
                let cv = ConvBArgs { y_off: b.wo_y + q0, ldy: es, res_off: vo(cur) + q0, out_off: vo(mid) + q0, cs_off: cs_slot + (2 * NKV * HD * 3) as u32,
                                     convw_off: l.attn_post_conv, has_res: (!block_start) as u32, pad: 0 };
                self.go_nb(e, C_WOPOST, "k_postconv_b", &cv, (E / 256, qn, 1), 256);
            }
            e.barrier();
            if !dry_state {
                for (si, sp) in spans.iter().enumerate() {
                    sp.lane.with_cs_buf(|buf, off, _| e.buf(buf, off, 4));
                    let su = StateArgs { x_off: b.wo_y + (row0[si] * E) as u32, ldx: es, cs_off: cs_slot + (2 * NKV * HD * 3) as u32, nch: es, t: sp.t as u32, pad: [0; 3] };
                    self.keep_state(e, si, &su, C_WOPOST, (E / 256, 1, 1));
                    self.go_nb(e, C_WOPOST, "k_state_upd", &su, (E / 256, 1, 1), 256);
                }
                e.barrier();
            }
        }
        if block_start { blocks.push(cur); }
        // ---- FFN input
        let mut st: Vec<u32> = blocks.iter().map(|&v| sh(vo(v), E)).collect(); st.push(sh(vo(mid), E));
        let x = if let Some(lw) = ctl {
            let mut jf = self.mix_job_off(&st, l.ffn_res_query as i32, 1, &[b.fin], &[], es);
            if self.hgemm { jf.hcopy[0][0] = b.cath[3] as i32; } else { jf.ncopy[0][0] = b.cat_u[3] as i32; }
            self.mix_t(e, C_MIXF, &[jf], t);
            let ho: Vec<(u32, bool)> = if self.hgemm { vec![(b.fh, rth)] } else { vec![] };
            self.ctrl_b(e, lw, &[(3, b.fin, b.msg[3], b.finc, l.ffn_norm)], t, &ho);
            b.finc
        } else {
            let mut jf = self.mix_job_off(&st, l.ffn_res_query as i32, 1, &[sh(b.fin, E)], &[l.ffn_norm as i32], es);
            if self.hgemm { jf.hout[0] = sh(b.fh, E / 2) as i32; jf.nof32 = rth as u32; }
            self.mix_t(e, C_MIXF, &[jf], n);
            b.fin
        };
        // ---- router + top-k + expert grouping
        let rs = GSeg { w_lo: l.router, w_hi: 0, row_bytes: es * 2, wtype: T_BF16, nrows: NEXP as u32, k: es, in_off: sh(x, E), out_off: b.rlog,
                        ldo: NEXP as u32, wsrc: 1, ..Default::default() };
        if rth { self.gemm_h(e, C_ROUTER, &[Engine::hseg_pb(l.router, NEXP as u32, es, sh(b.fh, E / 2), b.rlog)], n); }
        else { self.gemm(e, C_ROUTER, &[rs], n); }
        let tile = 32usize;
        if use_fused_topk_route3(self.hgemm, n) {
            let tra = TopkRoute3Args {
                logits_off: b.rlog,
                bias_off: l.exp_bias,
                sel_off: b.sel,
                t: nn_,
                wscale: self.wscale,
                perm_off: b.perm,
                inv_off: b.inv,
                aw_off: b.aw,
                ind_off: b.ind,
                gu_y: (FF / 32) as u32,
                dn_y: (E / 64) as u32,
                tiles_off: b.tl,
                pad: [0; 3],
            };
            self.go(e, C_TOPK, "k_moe_topk_route3", &tra, (1, 1, 1), 1024);
        } else {
            let ta = TopkBArgs { logits_off: b.rlog, bias_off: l.exp_bias, sel_off: b.sel, t: nn_, wscale: self.wscale, pad: [0; 3] };
            self.go(e, C_TOPK, "k_topk_b", &ta, (n, 1, 1), 32);
            if self.hgemm {
                let ra = Route3Args { sel_off: b.sel, t: nn_, perm_off: b.perm, inv_off: b.inv, aw_off: b.aw, ind_off: b.ind, gu_y: (FF / 32) as u32,
                                      dn_y: (E / 64) as u32, tiles_off: b.tl, pad: [0; 2] };
                self.go(e, C_TOPK, "k_moe_route3", &ra, (1, 1, 1), 1024);
            } else {
                let ra = RouteArgs { sel_off: b.sel, t: tt, perm_off: b.perm, inv_off: b.inv, tiles_off: b.tiles, ntiles_off: b.ntiles, tile: tile as u32, pad: 0 };
                self.go(e, C_ROUTER, "k_moe_route", &ra, (1, 1, 1), 1024);
            }
        }
        let max_tiles = (t * TOPK).div_ceil(tile) + NEXP;
        let gu = MoeGemmArgs {
            w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
            wtype: l.gu.ty, nrows: (2 * FF) as u32, k: es, x_off: x, gather: 1, perm_off: b.perm, tiles_off: b.tiles, ntiles_off: b.ntiles,
            out_off: b.act, ldo: FF as u32, situ: 1, pad: 0, beta: self.situ_beta, lbeta: self.situ_lbeta,
            gs_lo: l.gu.gs as u32, gs_hi: (l.gu.gs >> 32) as u32,
        };
        let mg = if self.tensor { "k_moe_gemm_t" } else { "k_moe_gemm" };
        if self.hgemm {
            // gather the FFN input rows of every (token, expert) assignment into contiguous half rows
            self.tohalf_x(e, C_GATHER, sh(b.fh, E / 2), E, b.xgh, E, E, n * TOPK, Some(b.perm), true);
            e.barrier();
            // tiles in two width classes, sized on the GPU by the routing kernel (indirect dispatch); classes run concurrently.
            // Each token picks TOPK distinct experts, so when n <= 64 every expert receives <= 64 tokens and class 0 (65..128) is empty.
            let cls_start = moe_class_start(n);
            if self.skip.get() & (1 << C_GU) == 0 {
                for (c, (name, _, tg)) in MOE_CLS.iter().enumerate().skip(cls_start) {
                    let g = MoeGemmHArgs {
                        w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                        wtype: l.gu.ty, nrows: (2 * FF) as u32, k: es, xh_off: b.xgh, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                        out_off: b.acth, ldo: FF as u32, situ: 1, aw_off: b.aw, beta: self.situ_beta, lbeta: self.situ_lbeta,
                        gs_lo: l.gu.gs as u32, gs_hi: (l.gu.gs >> 32) as u32, pad0: 0, pad1: 0,
                    };
                    e.pso(&self.gpu.pso(name)); e.bytes(&g, 5);
                    e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + c * 4) * 4, *tg);
                }
                e.barrier();
            }
            if self.skip.get() & (1 << C_DOWN) == 0 {
                for (c, (name, _, tg)) in MOE_CLS.iter().enumerate().skip(cls_start) {
                    let d = MoeGemmHArgs {
                        w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                        wtype: l.dn.ty, nrows: es, k: FF as u32, xh_off: b.acth, tiles_off: b.tl[c], ntiles_off: b.ind + (c * 4) as u32,
                        out_off: b.yd, ldo: es, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                        gs_lo: l.dn.gs as u32, gs_hi: (l.dn.gs >> 32) as u32, pad0: 0, pad1: 0,
                    };
                    e.pso(&self.gpu.pso(name)); e.bytes(&d, 5);
                    e.dispatch_indirect(&self.act.borrow(), (b.ind as usize + 4 * NTC + c * 4) * 4, *tg);
                }
                e.barrier();
            }
        } else {
        self.go(e, C_GU, mg, &gu, (FF / 32, max_tiles, 1), 128);
        let dn = MoeGemmArgs {
            w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
            wtype: l.dn.ty, nrows: es, k: FF as u32, x_off: b.act, gather: 0, perm_off: b.perm, tiles_off: b.tiles, ntiles_off: b.ntiles,
            out_off: b.yd, ldo: es, situ: 0, pad: 0, beta: 0.0, lbeta: 0.0,
            gs_lo: l.dn.gs as u32, gs_hi: (l.dn.gs >> 32) as u32,
        };
        self.go(e, C_DOWN, mg, &dn, (E / 64, max_tiles, 1), 128);
        }
        let ca = MoeCombArgs { yd_off: b.yd, inv_off: b.inv, sel_off: b.sel, out_off: sh(b.ffn_y, E) };
        if self.hgemm { self.go(e, C_COMB2, "k_moe_comb_h", &ca, (n, E.div_ceil(1024), 1), 256); }
        else { self.go(e, C_DOWN, "k_moe_comb_b", &ca, (E / 256, t, 1), 256); }
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        if let Some(dm) = dec_meta {
            debug_assert!(ts == 0 && !self.verify_record.get());
            let pc = PostConvDecArgs {
                desc_off: dm.desc_off,
                y_off: b.ffn_y,
                res_off: vo(mid),
                out_off: vo(out),
                cs_slot_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
                convw_off: l.ffn_post_conv,
                has_res: 1,
                no_state_write: dry_state as u32,
            };
            self.go(e, C_FFNPOST, "k_postconv_state_dec", &pc, (E / 256, spans.len(), 1), 256);
        } else {
            for (si, sp) in spans.iter().enumerate() {
                let (q0, qn) = (((row0[si] + ts) * E) as u32, sp.t - ts);
                sp.lane.with_cs_buf(|buf, off, _| e.buf(buf, off, 4));
                let cv = ConvBArgs { y_off: b.ffn_y + q0, ldy: es, res_off: vo(mid) + q0, out_off: vo(out) + q0, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
                                     convw_off: l.ffn_post_conv, has_res: 1, pad: 0 };
                self.go_nb(e, C_FFNPOST, "k_postconv_b", &cv, (E / 256, qn, 1), 256);
            }
            e.barrier();
            if !dry_state {
                for (si, sp) in spans.iter().enumerate() {
                    sp.lane.with_cs_buf(|buf, off, _| e.buf(buf, off, 4));
                    let su = StateArgs { x_off: b.ffn_y + (row0[si] * E) as u32, ldx: es, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32, nch: es, t: sp.t as u32, pad: [0; 3] };
                    self.keep_state(e, si, &su, C_FFNPOST, (E / 256, 1, 1));
                    self.go_nb(e, C_FFNPOST, "k_state_upd", &su, (E / 256, 1, 1), 256);
                }
                e.barrier();
            }
        }
        out
    }

}

/// The engine working on one lane: every method that reads or writes a stream's state (its K/V cache, convolution
/// rows, tokens, disk tier) lives here; the weights, kernels and scratch arenas are the engine's, reached through Deref.
impl<'a> Run<'a> {
    /// bind everything a kernel may read: the engine's buffers and this lane's cache, convolution rows and tokens
    fn bind(&self, e: &Enc) {
        self.bind_eng(e);
        self.bind_lane(e, self.lane);
    }
    /// (k_base, v_base, cap) of a slot's K/V inside its layer-2 buffer (offsets in cache
    /// elements: fp16, or bytes with --kv8). This is the contiguous cache, not the block
    /// table. A block-cache key address is `Lane::attn_key_offset`, a `u64` from the table.
    fn kv_slot(&self, slot: usize) -> (u32, u32, u32) {
        let (swa, i) = self.slot_kind[slot];
        let cap = if swa { self.lane.swa_cap.get() } else { self.lane.full_cap.get() };
        ((i * 2 * NKV * cap * HD) as u32, ((i * 2 + 1) * NKV * cap * HD) as u32, cap as u32)
    }
    /// bind the K/V buffer a layer's kernels address through buffer 3
    fn bind_kv(&self, e: &Enc, swa: bool) { if swa { e.buf(&self.lane.kv_swa.borrow(), 0, 3); } else { e.buf(&self.lane.kv_full.borrow(), 0, 3); } }
    /// positions the K/V cache can hold right now
    pub fn kv_capacity(&self) -> usize { if self.lane.swa_cap.get() >= EXT { self.lane.full_cap.get() } else { self.lane.swa_cap.get().min(self.lane.full_cap.get()) } }
    /// first position a generation loop must call `gen_reserve` for (with an idle GPU) before it may run that step
    pub fn gen_capacity(&self) -> usize {
        match &self.lane.tier { Some(t) if t.on.get() => t.next.get(), Some(t) => self.kv_capacity().min(t.min_n), None => self.kv_capacity() }
    }
    /// bytes of K/V cache held in memory (with the disk tier: the hot rings and the read staging)
    pub fn kv_bytes(&self) -> usize {
        self.lane.kv_swa.borrow().length() + self.lane.kv_full.borrow().length() + self.lane.tier.as_ref().map_or(0, |t| if t.on.get() { t.stg.borrow().length() } else { 0 })
    }
    /// Make room for `n` positions in the K/V cache (no-op if they fit). Nothing may be in flight on the GPU.
    /// Sliding-window layers keep one window (their ring is linear until the context reaches it, so it can grow like the
    /// full-attention rows); `headroom` adds slack for the tokens about to be generated so growth stays rare.
    pub fn reserve(&self, n: usize, headroom: bool) {
        assert!(n <= self.max_ctx, "context full ({} tokens)", self.max_ctx);
        if self.lane.use_blocks.get() { return; }
        let grow = |buf: &std::cell::RefCell<Buf>, capc: &std::cell::Cell<usize>, need: usize, limit: usize, nslots: usize| {
            let old = capc.get();
            if need <= old { return; }
            // slack for the tokens about to be generated: INK_HEADROOM=a,b -> need/a (at least b positions)
            let (hd, hm) = std::env::var("INK_HEADROOM").ok().and_then(|v| v.split_once(',').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))).unwrap_or((16, 0));
            let want = if headroom { need + (need / hd).max(hm) } else { need };
            let cap = (want.div_ceil(64) * 64).min(limit);
            let eb = self.kvb;
            let nb = self.gpu.buffer(nslots * 2 * NKV * cap * HD * eb);
            // every (slot, K|V, group) row block moves to its place in the larger layout
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for blk in 0..nslots * 2 * NKV { std::ptr::copy_nonoverlapping(src.add(blk * old * HD * eb), dst.add(blk * cap * HD * eb), old * HD * eb); }
            }
            *buf.borrow_mut() = nb;
            capc.set(cap);
        };
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        grow(&self.lane.kv_swa, &self.lane.swa_cap, n.min(EXT), EXT, n_swa);
        grow(&self.lane.kv_full, &self.lane.full_cap, n, self.max_ctx, N_SLOT - n_swa);
    }
    /// Allocate everything a prefill of `t` tokens ending at context length `n` needs and let the GPU map it now.
    /// The first use of a new buffer costs ~20 ms per GiB; a caller that knows its prompt before it starts the clock
    /// (one-shot runs: while the model loads) pays that here instead of inside the prefill.
    pub fn prepare(&self, n: usize, t: usize) {
        self.reserve(n + 1, true);
        self.ensure_batch(t.min(self.chunk));
        let cb = self.gpu.cmd();
        let e = Enc::new(&cb);
        self.bind(&e);
        // one trivial dispatch per buffer (a window-history copy of key 0 into the scratch guard area) makes the driver wire them
        let b = self.ba.get();
        let a = KvCopyArgs { n: 1, pos0: 0, ring_k: 0, ring_v: 0, cap: 64, sk_off: b.sk, sv_off: b.sv, scap: (EXT + b.cap + 2 * KGUARD) as u32,
                             kpos0: 0, to_ring: 0, pad: [0; 2] };
        e.pso(&self.gpu.pso("k_kvcopy")); e.bytes(&a, 5);
        self.bind_kv(&e, true); e.dispatch((1, 1, 1), 32);
        e.barrier();
        self.bind_kv(&e, false); e.dispatch((1, 1, 1), 32);
        e.end(); cb.commit(); cb.waitUntilCompleted();
    }
    /// bytes of GPU working memory besides the weights: (K/V cache, activations incl. the batch arena if allocated)
    pub fn mem(&self) -> (usize, usize) { (self.kv_bytes(), self.act.borrow().length()) }

    // ======================================================================== disk tier of the K/V cache (--kv-disk)
    pub fn tier_on(&self) -> bool { self.lane.tier.as_ref().is_some_and(|t| t.on.get()) }
    fn kind_bufs(&self, kind: usize) -> (&RefCell<Buf>, &Cell<usize>, usize) {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        if kind == 0 { (&self.lane.kv_swa, &self.lane.swa_cap, n_swa) } else { (&self.lane.kv_full, &self.lane.full_cap, N_SLOT - n_swa) }
    }
    /// what the cache would hold in memory for `n` positions without the tier, bytes
    pub fn kv_exact_bytes(&self, n: usize) -> usize {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        (n_swa * n.min(EXT) + (N_SLOT - n_swa) * n) * REC
    }
    /// bytes of the cache that are on disk only (read back for every token)
    pub fn kv_disk_bytes(&self, n: usize) -> usize {
        let Some(t) = &self.lane.tier else { return 0 };
        if !t.on.get() { return 0; }
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        let cold = |kind: usize, w0: usize| t.lo[kind].get().saturating_sub(w0);
        (n_swa * cold(0, n.saturating_sub(EXT)) + (N_SLOT - n_swa) * cold(1, 0)) * REC
    }
    /// (bytes read from the K/V file, time spent reading) since the engine was loaded
    pub fn tier_io(&self) -> (u64, std::time::Duration) { self.lane.tier.as_ref().map_or((0, Default::default()), |t| (t.read_bytes.get(), t.read_time.get())) }
    /// hot ring sizes (positions) and staging bytes for a context of `n` positions: together at most half of what the
    /// cache would take in memory
    fn tier_targets(&self, n: usize) -> (usize, usize, usize) {
        // dev: INK_TIER_HOT=positions forces small hot rings so short tests exercise the file
        if let Some(h) = std::env::var("INK_TIER_HOT").ok().and_then(|v| v.parse::<usize>().ok()) { return (h, h, 2 << 20); }
        let c = self.kv_exact_bytes(n);
        // staging: two halves, each at least one split of the longest attention (see `step`)
        let s = Self::tier_staging(c, self.max_ctx);
        let hot = (c / 2).saturating_sub(s);
        ((n.min(EXT) * hot / c / 64 * 64).max(64), (n * hot / c / 64 * 64).max(64), s)
    }
    fn tier_staging(c: usize, max_ctx: usize) -> usize {
        let max_kps = decode_split_kps(false, max_ctx).max(decode_split_kps(true, max_ctx));
        (((c / 16) >> 20).clamp(2, 32) << 20).max(2 * max_kps * REC)
    }
    fn tier_foff(&self, kind: usize, i: usize, p: usize) -> u64 {
        let n_swa = self.slot_kind.iter().filter(|k| k.0).count();
        ((if kind == 0 { i * EXT + p % EXT } else { n_swa * EXT + i * self.max_ctx + p }) * REC) as u64
    }
    fn split_target() -> usize { std::env::var("INK_SPLITS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize) }
    /// file pieces (offset, positions) that hold positions [a, b) of a slot, in position order (a sliding-window slot's
    /// file is a ring of one window)
    fn tier_pieces(&self, kind: usize, i: usize, a: usize, b: usize) -> Vec<(u64, usize)> {
        let mut v = Vec::new();
        let mut p = a;
        while p < b { let e = if kind == 0 { b.min((p / EXT + 1) * EXT) } else { b }; v.push((self.tier_foff(kind, i, p), e - p)); p = e; }
        v
    }
    /// copy one position (16 rows of 128 fp16) between two cache buffers / between a cache buffer and a file record
    unsafe fn rows_copy(src: *const u8, i: usize, cap_s: usize, idx_s: usize, dst: *mut u8, cap_d: usize, idx_d: usize) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(src.add(((i * 2 * NKV + r) * cap_s + idx_s) * HD * 2), dst.add(((i * 2 * NKV + r) * cap_d + idx_d) * HD * 2), HD * 2); }
    }
    unsafe fn rows_to_rec(src: *const u8, i: usize, cap: usize, idx: usize, rec: *mut u8) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(src.add(((i * 2 * NKV + r) * cap + idx) * HD * 2), rec.add(r * HD * 2), HD * 2); }
    }
    unsafe fn rec_to_rows(rec: *const u8, dst: *mut u8, i: usize, cap: usize, idx: usize) {
        for r in 0..2 * NKV { std::ptr::copy_nonoverlapping(rec.add(r * HD * 2), dst.add(((i * 2 * NKV + r) * cap + idx) * HD * 2), HD * 2); }
    }
    /// write positions [a, b) of every slot of a kind to the file; `idx(p)` is where position p sits in the kind's buffer now
    fn tier_write(&self, kind: usize, a: usize, b: usize, idx: impl Fn(usize) -> usize) {
        if a >= b { return; }
        let t = self.lane.tier.as_ref().unwrap();
        let (buf, capc, nsl) = self.kind_bufs(kind);
        let (src, cap) = (buf.borrow().contents().as_ptr() as *const u8, capc.get());
        let mut recs = vec![0u8; (b - a).min(2048) * REC];
        for i in 0..nsl {
            let mut p = a;
            while p < b {
                let e = b.min(p + 2048);
                for q in p..e { unsafe { Self::rows_to_rec(src, i, cap, idx(q), recs.as_mut_ptr().add((q - p) * REC)); } }
                let mut o = 0;
                for (off, n) in self.tier_pieces(kind, i, p, e) { t.file.write_all_at(&recs[o..o + n * REC], off).expect("write the K/V file"); o += n * REC; }
                p = e;
            }
        }
    }
    /// read file pieces (offset, bytes, destination) into memory, several at a time (uncached reads; large pieces are split)
    fn tier_read(&self, jobs: &[(u64, usize, *mut u8)]) {
        let t = self.lane.tier.as_ref().unwrap();
        let t0 = std::time::Instant::now();
        let piece: usize = std::env::var("INK_TIER_PIECE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4) << 20;
        let nthreads: usize = std::env::var("INK_TIER_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let mut parts: Vec<(u64, usize, usize)> = Vec::new();        // (file offset, bytes, destination address)
        for &(off, bytes, dst) in jobs {
            let mut o = 0;
            while o < bytes { let n = (bytes - o).min(piece); parts.push((off + o as u64, n, dst as usize + o)); o += n; }
        }
        let total: usize = parts.iter().map(|p| p.1).sum();
        let file = &t.file;
        let read = |&(off, n, dst): &(u64, usize, usize)| {
            let sl = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, n) };
            file.read_exact_at(sl, off).expect("read the K/V file");
        };
        if parts.len() <= 1 { parts.iter().for_each(read); }
        else {
            let next = std::sync::atomic::AtomicUsize::new(0);
            let (parts, next, read) = (&parts, &next, &read);
            std::thread::scope(|sc| {
                for _ in 0..parts.len().min(nthreads) {
                    sc.spawn(move || loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= parts.len() { break; }
                        read(&parts[k]);
                    });
                }
            });
        }
        t.read_bytes.set(t.read_bytes.get() + total as u64);
        t.read_time.set(t.read_time.get() + t0.elapsed());
    }
    fn tier_wait(&self) {
        let t = self.lane.tier.as_ref().unwrap();
        for p in t.pending.borrow_mut().iter_mut() { if let Some(cb) = p.take() { cb.waitUntilCompleted(); } }
    }
    /// Start keeping only the newest positions in memory: `n` positions are cached, in the normal layout. GPU idle.
    fn tier_enter(&self, n: usize) {
        let t = self.lane.tier.as_ref().unwrap();
        if t.on.get() || n < t.min_n { return; }
        let (hs, hf, s) = self.tier_targets(n);
        for (kind, h) in [(0usize, hs), (1usize, hf)] {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let old_cap = capc.get();
            let h = h.min(n);
            let (first, lo) = (if kind == 0 { n.saturating_sub(EXT) } else { 0 }, n - h.min(n));
            let idx_old = |p: usize| if kind == 0 { p % old_cap } else { p };
            // the file runs one split ahead of the hot ring's start, so a split boundary always lies where both have the keys
            let kps = decode_split_kps(kind == 0, self.max_ctx);
            let cold = (lo + kps + 64).min(n);
            self.tier_write(kind, first.max(t.filed[kind].get()), cold, idx_old);
            t.filed[kind].set(cold.max(t.filed[kind].get()));
            let nb = self.gpu.buffer(nsl * 2 * NKV * h * HD * 2);
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for i in 0..nsl { for p in lo.max(first)..n { Self::rows_copy(src, i, old_cap, idx_old(p), dst, h, p % h); } }
            }
            *buf.borrow_mut() = nb;
            capc.set(h);
            t.lo[kind].set(lo); t.cold[kind].set(cold);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(s);
        t.half.set(0); t.loaded.set(usize::MAX);
        t.on.set(true);
        t.next.set(n + 64);
    }
    /// Bring the whole cache back into memory in the normal layout (a prefill attends through it directly). GPU idle.
    pub fn tier_leave(&self, n: usize) {
        let Some(t) = &self.lane.tier else { return };
        if !t.on.get() { return; }
        self.tier_wait();
        for kind in 0..2 {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let (h, lo) = (capc.get(), t.lo[kind].get());
            let first = if kind == 0 { n.saturating_sub(EXT) } else { 0 };
            let cap = if kind == 0 { (n.min(EXT).div_ceil(64) * 64).clamp(KV_CAP0, EXT) } else { (n.div_ceil(64) * 64).clamp(KV_CAP0, self.max_ctx) };
            let idx_new = |p: usize| if kind == 0 { p % cap } else { p };
            let nb = self.gpu.buffer(nsl * 2 * NKV * cap * HD * 2);
            let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
            let mut recs = vec![0u8; 2048 * REC];
            let rp = recs.as_mut_ptr();
            let cold_end = lo.max(first).min(n);
            for i in 0..nsl {
                let mut p = first;
                while p < cold_end {
                    let e = cold_end.min(p + 2048);
                    let mut o = 0;
                    let mut jobs: Vec<(u64, usize, *mut u8)> = Vec::new();
                    for (off, k) in self.tier_pieces(kind, i, p, e) { jobs.push((off, k * REC, unsafe { rp.add(o) })); o += k * REC; }
                    self.tier_read(&jobs);
                    for q in p..e { unsafe { Self::rec_to_rows(rp.add((q - p) * REC), dst, i, cap, idx_new(q)); } }
                    p = e;
                }
                for q in cold_end..n { unsafe { Self::rows_copy(src, i, h, q % h, dst, cap, idx_new(q)); } }
            }
            drop(recs);
            *buf.borrow_mut() = nb;
            capc.set(cap);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(16384);
        t.on.set(false);
    }
    /// forget the cache (a new session)
    fn tier_reset(&self) {
        let Some(t) = &self.lane.tier else { return };
        t.filed[0].set(0); t.filed[1].set(0);
        if !t.on.get() { return; }
        self.tier_wait();
        for kind in 0..2 {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            *buf.borrow_mut() = self.gpu.buffer(nsl * 2 * NKV * KV_CAP0 * HD * 2);
            capc.set(KV_CAP0);
        }
        *t.stg.borrow_mut() = self.gpu.buffer(16384);
        t.on.set(false);
    }
    /// Make room for the step at `pos` (positions 0..pos are cached). Nothing may be in flight on the GPU.
    pub fn gen_reserve(&self, pos: usize) {
        let Some(t) = &self.lane.tier else { self.reserve(pos + 1, true); return };
        if !t.on.get() { self.tier_enter(pos); }
        if !t.on.get() { self.reserve(pos + 1, true); return; }
        // the hot rings follow the context so memory stays at half of the cache
        self.tier_wait();
        let (hs, hf, s) = self.tier_targets(pos + 1);
        for (kind, h) in [(0usize, hs), (1usize, hf)] {
            let (buf, capc, nsl) = self.kind_bufs(kind);
            let cap = capc.get();
            if h < cap + 64 { continue; }
            let nb = self.gpu.buffer(nsl * 2 * NKV * h * HD * 2);
            unsafe {
                let (src, dst) = (buf.borrow().contents().as_ptr() as *const u8, nb.contents().as_ptr() as *mut u8);
                for i in 0..nsl { for p in t.lo[kind].get()..pos { Self::rows_copy(src, i, cap, p % cap, dst, h, p % h); } }
            }
            *buf.borrow_mut() = nb;
            capc.set(h);
        }
        if s > t.stg.borrow().length() { *t.stg.borrow_mut() = self.gpu.buffer(s); t.half.set(0); }
        t.next.set(pos + 64);
    }
    /// Commit one decode step. Tiered decoding can submit several command buffers per step.
    pub fn step(&self, pos: usize, logits: bool, argmax: bool) -> DecodeStep {
        let cb = self.gpu.cmd();
        if !self.tier_on() {
            self.encode_step(&cb, pos, logits, argmax);
            cb.commit();
            return DecodeStep { prior: Vec::new(), last: cb };
        }
        let t = self.lane.tier.as_ref().unwrap();
        // Rows this step overwrites go to the file first - in fact one split earlier than that (and in batches of 16), so the
        // keys around the hot ring's start are in both places. Rows older than 32 steps are final (2 steps are in flight).
        for kind in 0..2 {
            let (_, capc, _) = self.kind_bufs(kind);
            let cap = capc.get();
            let need = (pos + 1).saturating_sub(cap);
            let lo = t.lo[kind].get().max(need);
            let done = pos.saturating_sub(32);
            let kps = decode_split_kps(kind == 0, self.max_ctx);
            let want = (lo + kps).min(done).max(need);
            if t.cold[kind].get() < want {
                let upto = (want + 15).min(done).max(want);
                self.tier_write(kind, t.cold[kind].get().max(t.filed[kind].get()), upto, |p| p % cap);
                t.cold[kind].set(upto);
                t.filed[kind].set(upto.max(t.filed[kind].get()));
            }
            t.lo[kind].set(lo);
        }
        // Plan this step's file reads: per layer execution the splits older than its hot ring, packed into staging halves.
        // The layer's keys are divided into cache-fixed absolute splits exactly as without the tier; the file supplies whole
        // splits up to an absolute split boundary `b` that the hot ring also reaches, so every split sums the same keys in
        // the same order as the in-memory path and the results are bit-identical.
        let half_recs = t.stg.borrow().length() / 2 / REC;
        let (mut plan, mut execs) = (Vec::new(), Vec::new());
        let (mut round, mut room) = (0usize, half_recs);
        let mut order: Vec<usize> = (0..LOOP_START).collect();
        for pass in 0..N_PASS { for ip in LOOP_START..LOOP_END { order.push(slot_of(pass, ip)); } }
        order.extend(LOOP_END..N_PHYS);
        for (exec, &slot) in order.iter().enumerate() {
            let (swa, i) = self.slot_kind[slot];
            let kind = if swa { 0 } else { 1 };
            let (w0, kps, sp_lo, nsplit) = decode_split_grid(swa, pos, self.max_ctx);
            let lo = t.lo[kind].get().max(w0);
            let b = if lo <= w0 { w0 } else { lo.div_ceil(kps) * kps };
            if b <= pos && b <= t.cold[kind].get().max(w0) && kps <= half_recs {
                let sp_end = if b > w0 { b / kps - sp_lo } else { 0 };
                let (mut c, mut sp) = (w0, 0usize);
                while sp < sp_end {
                    let next_boundary = (c / kps + 1) * kps;
                    let first_len = next_boundary - c;
                    if room < first_len { round += 1; room = half_recs; }
                    let mut ns = 1usize;
                    let mut c1 = next_boundary;
                    while sp + ns < sp_end && (c1 - c) + kps <= room {
                        c1 += kps;
                        ns += 1;
                    }
                    plan.push(Chunk { exec, kind, i, c0: c, c1, round, off: (half_recs - room) * REC, sp0: sp, ns, kps, rel_split: false });
                    room -= c1 - c;
                    c = c1;
                    sp += ns;
                }
                debug_assert_eq!(c, b);
                execs.push(ExecPlan { b, kps, nsplit, sp0: sp, ns: nsplit - sp, rel_split: false });
            } else {
                // (tiny hot rings in tests) no common boundary: divide the file part and the hot part separately - the same
                // keys, summed in a different grouping
                if lo > w0 && !t.unaligned.replace(true) { eprintln!("[inky] --kv-disk: hot ring shorter than a split, sums are grouped differently from the in-memory path"); }
                let spl = |nk: usize| { let kps = KPS_CHUNK * nk.div_ceil(KPS_CHUNK * Self::split_target()); (kps, nk.div_ceil(kps)) };
                let (mut c, mut sp) = (w0, 0usize);
                while c < lo {
                    if room < (lo - c).min(64) { round += 1; room = half_recs; }
                    let n = room.min(lo - c);
                    let (kps, ns) = spl(n);
                    plan.push(Chunk { exec, kind, i, c0: c, c1: c + n, round, off: (half_recs - room) * REC, sp0: sp, ns, kps, rel_split: true });
                    c += n; sp += ns; room -= n;
                }
                let (kps, ns) = spl(pos + 1 - lo);
                assert!(sp + ns <= MAX_SPLIT, "too many K/V file chunks for one layer");
                execs.push(ExecPlan { b: lo, kps, nsplit: sp + ns, sp0: sp, ns, rel_split: true });
            }
        }
        *t.execs.borrow_mut() = execs;
        *t.plan.borrow_mut() = plan;
        t.loaded.set(usize::MAX);
        let mut cx = Cx { e: Enc::new(&cb), cb: Some(cb), prior: Vec::new() };
        self.encode_step_cx(&mut cx, pos, logits, argmax);
        let Cx { e, cb, prior } = cx;
        e.end();
        let cb = cb.unwrap();
        cb.commit();
        // the next step starts in the other staging half
        let h = t.half.get();
        t.pending.borrow_mut()[h] = Some(cb.clone());
        t.half.set(h ^ 1);
        DecodeStep { prior, last: cb }
    }
    /// make the staging round of chunk `k` of the plan available: switch halves if it is a new one and read it from the file
    fn tier_load(&self, cx: &mut Cx, k: usize) {
        let t = self.lane.tier.as_ref().unwrap();
        let plan = t.plan.borrow();
        let round = plan[k].round;
        if t.loaded.get() == round { return; }
        if round > 0 {
            // hand the commands written so far to the GPU; they keep reading the half we are leaving
            let ncb = self.gpu.cmd();
            let oe = std::mem::replace(&mut cx.e, Enc::new(&ncb));
            oe.end();
            let ocb = cx.cb.replace(ncb).unwrap();
            ocb.commit();
            let h = t.half.get();
            t.pending.borrow_mut()[h] = Some(ocb.clone());
            cx.prior.push(ocb);
            t.half.set(h ^ 1);
            self.bind(&cx.e);
        }
        // whatever last used this half must be finished before it is overwritten
        let h = t.half.get();
        let prev = t.pending.borrow_mut()[h].take();
        if let Some(cb) = prev { cb.waitUntilCompleted(); }
        let stg = t.stg.borrow();
        let base = unsafe { (stg.contents().as_ptr() as *mut u8).add(h * (stg.length() / 2)) };
        let mut jobs = Vec::new();
        for c in plan.iter().filter(|c| c.round == round) {
            let mut o = c.off;
            for (off, n) in self.tier_pieces(c.kind, c.i, c.c0, c.c1) { jobs.push((off, n * REC, unsafe { base.add(o) })); o += n * REC; }
        }
        self.tier_read(&jobs);
        t.loaded.set(round);
    }

    /// conv state and sliding-window rings, to rerun the same decode steps
    pub fn snapshot(&self) -> (Vec<u8>, Vec<u8>) {
        let g = |b: &Buf| unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const u8, b.length()).to_vec() };
        (self.get_cs(), g(&self.lane.kv_swa.borrow()))
    }
    pub fn restore(&self, s: &(Vec<u8>, Vec<u8>)) {
        self.set_cs(&s.0);
        unsafe {
            std::ptr::copy_nonoverlapping(s.1.as_ptr(), self.lane.kv_swa.borrow().contents().as_ptr() as *mut u8, s.1.len());
        }
    }
    /// Score every far key (older than one window at context `n`) of the full-attention slots by the attention it received
    /// from a sample of the prompt's own queries, keep the best `keep` share per (slot, group) and optionally one summary
    /// key (mean of the evicted keys and values, score bias ln(count)). Returns (far keys, kept per group).
    pub fn prune_build(&self, n: usize, keep: f64, use_sum: bool, nq: usize, mode: &str) -> (usize, usize) {
        assert!(!self.kv8 && !self.tier_on() && n > EXT);
        let qc = self.qcap.borrow();
        let (qbuf, ncap) = qc.as_ref().expect("qcap_alloc + prefill first");
        let r0 = n - EXT;
        let f = ((r0 as f64 * keep).round() as usize).min(r0);
        let rows = f + 1;
        let nf = self.slot_kind.iter().filter(|k| !k.0).count();
        let bag = self.gpu.buffer(nf * 2 * NKV * rows * HD * 2);
        let lut: Vec<f32> = (0..65536u32).map(|h| { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else if e == 31 { 0.0 } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } }).collect();
        let f2h = |v: f32| -> u16 { let bits = v.to_bits(); let sign = ((bits >> 16) & 0x8000) as u16; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign } else { let mant = (bits >> 13) & 0x3ff; let rem = bits & 0x1fff;
                let mut h = ((exp as u32) << 10 | mant) + if rem > 0x1000 || (rem == 0x1000 && mant & 1 == 1) { 1 } else { 0 }; if h >= 0x7c00 { h = 0x7bff; } sign | h as u16 } };
        let cap = self.lane.full_cap.get();
        let (kvp, qp, bp) = (self.lane.kv_full.borrow().contents().as_ptr() as usize, qbuf.contents().as_ptr() as usize, bag.contents().as_ptr() as usize);
        let per = ncap * (NH * HD / 2 + NH) * 4;
        // sampled query positions: spread over the context, denser at the end
        let mut qpos: Vec<usize> = (0..nq).map(|i| (n - 1) - i * (n - 1) / nq.max(1)).collect();
        qpos.dedup();
        let tasks: Vec<(usize, usize)> = (0..nf).flat_map(|fi| (0..NKV).map(move |g| (fi, g))).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (lut, qpos, tasks, f2h) = (&lut, &qpos, &tasks, &f2h);
        std::thread::scope(|sc| {
            for _ in 0..8 { sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= tasks.len() { break; }
                let (fi, g) = tasks[k];
                let row = |kv: usize, j: usize| unsafe { std::slice::from_raw_parts((kvp as *const u16).add(((fi * 2 + kv) * NKV + g) * cap * HD + j * HD), HD) };
                let keys: Vec<f32> = (0..n).flat_map(|j| row(0, j).iter().map(|&h| lut[h as usize])).collect();
                let mut imp = vec![0f32; n];
                let mut sc_ = vec![0f32; n];
                for &t in qpos.iter() { for hh in 0..3 {
                    let h = g * 3 + hh;
                    let q: Vec<f32> = unsafe { std::slice::from_raw_parts((qp + fi * per) as *const u16, ncap * NH * HD) }[(t * NH + h) * HD..(t * NH + h + 1) * HD].iter().map(|&x| lut[x as usize]).collect();
                    let sink = unsafe { *((qp + fi * per + ncap * NH * HD * 2) as *const f32).add(t * NH + h) };
                    let mut m = sink;
                    for j in 0..=t {
                        let kr = &keys[j * HD..(j + 1) * HD];
                        let mut a = [0f32; 8];
                        for c in (0..HD).step_by(8) { for u in 0..8 { a[u] += q[c + u] * kr[c + u]; } }
                        let s = a.iter().sum::<f32>();
                        sc_[j] = s; if s > m { m = s; }
                    }
                    let mut z = (sink - m).exp();
                    for j in 0..=t { sc_[j] = (sc_[j] - m).exp(); z += sc_[j]; }
                    match mode {
                        "max" => for j in 0..=t { let p = sc_[j] / z; if p > imp[j] { imp[j] = p; } },
                        // only what a key received while it was inside one window of the query / only from further away
                        "near" => for j in t.saturating_sub(EXT - 1)..=t { imp[j] += sc_[j] / z; },
                        "far" => for j in 0..=t.saturating_sub(EXT) { imp[j] += sc_[j] / z; },
                        _ => for j in 0..=t { imp[j] += sc_[j] / z; },
                    }
                } }
                if mode == "rand" { for (j, x) in imp.iter_mut().enumerate() { *x = ((j as u64 * 0x9E3779B97F4A7C15 + (k as u64) * 0xD1B54A32D192ED03) >> 40) as f32; } }
                let mut order: Vec<usize> = (0..r0).collect();
                order.sort_by(|&a, &b| imp[b].partial_cmp(&imp[a]).unwrap());
                let mut kept: Vec<usize> = order[..f].to_vec();
                kept.sort();
                let out = |kv: usize, r: usize| unsafe { std::slice::from_raw_parts_mut((bp as *mut u16).add(((fi * 2 + kv) * NKV + g) * rows * HD + r * HD), HD) };
                for kv in 0..2 {
                    for (r, &j) in kept.iter().enumerate() { out(kv, r).copy_from_slice(row(kv, j)); }
                    let mut mean = vec![0f64; HD];
                    for &j in &order[f..] { for (mm, &h) in mean.iter_mut().zip(row(kv, j)) { *mm += lut[h as usize] as f64; } }
                    let cnt = (r0 - f).max(1) as f64;
                    for (o, mm) in out(kv, f).iter_mut().zip(&mean) { *o = f2h((*mm / cnt) as f32); }
                }
            }); }
        });
        drop(qc);
        *self.lane.prune.borrow_mut() = Some(Prune { bag, rows, r0, lastbias: ((r0 - f).max(1) as f32).ln(), use_sum: use_sum && r0 > f });
        (r0, f)
    }
    /// dev: how peaked is full attention? After a prefill of `n` tokens (tail mode off) the arena still holds the last layer's
    /// prepared queries. For the last `nq` query tokens and every head: how many of its keys score within `margin` logits of
    /// the best one (keys further down contribute less than e^-margin of it). Relative bias is ignored (it only exists within
    /// 1024 tokens). Returns per margin the mean fraction of keys that are within it, and the worst head's fraction.
    pub fn kv_screen(&self, n: usize, nq: usize, margins: &[f32]) -> Vec<(f32, f64, f64)> {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let b = self.ba.get();
        let slot = N_PHYS - 1;
        assert!(!self.slot_kind[slot].0, "last layer is expected to be a full-attention layer");
        let (kb, _, cap) = self.kv_slot(slot);
        let h2f = |h: u16| -> f32 { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } };
        let kv = self.lane.kv_full.borrow();
        let kp = kv.contents().as_ptr() as *const u16;
        let act = self.act.borrow();
        let qp = unsafe { (act.contents().as_ptr() as *const u16).add(b.qp as usize * 2) };
        let sink = unsafe { (act.contents().as_ptr() as *const f32).add(b.sink as usize) };
        let mut within = vec![(0.0f64, 0.0f64); margins.len()];
        let mut cnt = 0usize;
        for g in 0..NKV {
            // keys of this group as f32
            let keys: Vec<f32> = (0..n * HD).map(|i| h2f(unsafe { *kp.add(kb as usize + g * cap as usize * HD + i) })).collect();
            for t in n - nq..n { for hh in 0..3 {
                let h = g * 3 + hh;
                let q: Vec<f32> = (0..HD).map(|d| h2f(unsafe { *qp.add((t * NH + h) * HD + d) })).collect();
                let sc: Vec<f32> = (0..=t).map(|j| { let k = &keys[j * HD..(j + 1) * HD]; q.iter().zip(k).map(|(a, c)| a * c).sum::<f32>() }).collect();
                let m = sc.iter().cloned().fold(unsafe { *sink.add(t * NH + h) }, f32::max);
                for (i, &mg) in margins.iter().enumerate() {
                    let f = sc.iter().filter(|&&x| x > m - mg).count() as f64 / sc.len() as f64;
                    within[i].0 += f; within[i].1 = within[i].1.max(f);
                }
                cnt += 1;
            } }
        }
        margins.iter().zip(within).map(|(&m, (s, w))| (m, s / cnt as f64, w)).collect()
    }
    /// dev: what would storing the cache at `bits` per value cost in accuracy? Rounds the keys and values of positions
    /// [from, to) in place to a per-vector linear grid (the storage stays fp16; nothing gets smaller). kbits / vbits = 0: untouched.
    pub fn kv_quant_sim(&self, from: usize, to: usize, kbits: u32, vbits: u32) {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let h2f = |h: u16| -> f32 { let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 }; let e = ((h >> 10) & 31) as i32; let m = (h & 1023) as f32;
            if e == 0 { sg * m * 2f32.powi(-24) } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) } };
        let f2h = |v: f32| -> u16 { let bits = v.to_bits(); let sign = ((bits >> 16) & 0x8000) as u16; let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            if v == 0.0 || exp <= 0 { sign } else {
                let mant = (bits >> 13) & 0x3ff; let rem = bits & 0x1fff;
                let mut h = ((exp as u32) << 10 | mant) + if rem > 0x1000 || (rem == 0x1000 && mant & 1 == 1) { 1 } else { 0 };
                if h >= 0x7c00 { h = 0x7bff; }
                sign | h as u16 } };
        for slot in 0..N_SLOT {
            let (is_swa, _) = self.slot_kind[slot];
            let (kb, vb, cap) = self.kv_slot(slot);
            let buf = if is_swa { self.lane.kv_swa.borrow() } else { self.lane.kv_full.borrow() };
            let base = buf.contents().as_ptr() as *mut u16;
            for (b, bits) in [(kb, kbits), (vb, vbits)] {
                if bits == 0 { continue; }
                let lv = ((1u32 << (bits - 1)) - 1) as f32;
                for g in 0..NKV { for pos in from.max(if is_swa { to.saturating_sub(EXT) } else { 0 })..to {
                    let idx = if is_swa { pos % cap as usize } else { pos };
                    let row = unsafe { std::slice::from_raw_parts_mut(base.add(b as usize + (g * cap as usize + idx) * HD), HD) };
                    let v: Vec<f32> = row.iter().map(|&h| h2f(h)).collect();
                    if bits >= 100 {
                        // floating formats: 100 + m = round every value to m mantissa bits; 200 + m = the same with a 3-bit exponent
                        // (binades [0.25, 32), fixed step 2^-(m+2) below)
                        let (m, lim) = ((bits % 100) as i32, bits >= 200);
                        for (r, &x) in row.iter_mut().zip(&v) {
                            if x == 0.0 { continue; }
                            let mut e = x.abs().log2().floor() as i32;
                            if lim { e = e.clamp(-2, 4); }
                            let q = 2f32.powi(e - m);
                            let mut y = (x / q).round() * q;
                            if lim { y = y.clamp(-(2f32.powi(5) - q), 2f32.powi(5) - q); }
                            *r = f2h(y);
                        }
                        continue;
                    }
                    let mx = v.iter().fold(0f32, |a, x| a.max(x.abs()));
                    if mx == 0.0 { continue; }
                    for (r, x) in row.iter_mut().zip(&v) { *r = f2h((x / mx * lv).round() * mx / lv); }
                } }
            }
        }
    }
    /// dev: raw bytes of the K/V cache holding real keys after `n` positions: (sliding-window rings, full-attention rows)
    pub fn kv_dump(&self, n: usize) -> (Vec<u8>, Vec<u8>) {
        assert!(!self.kv8, "this dev tool reads the fp16 cache");
        let (mut swa, mut full) = (Vec::new(), Vec::new());
        for slot in 0..N_SLOT {
            let (is_swa, _) = self.slot_kind[slot];
            let (kb, vb, cap) = self.kv_slot(slot);
            let buf = if is_swa { self.lane.kv_swa.borrow() } else { self.lane.kv_full.borrow() };
            let base = buf.contents().as_ptr() as *const u8;
            let rows = if is_swa { n.min(EXT) } else { n };
            for b in [kb, vb] { for g in 0..NKV {
                let sl = unsafe { std::slice::from_raw_parts(base.add((b as usize + g * cap as usize * HD) * 2), rows * HD * 2) };
                if is_swa { swa.extend_from_slice(sl); } else { full.extend_from_slice(sl); }
            } }
        }
        (swa, full)
    }

    /// a copy of this lane's convolution rows (its recurrent state at the current position)
    pub fn get_cs(&self) -> Vec<u8> {
        self.lane.with_cs_buf(|b, off, len| unsafe {
            std::slice::from_raw_parts((b.contents().as_ptr() as *const u8).add(off), len).to_vec()
        })
    }
    /// put back convolution rows `get_cs` copied. Nothing may be in flight on the GPU.
    pub fn set_cs(&self, v: &[u8]) {
        self.lane.with_cs_buf(|b, off, len| {
            assert_eq!(v.len(), len);
            unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), (b.contents().as_ptr() as *mut u8).add(off), len); }
        });
    }
    pub fn reset(&self) {
        self.tier_reset();
        self.lane.with_cs_buf(|b, off, len| unsafe {
            std::ptr::write_bytes((b.contents().as_ptr() as *mut u8).add(off), 0, len);
        });
    }

    pub fn set_tokens(&self, start: usize, toks: &[u32]) {
        self.try_set_tokens(start, toks).unwrap_or_else(|err| panic!("{err}"));
    }
    pub fn try_set_tokens(&self, start: usize, toks: &[u32]) -> Result<(), String> {
        let capacity = self.max_ctx.min(self.lane.toks.length() / size_of::<u32>());
        // The embedding kernel reads signed token IDs.
        let limit = self.n_vocab.min(self.tok_embd.rows as usize).min(i32::MAX as usize + 1);
        validate_token_write(start, toks, capacity, limit)?;
        let p = self.lane.toks.contents().as_ptr().cast::<u32>();
        unsafe { std::ptr::copy_nonoverlapping(toks.as_ptr(), p.add(start), toks.len()); }
        Ok(())
    }
    pub fn token_at(&self, i: usize) -> u32 { unsafe { *(self.lane.toks.contents().as_ptr() as *const i32).add(i) as u32 } }
    /// Read a checked GPU sample after its command buffer completed.
    pub fn sampled_token_at(&self, pos: usize) -> Result<u32, &'static str> {
        let tok = self.token_at(pos);
        if tok == u32::MAX { Err("model returned invalid sampling logits") } else { Ok(tok) }
    }
    /// Encode one full decode step for token at `pos` (input token read from toks[pos]).
    /// If `logits` is set, also computes lm_head and writes argmax to toks[pos+1].
    pub fn encode_step(&self, cb: &CmdBuf, pos: usize, logits: bool, argmax: bool) {
        assert!(!self.tier_on(), "with the disk tier a step spans several command buffers: use step()");
        assert!(pos < self.kv_capacity(), "K/V capacity: call reserve() first");
        let mut cx = Cx { e: Enc::new(cb), cb: None, prior: Vec::new() };
        self.encode_step_cx(&mut cx, pos, logits, argmax);
        cx.e.end();
    }
    fn encode_step_cx(&self, cx: &mut Cx, pos: usize, logits: bool, argmax: bool) {
        self.guided_logit_rows.set(0);
        let omega = self.loopcd_omega.get();
        let run_loopcd = logits && omega > 0.0;
        assert!(!run_loopcd || !self.tier_on(), "--loopcd cannot be combined with --kv-disk");
        let e = &cx.e;
        self.bind(e);
        let ar = self.ar_at();
        // embedding -> vec 0 (x0)
        let ea = EmbedArgs { emb_lo: self.tok_embd.off as u32, emb_hi: (self.tok_embd.off >> 32) as u32, row_bytes: self.tok_embd.row_bytes,
                             norm_off: self.tok_norm, out0: self.v(0), out1: self.v(0), tok_index: pos as u32, eps: self.eps };
        self.go(e, C_EMBED, "k_embed", &ea, (1, 1, 1), 256);

        self.done.set(0);
        let mut va = VecAlloc { next: 1, n: N_VEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.layer_ex(cx, pos, ip, ip, cur, &mut blocks, None, &mut va, &[], false); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        let mut lg_weak_buf: Option<Buf> = None;
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = self.layer_ex(cx, pos, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned, false);
            }
            bank = (*blocks.get(prelude_blocks.len() + 1).unwrap_or(&0), cur);
            if pass == 0 && run_loopcd {
                // Exact dry-state weak reference pass (z_1) immediately after pass 0, while CS[18..24] and
                // KV[18..24] still hold the untouched pos - 1 state.
                let mut cur_w = cur;
                let mut blocks_w = blocks.clone();
                let pinned_w = [prelude, bank.0, bank.1];
                for ip in LOOP_END..N_PHYS {
                    cur_w = self.layer_ex(cx, pos, ip, ip, cur_w, &mut blocks_w, None, &mut va, &pinned_w, true);
                }
                let e = &cx.e;
                let mut st_w = blocks_w.clone(); st_w.push(cur_w);
                let j = self.mix_job(&st_w, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32]);
                self.mix(e, C_OMIX, &[j]);
                self.mv(e, C_LMHEAD, &[Engine::seg(&self.output, ar.fout, ar.logits)]);
                let weak_buf = self.ensure_lg_weak(1);
                let n_words = self.vocab_pad();
                let dc = DevCopyArgs { src: ar.logits, dst: 0, n: n_words as u32, pad: 0 };
                e.buf(&weak_buf, 0, 3);
                self.go(e, C_LMHEAD, "k_devcopy", &dc, (n_words.div_ceil(1024), 1, 1), 1024);
                self.bind_kv(e, true);
                lg_weak_buf = Some(weak_buf);
            }
        }
        for ip in LOOP_END..N_PHYS { cur = self.layer_ex(cx, pos, ip, ip, cur, &mut blocks, None, &mut va, &[], false); }

        let e = &cx.e;
        if logits {
            let mut st = blocks.clone(); st.push(cur);
            let j = self.mix_job(&st, self.output_res_query as i32, 1, &[ar.fout], &[self.output_norm as i32]);
            self.mix(e, C_OMIX, &[j]);
            self.mv(e, C_LMHEAD, &[Engine::seg(&self.output, ar.fout, ar.logits)]);
            if let Some(weak_buf) = lg_weak_buf.as_ref() {
                let la = LoopCdArgs {
                    logits_off: ar.logits,
                    n: self.n_vocab as u32,
                    stride: self.vocab_pad() as u32,
                    pad0: 0,
                    softcap: self.softcap,
                    pad1: 0,
                    pad2: 0,
                    pad3: 0,
                };
                let omegas = [omega];
                e.bytes_slice(&omegas, 8);
                e.buf(weak_buf, 0, 9);
                self.go(e, C_LMHEAD, "k_loopcd_blend", &la, (1, 1, 1), 1024);
                self.guided_logit_rows.set(1);
            }
        }
        if logits && argmax {
            let nparts = 128u32;
            let am = ArgmaxArgs { logits_off: ar.logits, n: self.n_vocab as u32, part_off: ar.amax, nparts, out_index: pos as u32 + 1, checked: 0, pad: [0; 2] };
            self.go(e, C_ARGMAX, "k_argmax1", &am, (nparts as usize, 1, 1), 1024);
            self.go(e, C_ARGMAX, "k_argmax2", &am, (1, 1, 1), 32);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(&self, cx: &mut Cx, pos: usize, slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
             msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]) -> usize {
        self.layer_ex(cx, pos, slot, ip, cur, blocks, msg, va, pinned, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_ex(&self, cx: &mut Cx, pos: usize, slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
                msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize], dry_state: bool) -> usize {
        if !dry_state {
            if self.done.get() >= self.stop.get() { return cur; }
            self.done.set(self.done.get() + 1);
        }
        let e = &cx.e;
        let l = &self.layers[ip];
        let ar = self.ar_at();
        let ctl = msg.map(|_| &self.loops[ip - LOOP_START]);
        let cs_slot = (slot * NCH * 3) as u32;

        // ---- attention inputs: depth mix (+ loop control) -> normalized q/k/v inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st = blocks.clone(); st.push(cur);
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        let xin: [u32; 3];
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            // raw mixed inputs u_j -> inp[j]; RMSNorm'd copies -> cat[j][0:E]; messages m_j -> msg[j], normalized -> cat[j][E:2E]
            let mut ja = self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &[]);
            if l.swa { ja.ncopy[0] = [ar.cat[0] as i32, ar.cat[1] as i32, ar.cat[2] as i32]; }
            else { for j in 0..3 { ja.ncopy[j][0] = ar.cat[j] as i32; } }
            let mut jm = self.mix_job(&[b0, b1], lw.msg_query as i32, 4, &ar.msg, &[]);
            for j in 0..4 { jm.ncopy[j][0] = (ar.cat[j] + E as u32) as i32; }
            self.mix(e, C_MIXA, &[ja, jm]);
            let mut cj = Vec::new();
            for j in 0..3 {
                let u = if l.swa { ar.inp[0] } else { ar.inp[j] };
                cj.push(CtrlJob { u_off: u, m_off: ar.msg[j], d_off: lw.down[j], out_off: ar.ctrlg[j], gain_off: lw.gain[j],
                                  forget_off: lw.forget[j], res_out: ar.cin[j], ss_off: ar.ssp[j], norm_w: norms[j], ctr: 8 + j as u32 });
            }
            self.ctrl_dec(e, &cj);
            xin = ar.cin;
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            self.mix(e, C_MIXA, &[self.mix_job(&st, l.attn_res_query, nq, &ar.inp[..nq as usize], &nn)]);
            xin = if l.swa { [ar.inp[0]; 3] } else { [ar.inp[0], ar.inp[1], ar.inp[2]] };
        }
        // ---- fused projections (one dispatch)
        let mut segs = vec![Engine::seg(&l.wq, xin[0], ar.q), Engine::seg(&l.wgate, xin[0], ar.gate), Engine::seg(&l.wrel, xin[0], ar.rel),
                            Engine::seg(&l.wsink, xin[0], ar.sink), Engine::seg(&l.wk, xin[1], ar.k), Engine::seg(&l.wv, xin[2], ar.v)];
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) { segs.push(Engine::seg(kp, xin[1], ar.kdyn)); segs.push(Engine::seg(vp, xin[2], ar.vdyn)); }
        self.mv(e, C_QKV, &segs);
        // ---- attention (split-K) with the split combine + output gate fused into the last split per kv group
        let (kb, vb, kcap) = self.kv_slot(slot);
        let tau = if l.swa { 1.0 } else { 1.0 + self.log_alpha * (((pos + 1) as f32 / self.log_floor).max(1.0)).ln() };
        let base = AttnArgs {
            pos: pos as u32, cap: kcap, k_base: kb, v_base: vb, cs_off: cs_slot,
            q_off: ar.q, k_off: ar.k, v_off: ar.v, r_off: ar.rel, sink_off: ar.sink,
            kdyn_off: if l.swa { -1 } else { ar.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { ar.vdyn as i32 }, part_off: ar.part,
            kconv_w: l.k_conv, vconv_w: l.v_conv, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias, tau,
            ctr: 0, gate_off: ar.gate, out_off: ar.attn_g, gstr: kcap * HD as u32, istr: HD as u32, ..Default::default()
        };
        let nsplit;
        if self.tier_on() {
            // disk tier: the keys older than the hot ring come from the file (staged in chunks, each its own dispatch writing
            // its share of the split partials); the hot ring is one more dispatch and also stores this token's K/V
            let t = self.lane.tier.as_ref().unwrap();
            let exec = self.done.get() - 1;
            let x = t.execs.borrow()[exec];
            let chunks: Vec<(usize, Chunk)> = t.plan.borrow().iter().cloned().enumerate().filter(|(_, c)| c.exec == exec).collect();
            for (k, c) in chunks {
                self.tier_load(cx, k);
                let e = &cx.e;
                let off = ((t.half.get() * (t.stg.borrow().length() / 2) + c.off) / 2) as u32;       // in fp16 elements
                let a = AttnArgs { nkeys: (c.c1 - c.c0) as u32, first_pos: c.c0 as u32, ring: 0, idx0: c.c0 as u32, k_base: off, v_base: off + (NKV * HD) as u32,
                                   gstr: HD as u32, istr: (2 * NKV * HD) as u32, kps: c.kps as u32, nsplit: x.nsplit as u32, sp0: c.sp0 as u32, noown: 1,
                                   pad: 1 | if c.rel_split { 4 } else { 0 }, ..base };
                e.buf(&t.stg.borrow(), 0, 3);
                self.go(e, C_ATTN, "k_attn", &a, (c.ns, NKV, 1), 128);
            }
            let e = &cx.e;
            let a = AttnArgs { nkeys: (pos + 1 - x.b) as u32, first_pos: x.b as u32, ring: 1, kps: x.kps as u32, nsplit: x.nsplit as u32, sp0: x.sp0 as u32,
                               pad: 1 | if x.rel_split { 4 } else { 0 }, ..base };
            self.bind_kv(e, l.swa);
            self.go(e, C_ATTN, "k_attn", &a, (x.ns, NKV, 1), 128);
            let total = x.nsplit;
            nsplit = total;
        } else if let (false, Some(pr)) = (l.swa, self.lane.prune.borrow().as_ref()) {
            // dev: far keys from the selection, recent keys from the cache
            let fi = self.slot_kind[slot].1;
            let spl = |nk: usize| { let kps = KPS_CHUNK * nk.div_ceil(KPS_CHUNK * Self::split_target()); (kps, nk.div_ceil(kps)) };
            let nb = if pr.use_sum { pr.rows } else { pr.rows - 1 };
            let ((k1, n1), (k2, n2)) = (if nb > 0 { spl(nb) } else { (32, 0) }, spl(pos + 1 - pr.r0));
            if nb > 0 {
                let kb2 = (fi * 2 * NKV * pr.rows * HD) as u32;
                let a = AttnArgs { nkeys: nb as u32, first_pos: 0, ring: 0, cap: pr.rows as u32, k_base: kb2, v_base: kb2 + (NKV * pr.rows * HD) as u32,
                                   gstr: (pr.rows * HD) as u32, kps: k1 as u32, nsplit: (n1 + n2) as u32, sp0: 0, noown: 1, pad: 1 | 4,
                                   lastbias: if pr.use_sum { pr.lastbias } else { 0.0 }, ..base };
                e.buf(&pr.bag, 0, 3);
                self.go(e, C_ATTN, "k_attn", &a, (n1, NKV, 1), 128);
            }
            let a = AttnArgs { nkeys: (pos + 1 - pr.r0) as u32, first_pos: pr.r0 as u32, ring: 0, kps: k2 as u32, nsplit: (n1 + n2) as u32, sp0: n1 as u32, pad: 1 | 4 | ((dry_state as u32) << 1), ..base };
            self.bind_kv(e, false);
            self.go(e, C_ATTN, "k_attn", &a, (n2, NKV, 1), 128);
            nsplit = n1 + n2;
        } else {
            // ---- attention (split-K) with the split combine + output gate fused into the last split per kv group
            self.bind_kv(e, l.swa);
            let (first_pos, kps, _sp_lo, nsplit_val) = decode_split_grid(l.swa, pos, self.max_ctx);
            let nkeys = pos + 1 - first_pos;
            nsplit = nsplit_val;
            let aa = AttnArgs { nkeys: nkeys as u32, first_pos: first_pos as u32, ring: l.swa as u32, kps: kps as u32, nsplit: nsplit as u32,
                                pad: (self.sep_comb as u32) | ((dry_state as u32) << 1), ..base };
            self.go(e, C_ATTN, "k_attn", &aa, (nsplit, NKV, 1), 128);
        }
        let e = &cx.e;
        if self.sep_comb || self.tier_on() || self.lane.prune.borrow().is_some() {
            let ca = CombineArgs { part_off: ar.part, nsplit: nsplit as u32, gate_off: ar.gate, out_off: ar.attn_g, hout: 0, pad: [0; 3] };
            self.go(e, C_COMB, "k_attn_combine", &ca, (NH, 1, 1), 128);
        }
        // ---- output projection + post-conv (+ residual unless this layer starts a new block)
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        let mut so = Engine::seg(&l.wo, ar.attn_g, self.v(mid));
        so.epi = if dry_state { 2 } else { 1 }; so.res_off = if block_start { -1 } else { self.v(cur) as i32 };
        so.cs_off = cs_slot + (2 * NKV * HD * 3) as u32; so.convw_off = l.attn_post_conv;
        self.mv(e, C_WO, &[so]);
        if block_start { blocks.push(cur); }
        // ---- FFN input: depth mix (+ control) -> normalized
        let mut st = blocks.clone(); st.push(mid);
        let x_off = if let Some(lw) = ctl {
            let mut jf = self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[]);
            jf.ncopy[0][0] = ar.cat[3] as i32;
            self.mix(e, C_MIXF, &[jf]);
            self.ctrl_dec(e, &[CtrlJob { u_off: ar.fin, m_off: ar.msg[3], d_off: lw.down[3], out_off: ar.ctrlg[3], gain_off: lw.gain[3],
                                         forget_off: lw.forget[3], res_out: ar.finc, ss_off: ar.ssp[3], norm_w: l.ffn_norm as i32, ctr: 11 }]);
            ar.finc
        } else {
            self.mix(e, C_MIXF, &[self.mix_job(&st, l.ffn_res_query as i32, 1, &[ar.fin], &[l.ffn_norm as i32])]);
            ar.fin
        };
        // ---- router with fused top-k (last threadgroup)
        let sel = if dry_state { ar.sel } else { ar.sel_log + ((self.done.get() - 1) * 16) as u32 };
        let rpost = MvArgs { post: (!self.sep_topk) as u32, ctr: 12, bias_off: l.exp_bias, sel_off: sel, wscale: self.wscale, ..Default::default() };
        self.mv_post(e, C_ROUTER, &[Engine::seg_pb(l.router, NEXP as u32, E as u32, x_off, ar.rlog)], rpost);
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let topk_args = MoeArgs { logits_off: ar.rlog, bias_off: l.exp_bias, sel_off: sel, wscale: self.wscale, ..Default::default() };
        if self.sep_topk { self.go(e, C_ROUTER, "k_topk", &topk_args, (1, 1, 1), 32); }
        let ma = MoeArgs {
            logits_off: ar.rlog, bias_off: l.exp_bias, sel_off: sel, x_off,
            norm_off: -1, ss_off: -1, ss_n: 0, eps: self.eps,
            gu_lo: l.gu.off as u32, gu_hi: (l.gu.off >> 32) as u32, gu_expert_bytes: l.gu.expert_bytes as u32, gu_row_bytes: l.gu.row_bytes,
            act_off: ar.act, wscale: self.wscale, beta: self.situ_beta, lbeta: self.situ_lbeta,
            dn_lo: l.dn.off as u32, dn_hi: (l.dn.off >> 32) as u32, dn_expert_bytes: l.dn.expert_bytes as u32, dn_row_bytes: l.dn.row_bytes,
            dn_type: l.dn.ty, out_off: self.v(out), res_off: self.v(mid) as i32, cs_off: cs_slot + ((2 * NKV * HD + E) * 3) as u32,
            convw_off: l.ffn_post_conv, gu_type: l.gu.ty, gu_gs_lo: l.gu.gs as u32, gu_gs_hi: (l.gu.gs >> 32) as u32,
            dn_gs_lo: l.dn.gs as u32, dn_gs_hi: (l.dn.gs >> 32) as u32, no_state_write: dry_state as u32, pad1: 0,
        };
        // ---- experts
        self.go(e, C_GU, "k_moe_gu", &ma, (FF / 4, TOPK, 1), 128);
        self.go(e, C_DOWN, "k_moe_down", &ma, (E / 4, 1, 1), 64);
        if !dry_state {
            self.last_out.set(out); self.last_mid.set(mid);
        }
        out
    }

    // ======================================================================== batched prefill
    /// Encode prefill of `t` tokens (toks[p0..p0+t]) of this lane as one batch: a step with one span.
    pub fn encode_prefill(&self, cb: &CmdBuf, p0: usize, t: usize, logits: bool, argmax: bool) {
        if logits && self.loopcd_omega() > 0.0 {
            // A guided first draw needs both the one-pass and full-depth logits at the last
            // prompt position. Feed it through the same private decode path as later draws.
            if t > 1 {
                self.eng.encode_spans(cb, &[Span { lane: self.lane, p0, t: t - 1 }], false, false);
            }
            self.encode_step(cb, p0 + t - 1, true, argmax);
        } else {
            self.eng.encode_spans(cb, &[Span { lane: self.lane, p0, t }], logits, argmax);
        }
    }
}

fn validate_token_write(start: usize, toks: &[u32], capacity: usize, limit: usize) -> Result<(), String> {
    if start > capacity || toks.len() > capacity - start {
        return Err(format!("cannot write {} tokens at position {start} in a {capacity}-token context", toks.len()));
    }
    if let Some((i, &id)) = toks.iter().enumerate().find(|&(_, &id)| id as usize >= limit) {
        return Err(format!("token {id} at position {} is outside the embedding vocabulary (0..{limit})", start + i));
    }
    Ok(())
}

/// Write `half` directly from `k_attn_combine` / `k_attn_combine_dec` into `b.agh` on the fp16 decode path,
/// skipping the standalone `k_tohalf` + barrier unless skip-mask profiling isolates `C_COMB` or `C_WO`.
#[inline]
fn decode_combine_direct_half(hgemm: bool, skip: u32) -> bool {
    hgemm && (skip & ((1 << C_COMB) | (1 << C_WO)) == 0)
}

/// Cache-fixed absolute split width for decode attention (independent of `pos` and batch membership):
/// - Sliding-window (`EXT = 1024`): always 64 tokens (1 block), giving at most `17 <= MAX_SPLIT` splits.
/// - Full attention: `512 * ceil(max_ctx / 131_072).max(1)` (`512` = 8 blocks for all contexts `<= 131,072`),
///   fitting in `MAX_SPLIT = 256` without enlarging `Arena`.
#[inline]
pub fn decode_split_kps(swa: bool, max_ctx: usize) -> usize {
    if swa {
        crate::blocks::BLOCK
    } else {
        512 * max_ctx.div_ceil(131_072).max(1)
    }
}

/// Cache-fixed absolute decode split range `(first_pos, kps, sp_lo, nsplit)` for a token at `pos`.
/// Local split `s` in `0..nsplit` covers absolute split `abs_sp = sp_lo + s`, i.e., absolute positions
/// `[abs_sp * kps, (abs_sp + 1) * kps)` clipped to `[first_pos, pos + 1)`.
#[inline]
pub fn decode_split_grid(swa: bool, pos: usize, max_ctx: usize) -> (usize, usize, usize, usize) {
    let kps = decode_split_kps(swa, max_ctx);
    let first_pos = if swa { (pos + 1).saturating_sub(EXT) } else { 0 };
    let sp_lo = first_pos / kps;
    let sp_hi = pos / kps;
    let nsplit = sp_hi - sp_lo + 1;
    debug_assert!(nsplit <= MAX_SPLIT);
    (first_pos, kps, sp_lo, nsplit)
}

/// Extra `f32` scratch slots appended after `n * ar.total` in `ensure_dbatch(n)` for Layer 3
/// batched decode (`DecodeSpanDesc`, `kv_cur`, SWA `DecAttnGroup`, Full `DecAttnGroup`).
#[inline]
pub fn decode_batch_extra_floats(n: usize) -> usize {
    let align64 = |x: usize| x.div_ceil(64) * 64;
    let desc = align64(n * (std::mem::size_of::<DecodeSpanDesc>() / 4));
    let kv_cur = align64(n * 2 * NKV * HD);
    let grp_floats = std::mem::size_of::<DecAttnGroup>() / 4;
    let swa_grps = align64(n * MAX_SWA_SPLIT * grp_floats);
    let full_grps = align64(n * MAX_SPLIT * grp_floats);
    desc + kv_cur + swa_grps + full_grps
}

/// Build `DecAttnGroup` descriptors for one attention kind (`swa` or `full`) across a decode batch.
///
/// Invariants (matching `specs/GroupedLayout.tla`):
/// - Every `(si, abs_sp)` with `abs_sp in (first_pos / kps)..=(pos / kps)` appears in EXACTLY ONE group.
/// - Only strictly interior full splits (`first_pos <= abs_sp * kps && pos >= (abs_sp + 1) * kps`, so
///   `p == pos` is never in the split and every key in `[abs_sp * kps, (abs_sp + 1) * kps)` is visible)
///   with identical `Some((slot != 0, gen != 0))` across ALL `kps / 64` blocks in `abs_sp` may be grouped,
///   capped at `DEC_GRP_MAX = 8` lanes per group.
/// - Partial/boundary/divergent splits (and all splits when `no_grp` is true) emit `count = 1`.
pub fn build_decode_groups<F>(
    ranges: &[(usize, usize)],
    kps: usize,
    no_grp: bool,
    mut block_entry: F,
) -> Vec<DecAttnGroup>
where
    F: FnMut(usize, usize) -> Option<(u32, u32)>,
{
    debug_assert!(kps >= crate::blocks::BLOCK && kps % crate::blocks::BLOCK == 0);
    let blks_per_split = kps / crate::blocks::BLOCK;
    let n = ranges.len();
    let mut out = Vec::with_capacity(n * if blks_per_split == 1 { MAX_SWA_SPLIT } else { 16 });

    if no_grp || n <= 1 {
        for (si, &(first_pos, pos)) in ranges.iter().enumerate() {
            let sp_lo = first_pos / kps;
            let sp_hi = pos / kps;
            for abs_sp in sp_lo..=sp_hi {
                let mut g = DecAttnGroup { abs_sp: abs_sp as u32, count: 1, span_idx: [0; DEC_GRP_MAX] };
                g.span_idx[0] = si as u16;
                out.push(g);
            }
        }
        return out;
    }

    let mut interior: Vec<(u32, u64, u16)> = Vec::new();
    for (si, &(first_pos, pos)) in ranges.iter().enumerate() {
        let sp_lo = first_pos / kps;
        let sp_hi = pos / kps;
        for abs_sp in sp_lo..=sp_hi {
            let split_start = abs_sp * kps;
            let split_end = split_start + kps;
            let is_interior = first_pos <= split_start && pos >= split_end;
            if !is_interior {
                let mut g = DecAttnGroup { abs_sp: abs_sp as u32, count: 1, span_idx: [0; DEC_GRP_MAX] };
                g.span_idx[0] = si as u16;
                out.push(g);
            } else {
                let blk0 = abs_sp * blks_per_split;
                match block_entry(si, blk0) {
                    Some((slot, gen)) if slot != 0 && gen != 0 => {
                        let key = ((slot as u64) << 32) | (gen as u64);
                        interior.push((abs_sp as u32, key, si as u16));
                    }
                    _ => {
                        let mut g = DecAttnGroup { abs_sp: abs_sp as u32, count: 1, span_idx: [0; DEC_GRP_MAX] };
                        g.span_idx[0] = si as u16;
                        out.push(g);
                    }
                }
            }
        }
    }

    if interior.is_empty() {
        return out;
    }

    interior.sort_unstable();

    let mut i = 0;
    while i < interior.len() {
        let abs_sp = interior[i].0;
        let first_key = interior[i].1;
        let mut j = i + 1;
        while j < interior.len() && interior[j].0 == abs_sp && interior[j].1 == first_key {
            j += 1;
        }
        let run = &interior[i..j];
        if run.len() == 1 || blks_per_split == 1 {
            for chunk in run.chunks(DEC_GRP_MAX) {
                let mut g = DecAttnGroup { abs_sp, count: chunk.len() as u32, span_idx: [0; DEC_GRP_MAX] };
                for (u, item) in chunk.iter().enumerate() {
                    g.span_idx[u] = item.2;
                }
                out.push(g);
            }
        } else {
            let blk0 = abs_sp as usize * blks_per_split;
            let mut open: Vec<(usize, DecAttnGroup)> = Vec::new();
            for &(_, _, si_u16) in run {
                let si = si_u16 as usize;
                let mut all_valid = true;
                for b in 1..blks_per_split {
                    if !matches!(block_entry(si, blk0 + b), Some((s, g)) if s != 0 && g != 0) {
                        all_valid = false;
                        break;
                    }
                }
                if !all_valid {
                    let mut g = DecAttnGroup { abs_sp, count: 1, span_idx: [0; DEC_GRP_MAX] };
                    g.span_idx[0] = si_u16;
                    out.push(g);
                    continue;
                }
                let mut placed = false;
                for (rep_si, grp) in open.iter_mut() {
                    if (grp.count as usize) >= DEC_GRP_MAX {
                        continue;
                    }
                    let mut same = true;
                    for b in 1..blks_per_split {
                        if block_entry(si, blk0 + b) != block_entry(*rep_si, blk0 + b) {
                            same = false;
                            break;
                        }
                    }
                    if same {
                        grp.span_idx[grp.count as usize] = si_u16;
                        grp.count += 1;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    let mut g = DecAttnGroup { abs_sp, count: 1, span_idx: [0; DEC_GRP_MAX] };
                    g.span_idx[0] = si_u16;
                    open.push((si, g));
                }
            }
            for (_, g) in open {
                out.push(g);
            }
        }
        i = j;
    }

    out
}

/// Write `half` directly from `ctrl_b`'s `down` `gemm_h` into `b.gh[c]`, skipping `k_tohalf` + barrier
/// unless skip-mask profiling isolates `C_CTRLD` or `C_CTRLA`.
#[inline]
fn ctrl_down_direct_half(skip: u32) -> bool {
    skip & ((1 << C_CTRLD) | (1 << C_CTRLA)) == 0
}

/// Starting index in `MOE_CLS`: each token selects `TOPK` distinct experts, so when `n <= 64` every expert
/// receives at most 64 tokens and class 0 (`65..128` tokens, `k_moe_sg8`) is guaranteed to have 0 tiles.
#[inline]
fn moe_class_start(n: usize) -> usize {
    if n <= 64 { 1 } else { 0 }
}

/// Use fused 1-TG `k_moe_topk_route3` on the fp16 path whenever `n <= 64`.
#[inline]
fn use_fused_topk_route3(hgemm: bool, n: usize) -> bool {
    hgemm && n <= 64
}

fn guided_rows(omegas: &[f32]) -> u128 {
    const { assert!(MAX_SPANS <= u128::BITS as usize); }
    let mut mask = 0;
    for (i, &omega) in omegas.iter().enumerate() {
        if omega > 0.0 { mask |= 1u128 << i; }
    }
    mask
}

/// Pure Rust reference implementation of `k_loopcd_blend` (Liu et al., 2026, arXiv:2610.02185,
/// Eqs. 3-4 & Appendix A.1-A.3) with exact logit softcapping:
///   s3(v) = softcap(z3(v)), s1(v) = softcap(z1(v))
///   omega_t = omega_max * clamp(1 - (p_{3,(1)} - p_{3,(2)}), 0, 1)
///   z3[v] <- s3(v) + omega_t * (s3(v) - s1(v))
/// Returns the adaptive weight. A disabled row stays raw for its regular sampler.
#[cfg_attr(not(test), allow(dead_code))]
pub fn loopcd_blend_row(z3: &mut [f32], z1: &[f32], softcap: f32, omega_max: f32) -> f32 {
    assert_eq!(z3.len(), z1.len(), "loopcd_blend_row: z3 and z1 must have equal length");
    let cap = |x: f32| if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x };
    if z3.is_empty() || omega_max <= 0.0 { return 0.0; }
    let mut max1 = f32::NEG_INFINITY;
    let mut max2 = f32::NEG_INFINITY;
    for &raw in z3.iter() {
        let s3 = cap(raw);
        if s3 >= max1 {
            max2 = max1;
            max1 = s3;
        } else if s3 > max2 {
            max2 = s3;
        }
    }
    let mut z = 0.0f32;
    for &raw in z3.iter() {
        z += (cap(raw) - max1).exp();
    }
    let omega_t = if z.is_finite() && z > 0.0 {
        let p1 = 1.0 / z;
        let p2 = if max2 > -1e30 { (max2 - max1).exp() / z } else { 0.0 };
        omega_max * (1.0 - (p1 - p2)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    for (dst, &raw1) in z3.iter_mut().zip(z1.iter()) {
        if !dst.is_finite() || !raw1.is_finite() {
            *dst = f32::NAN;
            continue;
        }
        let s3 = cap(*dst);
        if omega_t > 0.0 {
            let s1 = cap(raw1);
            *dst = s3 + omega_t * (s3 - s1);
        } else {
            *dst = s3;
        }
    }
    omega_t
}

#[cfg(test)]
mod sampling_tests;

#[cfg(test)]
mod tests {
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
        let src = format!("{}\n{}", include_str!("kernels.metal"), include_str!("prefill.metal"));
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
        let kernels_src = include_str!("kernels.metal");
        let prefill_src = include_str!("prefill.metal");
        let bench_src = include_str!("bench_kernels.metal");
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
        let gpu = Gpu::new(include_str!("kernels.metal"), &["k_loopcd_blend"]);
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
}
