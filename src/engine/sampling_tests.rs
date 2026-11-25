use super::{ArgmaxArgs, SampArgs};
use crate::metal::{Enc, Gpu};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCreateSystemDefaultDevice,
};

// Three partitions exercise both stages of the reduction, including a partial final group.
const WIDTH: usize = 2049;
const PARTS: usize = WIDTH.div_ceil(1024);

fn sample_rows(gpu: &Gpu, rows: &[[f32; WIDTH]; 2], temps: [f32; 2], bans: [u32; 2]) -> [u32; 2] {
    let score_words = rows.len() * WIDTH;
    let arena = gpu.buffer((score_words + rows.len() * 2 * PARTS) * size_of::<f32>());
    let tokens = gpu.buffer(rows.len() * size_of::<u32>());
    // The buffers are idle and the logits occupy the first score_words words.
    unsafe {
        std::ptr::copy_nonoverlapping(
            rows.as_ptr().cast::<f32>(),
            arena.contents().as_ptr().cast(),
            score_words,
        );
    }

    let cb = gpu.cmd();
    let e = Enc::new(&cb);
    e.buf(&arena, 0, 2);
    e.buf(&tokens, 0, 6);
    e.pso(&gpu.pso("k_gumbel"));
    for i in 0..rows.len() {
        let args = SampArgs {
            logits_off: (i * WIDTH) as u32,
            n: WIDTH as u32,
            seed_lo: 19,
            seed_hi: 0,
            pos: 3,
            ban_tok: bans[i],
            temp: temps[i],
            softcap: 30.0,
        };
        e.bytes(&args, 5);
        e.dispatch((WIDTH.div_ceil(256), 1, 1), 256);
    }
    e.barrier();
    for (kernel, groups, threads) in [("k_argmax1", PARTS, 1024), ("k_argmax2", 1, 32)] {
        e.pso(&gpu.pso(kernel));
        for i in 0..rows.len() {
            let args = ArgmaxArgs {
                logits_off: (i * WIDTH) as u32,
                n: WIDTH as u32,
                part_off: (score_words + i * 2 * PARTS) as u32,
                nparts: PARTS as u32,
                out_index: i as u32,
                checked: 1,
                pad: [0; 2],
            };
            e.bytes(&args, 5);
            e.dispatch((groups, 1, 1), threads);
        }
        e.barrier();
    }
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
    assert_eq!(
        cb.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cb.error()
    );
    // Completion makes both token writes visible to the CPU.
    unsafe { *tokens.contents().as_ptr().cast::<[u32; 2]>() }
}

#[test]
fn metal_sampling_validates_each_row_and_preserves_greedy_order() {
    if MTLCreateSystemDefaultDevice().is_none() {
        eprintln!("Metal sampling regression skipped: no Metal device");
        return;
    }
    // Compile the production sampling region alone so this test also runs on GPUs without tensor ops.
    let all = include_str!("../kernels.metal");
    let start = all.find("// argmax over logits").unwrap();
    let end = all[start..].find("// Adaptive LoopCD-Logits").unwrap();
    let src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{}",
        &all[start..start + end]
    );
    let gpu = Gpu::new(&src, &["k_gumbel", "k_argmax1", "k_argmax2"]);

    let mut rows = [[-20.0; WIDTH]; 2];
    let above_twenty = f32::from_bits(20.0_f32.to_bits() + 1);
    rows[0][1] = 21.0; // banned
    rows[0][2] = 20.0;
    rows[0][1025] = above_twenty;
    rows[0][2048] = above_twenty; // tie across workgroups
    rows[1][12] = 20.0;
    for temp in [0.0, 1.25e-38, 1e-6] {
        let tokens = sample_rows(&gpu, &rows, [temp, 0.5], [1, u32::MAX]);
        assert_eq!(tokens, [1025, 12], "temperature {temp}");
    }

    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        rows[0][2048] = bad;
        for (temp, ban) in [(0.0, 1), (0.5, 1), (1.25e-38, 2048), (0.5, 2048)] {
            let tokens = sample_rows(&gpu, &rows, [temp, 0.5], [ban, u32::MAX]);
            assert_eq!(
                tokens,
                [u32::MAX, 12],
                "logit {bad}, banned {ban}, temperature {temp}"
            );
        }
    }
    rows[0].fill(f32::NAN);
    let tokens = sample_rows(&gpu, &rows, [1.0, 0.5], [u32::MAX; 2]);
    assert_eq!(tokens, [u32::MAX, 12], "all-NaN row");

    // Standalone (1-row) + batched (2-row) near-zero temperature agreement with CPU `Sampler::sample_next`
    // alongside an ordinary-temperature finite control (`t-audit-engine-sampling-near-zero` & `t-audit-engine-sampling-finite`).
    let mut cpu_tie_row = [-15.0f32; WIDTH];
    cpu_tie_row[513] = 420.0;
    cpu_tie_row[1537] = 425.0; // `30 * tanh(x / 30)` saturates to 30.0 in f32 for both, while raw argmax is 1537!
    let mut finite_ctrl = [-15.0f32; WIDTH];
    finite_ctrl[77] = 25.0;

    for temp in [0.0f32, 1.0e-38, 1.25e-38, 1.0e-6] {
        let mut cpu_sampler = crate::sampler::Sampler::new(temp, 0, 1.0, 30.0, 19);
        let cpu_tok = cpu_sampler.sample_next(&cpu_tie_row).unwrap();
        assert_eq!(cpu_tok, 1537, "CPU near-zero raw argmax at temp={temp}");
        let gpu_pair = sample_rows(&gpu, &[cpu_tie_row, finite_ctrl], [temp, 0.7], [u32::MAX; 2]);
        assert_eq!(gpu_pair[0], cpu_tok, "GPU batched near-zero mismatch at temp={temp}");
        assert_eq!(gpu_pair[1], 77, "finite ordinary-temperature control disturbed at temp={temp}");
    }
}
