#!/usr/bin/env python3
"""Compile NDJSON/JSONL traces into TLA+ TraceData + .cfg for TLC conformance replay.

Design rules:
  1. Unknown events or models are ALWAYS a hard fatal error (exit 2) — never dropped.
  2. Zero slot remapping and zero offset/stride arithmetic in Python: raw `slot`,
     `generation`, and 64-bit `offset` limbs `(offset >> 32, (offset >> 20) & 0xfff,
     offset & 0xfffff)` pass straight into TLA+, where `PerKindBlockLayout.tla` /
     `BlockLayout.tla` check lowest-free-slot allocation, generations, and 64-bit
     byte offsets inside TLC.
  3. Directly consumes:
     - `src/serve/scheduler_trace.rs` JSONL (`INKY_SCHED_TRACE_OUT`, schema v1 & v2:
       `arrive` keyed by `job`, `refuse429`, `refuse_ctx`, `admit`, `build_step`
       with `mode`, `finish_step`, `disconnect`, `cancel`, `finish`);
     - per-kind `src/blocks.rs` JSONL (`scripts/blocklayout-traces.rs` and
       `cache_scheduler::tests::dump_block_layout` `INKY_BLOCK_TRACE_OUT`);
     - shared `BlockLayout` JSONL; and
     - `CacheScheduler` JSONL (`cache_scheduler::tests::dump_cache_scheduler`
       `INKY_CACHE_TRACE_OUT`).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys


def tla_bool(v: bool) -> str:
    return "TRUE" if v else "FALSE"


def load_jsonl(path: pathlib.Path) -> tuple[dict, list[tuple[int, dict]]]:
    cfg: dict | None = None
    events: list[tuple[int, dict]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rec = json.loads(s)
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{lineno}: trace record must be a JSON object")
        ev_type = rec.get("event") if "event" in rec else rec.get("op")
        if not isinstance(ev_type, str) or not ev_type.strip():
            raise ValueError(f"{path}:{lineno}: missing string 'event'/'op' field")
        ev_name = ev_type.strip()
        if ev_name == "config":
            if cfg is not None or events:
                raise ValueError(f"{path}:{lineno}: duplicate or out-of-order 'config' record")
            cfg = rec
        else:
            r = dict(rec)
            r["_op"] = ev_name
            events.append((lineno, r))
    if not events:
        raise ValueError(f"{path}: trace contains zero events")
    return (cfg or {}), events


def detect_model(cfg: dict, events: list[tuple[int, dict]]) -> str:
    if "model" in cfg:
        m = str(cfg["model"]).strip().lower()
        if m in ("per_kind_block_layout", "block_layout_per_kind"):
            return "per_kind_block_layout"
        if m in ("shared_block_layout", "block_layout_shared"):
            return "shared_block_layout"
        if m == "block_layout":
            pm = str(cfg.get("pool_mode", "per_kind")).strip().lower()
            if pm == "per_kind":
                return "per_kind_block_layout"
            if pm == "shared":
                return "shared_block_layout"
            raise ValueError(f"unknown block_layout pool_mode {pm!r}")
        if m == "cache_scheduler":
            return "cache_scheduler"
        if m == "serve_scheduler":
            return "serve_scheduler"
        raise ValueError(f"unknown trace model {m!r}")

    if "chunk" in cfg and "context" in cfg and "lanes" in cfg:
        return "serve_scheduler"
    first_op = events[0][1]["_op"]
    if first_op in ("prepare", "alloc", "begin_step", "write", "end_step"):
        return "per_kind_block_layout"
    if first_op == "arrive" and "job" in events[0][1]:
        return "serve_scheduler"
    return "cache_scheduler"


def apply_trace_sabotage(
    model: str, cfg: dict, events: list[tuple[int, dict]], sabotage: str
) -> tuple[dict, list[tuple[int, dict]]]:
    if sabotage == "none":
        return cfg, events
    cfg = copy.deepcopy(cfg)
    events = copy.deepcopy(events)

    if sabotage == "unknown-event":
        events.insert(1, (9999, {"event": "bogus_op_xyz", "_op": "bogus_op_xyz"}))
        return cfg, events

    if model == "per_kind_block_layout":
        if sabotage == "block-nonlowest-slot":
            # Move window block 1 allocation to slot 5 while slots 2..4 are free!
            slot_base = int(cfg.get("slot_base", cfg.get("base_slot_shift", 0)))
            for _, ev in events:
                if ev["_op"] == "alloc" and ev.get("kind") == "window" and int(ev.get("block", -1)) == 1:
                    ev["slot"] = 5
                    # Give slot 5 its valid u64 stride offset so Address64 passes and
                    # PerKindBlockLayout!Alloc's lowest-free-slot guard inside TLC catches it!
                    ev["offset"] = ((5 - 1) + slot_base) * (10 * 1024 * 1024)
                    return cfg, events
        elif sabotage == "block-stale-generation":
            seen = set()
            for _, ev in events:
                if ev["_op"] == "alloc":
                    key = (ev["kind"], int(ev["slot"]))
                    if key in seen:
                        ev["generation"] = 1  # stale generation -> violates GenerationMatches
                        return cfg, events
                    seen.add(key)
        raise ValueError(f"sabotage {sabotage!r} did not match per_kind_block_layout trace")

    if model == "serve_scheduler":
        if sabotage == "serve-over-inflight":
            # Drop the first refuse429 so an over-capacity arrival stays in flight -> violates InFlightBounded!
            for i, (_, ev) in enumerate(events):
                if ev["_op"] == "refuse429":
                    events.pop(i)
                    return cfg, events
        elif sabotage == "serve-over-chunk":
            # Inflate a prefill span past CHUNK -> violates StepBudgetAndModeValid!
            chunk = int(cfg.get("chunk", 2))
            for _, ev in events:
                if ev["_op"] == "build_step" and ev.get("mode") == "prefill":
                    ev["spans"][0]["tokens"] = chunk + 1
                    return cfg, events
        raise ValueError(f"sabotage {sabotage!r} did not match serve_scheduler trace")

    raise ValueError(f"sabotage {sabotage!r} unsupported for model {model!r}")


def u64_to_mib_limbs_tla(val: int) -> str:
    if val < 0:
        return "[hi |-> -1, mib |-> 0, sub |-> 0]"
    hi = val >> 32
    mib = (val >> 20) & 0xFFF
    sub = val & 0xFFFFF
    return f"[hi |-> {hi}, mib |-> {mib}, sub |-> {sub}]"


def lane_sym(raw, lane_names: list[str]) -> str:
    if isinstance(raw, int):
        while len(lane_names) <= raw:
            lane_names.append(f"l{len(lane_names)}")
        return lane_names[raw]
    s = str(raw)
    if s.isdigit():
        return lane_sym(int(s), lane_names)
    if s not in lane_names:
        lane_names.append(s)
    return s


def compile_per_kind_block_layout(
    path: pathlib.Path, cfg: dict, events: list[tuple[int, dict]], out_dir: pathlib.Path
) -> tuple[str, str, int]:
    allowed_ops = {
        "prepare", "alloc", "begin_step", "write", "end_step",
        "slide_window", "keep_start", "drop_start", "match", "finish",
    }
    lanes_cfg = cfg.get("lanes", 2)
    lane_names = [f"l{i}" for i in range(lanes_cfg)] if isinstance(lanes_cfg, int) else [str(x) for x in lanes_cfg]
    b_size = int(cfg.get("block_size", cfg.get("B", 2)))
    w_size = int(cfg.get("window_size", cfg.get("W", 2)))
    max_pos = int(cfg.get("max_pos", 6))
    max_span = int(cfg.get("max_span", 2))
    default_cap = int(cfg.get("capacity", 5))
    full_cap = int(cfg.get("full_capacity", default_cap))
    win_cap = int(cfg.get("window_capacity", default_cap))
    max_gen = int(cfg.get("max_generation", 3))
    slot_base = int(cfg.get("slot_base", cfg.get("base_slot_shift", 0)))

    for lineno, ev in events:
        op = ev["_op"]
        if op not in allowed_ops:
            raise ValueError(f"{path}:{lineno}: unknown per_kind_block_layout event {op!r}")
        if "lane" in ev:
            lane_sym(ev["lane"], lane_names)

    default_lane = lane_names[0]
    tla_records = []
    for lineno, ev in events:
        op = ev["_op"]
        ln = lane_sym(ev["lane"], lane_names) if "lane" in ev else default_lane
        tokens = int(ev.get("tokens", 1))
        kind = str(ev.get("kind", "full")).strip().lower()
        if kind not in ("full", "window"):
            raise ValueError(f"{path}:{lineno}: invalid block kind {kind!r}")
        blk = int(ev.get("block", 0))
        slot = int(ev.get("slot", 1))
        gen = int(ev.get("generation", 1))
        if op == "alloc":
            if "slot" not in ev or "generation" not in ev or "offset" not in ev:
                raise ValueError(f"{path}:{lineno}: alloc requires 'slot', 'generation', and 'offset'")
            off_tla = u64_to_mib_limbs_tla(int(ev["offset"]))
        else:
            off_tla = "[hi |-> 0, mib |-> 0, sub |-> 0]"

        has_budget = "budget" in ev
        budget_val = int(ev.get("budget", 0))
        has_pos = "pos" in ev
        pos_val = int(ev.get("pos", 0))

        tla_records.append(
            f'    [op |-> "{op}", lane |-> {ln}, tokens |-> {tokens}, '
            f'kind |-> "{kind}", block |-> {blk}, slot |-> {slot}, gen |-> {gen}, '
            f'offset |-> {off_tla}, '
            f'hasBudget |-> {tla_bool(has_budget)}, budgetVal |-> {budget_val}, '
            f'hasPos |-> {tla_bool(has_pos)}, posVal |-> {pos_val}]'
        )

    lanes_set = "{" + ", ".join(lane_names) + "}"
    data_tla = (
        "--------------------- MODULE PerKindBlockLayoutTraceData ---------------------\n"
        f"CONSTANTS {', '.join(lane_names)}\n"
        "Events == <<\n" + ",\n".join(tla_records) + "\n>>\n"
        "=============================================================================\n"
    )
    cfg_txt = (
        "SPECIFICATION TraceSpec\n"
        "CHECK_DEADLOCK FALSE\n"
        "CONSTANTS\n"
        + "".join(f" {ln} = {ln}\n" for ln in lane_names)
        + f" Lanes = {lanes_set}\n"
        f" FullCapacity = {full_cap}\n"
        f" WindowCapacity = {win_cap}\n"
        f" B = {b_size}\n"
        f" W = {w_size}\n"
        f" MaxPos = {max_pos}\n"
        f" MaxSpan = {max_span}\n"
        f" MaxGeneration = {max_gen}\n"
        f" SlotBase = {slot_base}\n"
        ' Bug = "none"\n'
        "PROPERTY Conforms\n"
        "INVARIANTS\n"
        " TypeOK\n"
        " GenerationMatches\n"
        " WindowCovered\n"
        " FullCovered\n"
        " SpanPinned\n"
        " StartRetained\n"
        " TailPrivateZero\n"
        " TableIndex\n"
        " Address64\n"
    )
    (out_dir / "PerKindBlockLayoutTraceData.tla").write_text(data_tla)
    (out_dir / "PerKindBlockLayoutTraceCheck.cfg").write_text(cfg_txt)
    return "PerKindBlockLayoutTraceCheck", "PerKindBlockLayoutTraceCheck.cfg", len(tla_records)


def compile_shared_block_layout(
    path: pathlib.Path, cfg: dict, events: list[tuple[int, dict]], out_dir: pathlib.Path
) -> tuple[str, str, int]:
    allowed_ops = {
        "prepare", "alloc", "begin_step", "write", "end_step",
        "slide_window", "keep_start", "drop_start", "match", "finish",
    }
    lanes_cfg = cfg.get("lanes", 2)
    lane_names = [f"l{i}" for i in range(lanes_cfg)] if isinstance(lanes_cfg, int) else [str(x) for x in lanes_cfg]
    b_size = int(cfg.get("block_size", cfg.get("B", 2)))
    w_size = int(cfg.get("window_size", cfg.get("W", 2)))
    max_pos = int(cfg.get("max_pos", 4))
    max_span = int(cfg.get("max_span", 2))
    capacity = int(cfg.get("capacity", 4))
    max_gen = int(cfg.get("max_generation", 3))

    for lineno, ev in events:
        op = ev["_op"]
        if op not in allowed_ops:
            raise ValueError(f"{path}:{lineno}: unknown shared_block_layout event {op!r}")
        if "lane" in ev:
            lane_sym(ev["lane"], lane_names)

    default_lane = lane_names[0]
    tla_records = []
    for lineno, ev in events:
        op = ev["_op"]
        ln = lane_sym(ev["lane"], lane_names) if "lane" in ev else default_lane
        tokens = int(ev.get("tokens", 1))
        kind = str(ev.get("kind", "full")).strip().lower()
        if kind not in ("full", "window"):
            raise ValueError(f"{path}:{lineno}: invalid block kind {kind!r}")
        blk = int(ev.get("block", 0))
        slot = int(ev.get("slot", 1))
        has_budget = "budget" in ev
        budget_val = int(ev.get("budget", 0))
        has_pos = "pos" in ev
        pos_val = int(ev.get("pos", 0))
        has_pin_count = "pin_count" in ev
        pin_count = int(ev.get("pin_count", 0))
        has_gen = "generation" in ev
        gen_val = int(ev.get("generation", 0))
        has_offset = ("offset_hi" in ev) or ("offset_lo" in ev)
        off_hi = int(ev.get("offset_hi", 0))
        off_lo = int(ev.get("offset_lo", 0))

        tla_records.append(
            f'    [op |-> "{op}", lane |-> {ln}, tokens |-> {tokens}, '
            f'kind |-> "{kind}", block |-> {blk}, slot |-> {slot}, '
            f'hasBudget |-> {tla_bool(has_budget)}, budgetVal |-> {budget_val}, '
            f'hasPos |-> {tla_bool(has_pos)}, posVal |-> {pos_val}, '
            f'hasPinCount |-> {tla_bool(has_pin_count)}, pinCount |-> {pin_count}, '
            f'hasGen |-> {tla_bool(has_gen)}, gen |-> {gen_val}, '
            f'hasOffset |-> {tla_bool(has_offset)}, offsetHi |-> {off_hi}, offsetLo |-> {off_lo}]'
        )

    lanes_set = "{" + ", ".join(lane_names) + "}"
    data_tla = (
        "------------------------ MODULE BlockLayoutTraceData ------------------------\n"
        f"CONSTANTS {', '.join(lane_names)}\n"
        "Events == <<\n" + ",\n".join(tla_records) + "\n>>\n"
        "=============================================================================\n"
    )
    cfg_txt = (
        "SPECIFICATION TraceSpec\n"
        "CHECK_DEADLOCK FALSE\n"
        "CONSTANTS\n"
        + "".join(f" {ln} = {ln}\n" for ln in lane_names)
        + f" Lanes = {lanes_set}\n"
        f" Capacity = {capacity}\n"
        f" B = {b_size}\n"
        f" W = {w_size}\n"
        f" MaxPos = {max_pos}\n"
        f" MaxSpan = {max_span}\n"
        f" MaxGeneration = {max_gen}\n"
        ' Bug = "none"\n'
        "PROPERTY Conforms\n"
        "INVARIANTS\n"
        " TypeOK\n"
        " GenerationMatches\n"
        " WindowCovered\n"
        " FullCovered\n"
        " SpanPinned\n"
        " StartRetained\n"
        " TailPrivateZero\n"
        " TableIndex\n"
        " Address64\n"
    )
    (out_dir / "BlockLayoutTraceData.tla").write_text(data_tla)
    (out_dir / "BlockLayoutTraceCheck.cfg").write_text(cfg_txt)
    return "BlockLayoutTraceCheck", "BlockLayoutTraceCheck.cfg", len(tla_records)


def compile_serve_scheduler(
    path: pathlib.Path, cfg: dict, events: list[tuple[int, dict]], out_dir: pathlib.Path
) -> tuple[str, str, int]:
    allowed_ops = {
        "arrive", "refuse429", "refuse_ctx", "admit",
        "build_step", "finish_step", "disconnect", "cancel", "finish",
    }
    num_lanes = int(cfg.get("lanes", 2))
    max_queue = int(cfg.get("queue", 1))
    chunk = int(cfg.get("chunk", 2))
    max_ctx = int(cfg.get("context", 32))

    jobs: set[int] = set()
    for lineno, ev in events:
        op = ev["_op"]
        if op not in allowed_ops:
            raise ValueError(f"{path}:{lineno}: unknown serve_scheduler event {op!r}")
        if "job" in ev:
            jobs.add(int(ev["job"]))
        if op == "build_step":
            for sp in ev.get("spans", []):
                jobs.add(int(sp["job"]))

    if not jobs:
        raise ValueError(f"{path}: serve_scheduler trace has no jobs")

    tla_records = []
    for lineno, ev in events:
        op = ev["_op"]
        job = int(ev.get("job", 1))
        prompt = int(ev.get("prompt", 1))
        max_len = int(ev.get("max", 1))
        lane = int(ev.get("lane", 0))
        cached = int(ev.get("cached", 0))
        src_kind = str(ev.get("source", "any"))
        from_lane = int(ev.get("from", lane))
        mode = str(ev.get("mode", "none"))
        reason = str(ev.get("reason", "length"))
        generated = int(ev.get("generated", 0))

        span_items = []
        if op == "build_step":
            for sp in ev.get("spans", []):
                span_items.append(
                    f'[job |-> {int(sp["job"])}, lane |-> {int(sp["lane"])}, '
                    f'pos |-> {int(sp["pos"])}, tokens |-> {int(sp["tokens"])}]'
                )
        spans_set = "{" + ", ".join(span_items) + "}"

        tla_records.append(
            f'    [op |-> "{op}", job |-> {job}, prompt |-> {prompt}, max |-> {max_len}, '
            f'lane |-> {lane}, cached |-> {cached}, srcKind |-> "{src_kind}", fromLane |-> {from_lane}, '
            f'mode |-> "{mode}", spans |-> {spans_set}, reason |-> "{reason}", generated |-> {generated}]'
        )

    jobs_set = "{" + ", ".join(str(j) for j in sorted(jobs)) + "}"
    data_tla = (
        "----------------------- MODULE ServeSchedulerTraceData -----------------------\n"
        "Events == <<\n" + ",\n".join(tla_records) + "\n>>\n"
        "=============================================================================\n"
    )
    cfg_txt = (
        "SPECIFICATION TraceSpec\n"
        "CHECK_DEADLOCK FALSE\n"
        "CONSTANTS\n"
        f" Jobs = {jobs_set}\n"
        f" NumLanes = {num_lanes}\n"
        f" MaxQueue = {max_queue}\n"
        f" ChunkBudget = {chunk}\n"
        f" MaxCtx = {max_ctx}\n"
        "PROPERTY Conforms\n"
        "INVARIANTS\n"
        " TypeOK\n"
        " InFlightBounded\n"
        " StepBudgetAndModeValid\n"
        " ActiveLanesConsistent\n"
    )
    (out_dir / "ServeSchedulerTraceData.tla").write_text(data_tla)
    (out_dir / "ServeSchedulerTraceCheck.cfg").write_text(cfg_txt)
    return "ServeSchedulerTraceCheck", "ServeSchedulerTraceCheck.cfg", len(tla_records)


def compile_cache_scheduler(
    path: pathlib.Path, cfg: dict, events: list[tuple[int, dict]], out_dir: pathlib.Path
) -> tuple[str, str, int]:
    allowed_ops = {
        "arrive", "refuse429", "admit", "begin_restore", "restore_block",
        "finish_restore", "evict_mem", "start_writeback", "complete_step_io",
        "reclaim_disk", "drop_unreferenced", "slide_window", "build_step",
        "gpu_execute", "finish_step", "finish", "cancel",
    }
    lanes_cfg = cfg.get("lanes", 3)
    lane_names = [chr(ord("a") + i) for i in range(lanes_cfg)] if isinstance(lanes_cfg, int) else [str(x) for x in lanes_cfg]
    prefixes_cfg = cfg.get("prefixes", 2)
    prefix_names = [f"p{i + 1}" for i in range(prefixes_cfg)] if isinstance(prefixes_cfg, int) else [str(x) for x in prefixes_cfg]

    b_size = int(cfg.get("block_size", 2))
    w_size = int(cfg.get("window_size", 2))
    max_pos = int(cfg.get("max_pos", 4))
    max_span = int(cfg.get("max_span", 2))
    step_budget = int(cfg.get("step_token_budget", 2))
    mem_budget = int(cfg.get("mem_budget", 4))
    disk_budget = int(cfg.get("disk_budget", 4))
    max_queue = int(cfg.get("max_queue", 1))
    max_bypass = int(cfg.get("max_bypass", 1))
    req_lengths = sorted({int(x) for x in cfg.get("request_lengths", [2, 4])})

    for lineno, ev in events:
        op = ev["_op"]
        if op not in allowed_ops:
            raise ValueError(f"{path}:{lineno}: unknown cache_scheduler event {op!r}")
        if "lane" in ev:
            lane_sym(ev["lane"], lane_names)
        if "prefix" in ev:
            s = str(ev["prefix"])
            if s not in prefix_names:
                prefix_names.append(s)
        if op == "build_step":
            for sp in ev.get("spans", []):
                lane_sym(sp["lane"], lane_names)
        if "target" in ev:
            req_lengths.append(int(ev["target"]))
    req_lengths = sorted(set(req_lengths))

    default_lane = lane_names[0]
    default_prefix = prefix_names[0]
    default_target = req_lengths[0]
    zero_spans = " @@ ".join(f"{ln} :> 0" for ln in lane_names)

    tla_records = []
    for lineno, ev in events:
        op = ev["_op"]
        ln = lane_sym(ev["lane"], lane_names) if "lane" in ev else default_lane
        pfx = str(ev.get("prefix", default_prefix))
        tgt = int(ev.get("target", default_target))
        kind = str(ev.get("kind", "full")).strip().lower()
        if kind not in ("full", "window"):
            raise ValueError(f"{path}:{lineno}: invalid block kind {kind!r}")
        blk = int(ev.get("block", 0))

        has_state = "state" in ev
        state_val = str(ev.get("state", "Idle"))
        has_wait = "wait_ticks" in ev
        wait_val = int(ev.get("wait_ticks", 0))
        has_pos = "pos" in ev
        pos_val = int(ev.get("pos", 0))
        has_res = "reservation" in ev
        res_val = int(ev.get("reservation", 0))
        has_block = ("block" in ev) and ("prefix" in ev) and ("kind" in ev)
        has_tier = "tier" in ev and has_block
        tier_val = str(ev.get("tier", "none"))
        has_pin_count = "pin_count" in ev
        pin_count = int(ev.get("pin_count", 0))

        chosen_set = "{}"
        spans_fn = f"({zero_spans})"
        if op == "build_step":
            span_map = {l: 0 for l in lane_names}
            chosen_list = []
            for sp in ev.get("spans", []):
                sln = lane_sym(sp["lane"], lane_names)
                tok = int(sp.get("tokens", 1))
                span_map[sln] = tok
                if tok > 0:
                    chosen_list.append(sln)
            chosen_set = "{" + ", ".join(chosen_list) + "}"
            spans_fn = "(" + " @@ ".join(f"{l} :> {span_map[l]}" for l in lane_names) + ")"

        tla_records.append(
            f'    [op |-> "{op}", lane |-> {ln}, prefix |-> {pfx}, target |-> {tgt}, '
            f'kind |-> "{kind}", block |-> {blk}, '
            f'hasState |-> {tla_bool(has_state)}, stateVal |-> "{state_val}", '
            f'hasWait |-> {tla_bool(has_wait)}, waitVal |-> {wait_val}, '
            f'hasPos |-> {tla_bool(has_pos)}, posVal |-> {pos_val}, '
            f'hasRes |-> {tla_bool(has_res)}, resVal |-> {res_val}, '
            f'hasBlock |-> {tla_bool(has_block)}, '
            f'hasTier |-> {tla_bool(has_tier)}, tierVal |-> "{tier_val}", '
            f'hasPinCount |-> {tla_bool(has_pin_count)}, pinCount |-> {pin_count}, '
            f'chosen |-> {chosen_set}, spans |-> {spans_fn}]'
        )

    all_consts = lane_names + [p for p in prefix_names if p not in lane_names]
    lanes_set = "{" + ", ".join(lane_names) + "}"
    prefixes_set = "{" + ", ".join(prefix_names) + "}"
    req_set = "{" + ", ".join(str(r) for r in req_lengths) + "}"

    data_tla = (
        "---------------------- MODULE CacheSchedulerTraceData ----------------------\n"
        "EXTENDS TLC\n"
        f"CONSTANTS {', '.join(all_consts)}\n"
        "Events == <<\n" + ",\n".join(tla_records) + "\n>>\n"
        "=============================================================================\n"
    )
    cfg_txt = (
        "SPECIFICATION TraceSpec\n"
        "CHECK_DEADLOCK FALSE\n"
        "CONSTANTS\n"
        + "".join(f" {c} = {c}\n" for c in all_consts)
        + f" Lanes = {lanes_set}\n"
        f" Prefixes = {prefixes_set}\n"
        f" RequestLengths = {req_set}\n"
        f" BlockSize = {b_size}\n"
        f" WindowSize = {w_size}\n"
        f" MaxPos = {max_pos}\n"
        f" MaxSpan = {max_span}\n"
        f" StepTokenBudget = {step_budget}\n"
        f" MemBudget = {mem_budget}\n"
        f" DiskBudget = {disk_budget}\n"
        f" MaxQueue = {max_queue}\n"
        f" MaxBypass = {max_bypass}\n"
        ' Bug = "none"\n'
        "PROPERTIES\n"
        " Conforms\n"
        " ActionSealedImmutable\n"
        " ActionNoQueuedRefusal\n"
        " ActionLazyColdWriteback\n"
        " ActionDecodePriority\n"
        "INVARIANTS\n"
        " TypeOK\n"
        " StepBlocksInMemoryAndOwned\n"
        " SealedValid\n"
        " UnsealedPrivateInvisibleZero\n"
        " LaneHistoryCovered\n"
        " RefCountMatchesReferrers\n"
        " StartPointsKept\n"
        " BudgetsAndReservations\n"
        " ColdBlocksSpillable\n"
        " BypassBoundHolds\n"
        " StepBudgetHolds\n"
        " QueueBounded\n"
    )
    (out_dir / "CacheSchedulerTraceData.tla").write_text(data_tla)
    (out_dir / "CacheSchedulerTraceCheck.cfg").write_text(cfg_txt)
    return "CacheSchedulerTraceCheck", "CacheSchedulerTraceCheck.cfg", len(tla_records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="auto",
        choices=["auto", "per_kind_block_layout", "shared_block_layout", "serve_scheduler", "cache_scheduler"],
    )
    ap.add_argument("--trace", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--sabotage", default="none")
    args = ap.parse_args()

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cfg, events = load_jsonl(args.trace)
        model = detect_model(cfg, events) if args.model == "auto" else args.model
        cfg, events = apply_trace_sabotage(model, cfg, events, args.sabotage)
        if model == "per_kind_block_layout":
            spec_mod, cfg_file, count = compile_per_kind_block_layout(args.trace, cfg, events, args.out_dir)
        elif model == "shared_block_layout":
            spec_mod, cfg_file, count = compile_shared_block_layout(args.trace, cfg, events, args.out_dir)
        elif model == "serve_scheduler":
            spec_mod, cfg_file, count = compile_serve_scheduler(args.trace, cfg, events, args.out_dir)
        else:
            spec_mod, cfg_file, count = compile_cache_scheduler(args.trace, cfg, events, args.out_dir)
        print(f"{model} {spec_mod} {cfg_file} {count}")
        return 0
    except Exception as exc:
        print(f"TRACE_COMPILE_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
