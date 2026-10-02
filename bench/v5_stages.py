"""Phase-6 S2 stage breakdown of the v5 boundary (TP4): producer / reducer / consumer alone and concurrent.

    python bench/v5_stages.py --tokens 1024,4096 --configs 4:1024:4:1,4:1024:4:2 --json build/p6_s2_stages.json

config = R:threads:blocks:down_raster[:variant[:rows[:down_swizzle]]] (GEMMs on SM_count - R SMs, reducer = blocks CTAs of
`threads`; variant 2 = the S3 role reducer stand-alone, threads 384, `rows` rows per lane).
Stages (per-step median with barrier + sync before each call / back-to-back mean, max over ranks, ms):
  down0_s{sms}_r{1,2}: tp4_down mode 0 (stock epilogue, local D) on sms SMs, raster 1 / 2
  down7:          tp4_down mode 7 alone (scatter into the 4 owners' inboxes + value flags), this config's sms / raster
  down7_nodrain:  (--nodrain 1) the same without the per-tile cp.async.bulk.wait_group 0 before the flag (incorrect; timing only)
  down7_local:    (--nodrain 1) mode 7 with all 4 destinations = this rank's own inbox slot / tile flags (no NVLink stores)
  red_alone:      tp4_step + reducer v5 alone, tile flags pre-set (no waiting on the producer)
  down7_red:      tp4_step + reducer (2nd stream) || down mode 7 (the producer stage of the chain, real flags)
  qkv0 / qkv5:    QKV GEMM mode 0 / mode 5 with panel flags pre-set, this config's sms
  chain:          the full v5 forward (eager), real flags

Phase 6 S3 sweeps (--sweep, instead of the stage list above):
  down:  producer swizzle sweep: tp4_down mode 0 and mode 7, raster 1, swizzle in --swizzles, sms = SM_count - 4, at --tokens
  qkv:   consumer tile sweep, mode 0 (no waits, no communication): tile 128 x raster {1,2} x --swizzles x sms {128, 132} and tile
         256 r2 s1 / r1 s2, at K = 8192, N = --qkv-n (comma list, default 2560,14336)
  trace: per --fusion-configs entry fusion[+steal|+st2|+st3]:tile:R:rows[:down_swizzle[:shape[:pdl_trigger[:down_tail]]]] the eager chain b2b / per-step, then --trace-iters traced
         eager forwards (CTAPP_TRACE stamps, us from the first producer CTA entry, median over iterations, rank 0 and max over ranks)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from ctapp import tp4  # noqa: E402
from ctapp.tp4 import DTYPE, N1  # noqa: E402

BIG = 1 << 30


def worker(rank, world, dev, gn, tokens, configs, iters, warmup, nodrain, sweep, sw):
    if sweep == "down":
        return sweep_down(rank, world, dev, gn, tokens, iters, warmup, sw)
    if sweep == "qkv":
        return sweep_qkv(rank, world, dev, gn, tokens, iters, warmup, sw)
    if sweep == "trace":
        return sweep_trace(rank, world, dev, gn, tokens, iters, warmup, sw)
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    Wd, gamma, Wq = tp4.make_weights(dev)
    Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    del Wd, Wq
    out = {}
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count

    def T(fn):
        return tp4.time_step(fn, dev, iters, warmup)

    def report(M, key, v):
        out.setdefault(str(M), {})[key] = v
        if rank == 0:
            print(f"M={M:6d} {key:34s} per-step {v[0]:.3f}  b2b {v[1]:.3f} ms", flush=True)

    for M in tokens:
        h, resid = tp4.make_inputs(M, dev)
        h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
        del h
        e = None
        dz = torch.empty(M, N1, dtype=DTYPE, device=dev)
        for cfg in configs:
            R, thr, blk, ras = cfg[:4]
            var = cfg[4] if len(cfg) > 4 else 0
            rows = cfg[5] if len(cfg) > 5 else 4
            dsw = cfg[6] if len(cfg) > 6 else 1
            bd = tp4.CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol="v5", R=R, max_M=M,
                                   down_raster=ras, red_threads=thr, red_blocks=blk, red_variant=var, role_rows=rows, down_swizzle=dsw)
            bd.set_weights(Wd_r, Wqf_r)
            e = bd.ext
            tag = f"R{R}_t{thr}_b{blk}_r{ras}" + (f"_v{var}" if var else "") + (f"_rows{rows}" if var == 2 else "") + (f"_s{dsw}" if dsw != 1 else "")
            sms = bd.sms
            for s_ in sorted({sms, nsm}):
                for r_ in (1, 2):
                    key = f"down0_s{s_}_r{r_}"
                    if key not in out.get(str(M), {}):
                        report(M, key, T(lambda: e.tp4_down(h_r, Wd_r, dz, r_, 1, 0, rank, world, 0, 0, bd.tile_flags, 0, 0, None, 0, 0, sms=s_)))
            report(M, f"{tag} down7", T(lambda: bd._down_v5(h_r, Wd_r, M)))
            if nodrain:
                e.tp4_set_debug(32)   # kDbgNoStoreDrain: flag without waiting for the tile's TMA stores (timing only)
                report(M, f"{tag} down7_nodrain", T(lambda: bd._down_v5(h_r, Wd_r, M)))
                e.tp4_set_debug(0)
                # all 4 destinations = my own inbox slot + my own tile flags: same epilogue machinery, no NVLink stores
                loc_dst, loc_tf = [bd._dst_ptrs[rank]] * 4, [bd._peer_tf[rank]] * 4
                report(M, f"{tag} down7_local", T(lambda: e.tp4_down(
                    h_r, Wd_r, bd._dview[:M], bd.down_raster, 1, 7, rank, world, 0, 0, bd.tile_flags, 0, 0, None, 0, 0, sms=sms,
                    epoch_dev=bd.epoch_dev, dst_ptrs=loc_dst, dst_rows=bd.ppo * 128, peer_tile_flags=loc_tf, panels_per_owner=bd.ppo)))
                torch.cuda.synchronize(dev); tp4.barrier()
            tp4.barrier()
            # reducer alone: every tile flag pre-set
            torch.cuda.synchronize(dev); tp4.barrier()
            bd.tile_flags.fill_(BIG)
            torch.cuda.synchronize(dev); tp4.barrier()

            def red_alone():
                e.tp4_step(bd.epoch_dev, bd.rowss_local, bd.panel_done, M)
                bd._reduce_v5(M, resid, 0)
            report(M, f"{tag} red_alone", T(red_alone))
            torch.cuda.synchronize(dev); tp4.barrier()
            bd.tile_flags.zero_()
            torch.cuda.synchronize(dev); tp4.barrier()

            def down7_red():
                cur = torch.cuda.current_stream()
                e.tp4_step(bd.epoch_dev, bd.rowss_local, bd.panel_done, M)
                bd.ev_start.record(cur)
                bd.s_red.wait_event(bd.ev_start)
                with torch.cuda.stream(bd.s_red):
                    bd._reduce_v5(M, resid, 0)
                    bd.ev_red.record(bd.s_red)
                bd._down_v5(h_r, Wd_r, M)
                cur.wait_event(bd.ev_red)
            report(M, f"{tag} down7_red", T(down7_red))
            # consumer alone: panel flags pre-set
            torch.cuda.synchronize(dev); tp4.barrier()
            bd.panel_flag.fill_(BIG)
            torch.cuda.synchronize(dev); tp4.barrier()
            report(M, f"{tag} qkv5", T(lambda: bd._qkv_v5(Wqf_r, M, 0)))
            report(M, f"{tag} qkv0", T(lambda: e.tp4_qkv(bd.xbuf[:M], Wqf_r, bd._out_full[:M], bd.qkv_raster, bd.qkv_swizzle, 0, 0,
                                                       bd.panel_flag, 1, bd.rowss[0], sms=sms)))
            torch.cuda.synchronize(dev); tp4.barrier()
            bd.panel_flag.zero_()
            torch.cuda.synchronize(dev); tp4.barrier()
            report(M, f"{tag} chain", T(lambda: bd.forward(h_r, resid)))
            torch.cuda.synchronize(dev); tp4.barrier()
            del bd
            torch.cuda.empty_cache()
        if "qkv0_s132" not in out.get(str(M), {}):
            bq = torch.empty(M, N2r, dtype=DTYPE, device=dev)
            xq = torch.randn(M, N1, device=dev).to(DTYPE)
            rq = torch.ones(M, dtype=torch.float32, device=dev)
            pf = torch.zeros(M // 128, dtype=torch.int32, device=dev)
            report(M, "qkv0_s132", T(lambda: e.tp4_qkv(xq, Wqf_r, bq, 2, 1, 0, 0, pf, 1, rq, sms=nsm)))
            del bq, xq, rq, pf
        del h_r, resid, dz
    return out if rank == 0 else {}


def sweep_down(rank, world, dev, gn, tokens, iters, warmup, sw):
    """Producer swizzle sweep: mode 0 (local D) and mode 7 (scatter + flags) at raster 1, sms = SM_count - 4."""
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    Wd, gamma, Wq = tp4.make_weights(dev)
    Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
    del Wd, Wq
    out = {}
    for M in tokens:
        h, resid = tp4.make_inputs(M, dev)
        h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
        del h
        dz = torch.empty(M, N1, dtype=DTYPE, device=dev)
        bd = tp4.CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol="v5", R=4, max_M=M)
        bd.set_weights(Wd_r, Wqf_r)
        e, sms = bd.ext, bd.sms
        for swz, tl in [(s_, 0) for s_ in sw["swizzles"]] + [(1, t_) for t_ in sw["tails"]]:
            v0 = tp4.time_step(lambda: e.tp4_down(h_r, Wd_r, dz, 1, swz, 0, rank, world, 0, 0, bd.tile_flags, 0, 0, None, 0, 0, sms=sms,
                                                  tail_cols=tl), dev, iters, warmup)
            bd.down_swizzle, bd.down_tail = swz, tl
            v7 = tp4.time_step(lambda: bd._down_v5(h_r, Wd_r, M), dev, iters, warmup)
            key = f"s{swz}" + (f"_tail{tl}" if tl else "")
            out.setdefault(str(M), {})[key] = {"mode0": v0, "mode7": v7}
            if rank == 0:
                print(f"M={M:6d} down r1 {key:9s} sms{sms}: mode0 per-step {v0[0]:.3f} b2b {v0[1]:.3f} | mode7 per-step {v7[0]:.3f} "
                      f"b2b {v7[1]:.3f} ms (mode7/mode0 b2b {v7[1] / v0[1]:.3f})", flush=True)
            torch.cuda.synchronize(dev); tp4.barrier()
        del bd, h_r, resid, dz
        torch.cuda.empty_cache()
    return out if rank == 0 else {}


def sweep_qkv(rank, world, dev, gn, tokens, iters, warmup, sw):
    """Consumer GEMM (mode 0: no waits) tile / raster / swizzle / sms sweep at K = 8192 (every rank on its own GPU)."""
    from ctapp.ext import load_tp4
    e = load_tp4()
    nsm = torch.cuda.get_device_properties(dev).multi_processor_count
    out = {}
    g = torch.Generator(device=dev).manual_seed(0)
    for N in sw["qkv_n"]:
        W = (torch.randn(N, N1, device=dev, generator=g) * N1 ** -0.5).to(DTYPE)
        for M in tokens:
            X = torch.randn(M, N1, device=dev, generator=g).to(DTYPE)
            Y = torch.empty(M, N, dtype=DTYPE, device=dev)
            rq = torch.full((M,), float(N1), dtype=torch.float32, device=dev)
            pf = torch.zeros(M // 128, dtype=torch.int32, device=dev)
            cfgs = [(128, r, s_, sm) for r in (1, 2) for s_ in sw["swizzles"] for sm in (nsm - 4, nsm)]
            cfgs += [(256, 2, 1, nsm - 4), (256, 2, 1, nsm), (256, 1, 2, nsm - 4), (256, 1, 2, nsm)]
            ref = torch.mm(X, W.t())
            for tile, r, s_, sm in cfgs:
                if N % tile:
                    continue
                v = tp4.time_step(lambda: e.tp4_qkv(X, W, Y, r, s_, 0, 0, pf, 1, rq, sms=sm, tile=tile), dev, iters, warmup)
                err = ((Y.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
                key = f"N{N}_t{tile}_r{r}_s{s_}_sms{sm}"
                out.setdefault(str(M), {})[key] = {"per_step": v[0], "b2b": v[1], "rel_vs_mm": err}
                if rank == 0:
                    print(f"M={M:6d} qkv0 {key:28s} per-step {v[0]:.3f}  b2b {v[1]:.3f} ms  (rel vs torch.mm {err:.1e})", flush=True)
            mm = tp4.time_step(lambda: torch.mm(X, W.t()), dev, iters, warmup)
            out[str(M)][f"N{N}_torch_mm"] = {"per_step": mm[0], "b2b": mm[1]}
            if rank == 0:
                print(f"M={M:6d} qkv  N{N}_torch_mm (cuBLAS, no RMS scale)  per-step {mm[0]:.3f}  b2b {mm[1]:.3f} ms", flush=True)
            del X, Y, rq, pf, ref
        del W
        torch.cuda.empty_cache()
    return out if rank == 0 else {}


def sweep_trace(rank, world, dev, gn, tokens, iters, warmup, sw):
    """Chain timing + CTAPP_TRACE stage stamps per fusion config."""
    import statistics
    import torch.distributed as dist
    SH = {"B": (tp4.K_FFN, tp4.N2), "A": (8192, 57344)}
    out = {}
    cache = {}
    for M in tokens:
        for fc in sw["fusion_configs"]:
            parts = fc.split(":")
            fu, tile, R, rows = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            fu, _, suf = fu.partition("+")   # "+steal": work stealing; "+st2" / "+st3": role reducer cp.async stages
            steal, stg = suf == "steal", int(suf[2:]) if suf in ("st2", "st3") else 0
            dsw = int(parts[4]) if len(parts) > 4 else 1
            shape = parts[5] if len(parts) > 5 else "B"
            ptrig = int(parts[6]) if len(parts) > 6 and parts[6] != "" else None
            tail = (parts[7] if parts[7] == "auto" else int(parts[7])) if len(parts) > 7 else 0
            K, N2f = SH[shape]
            Kr, N2r = K // world, N2f // world
            if cache.get("shape") != shape:
                cache.clear()
                Wd, gamma, Wq = tp4.make_weights(dev, K=K, N2_=N2f)
                cache["w"] = tp4.shard_weights(Wd, Wq, gamma, rank, world)
                cache["shape"] = shape
                del Wd, Wq
                torch.cuda.empty_cache()
            Wd_r, _, Wqf_r = cache["w"]
            if cache.get("M") != (M, shape):
                h, resid = tp4.make_inputs(M, dev, K=K)
                cache["in"] = (h[:, rank * Kr:(rank + 1) * Kr].contiguous(), resid)
                cache["M"] = (M, shape)
                del h
            h_r, resid = cache["in"]
            qr, qs = (2, 1) if shape == "B" else (1, 2)
            if shape == "B" and tile == 128:
                qr, qs = sw["qkv128"]
            bd = tp4.CtappBoundary(M, Kr, N2r, rank, world, gn, dev, resid=True, protocol="v5", R=R, max_M=M, fusion=fu, qkv_tile=tile,
                                   role_rows=rows, down_swizzle=dsw, qkv_raster=qr, qkv_swizzle=qs, pdl_trigger=ptrig, steal=steal, role_stages=stg,
                                   down_tail=tail)
            bd.set_weights(Wd_r, Wqf_r)
            key = f"{shape}_{fu}{'+' + suf if suf else ''}_t{tile}_R{R}_rows{rows}_s{dsw}" + (f"_trig{ptrig}" if ptrig is not None else "") + (f"_tail{tail}" if tail else "")
            v = tp4.time_step(lambda: bd.forward(h_r, resid), dev, iters, warmup)
            torch.cuda.synchronize(dev); tp4.barrier()
            stamps = []
            for _ in range(sw["trace_iters"]):
                bd.trace = True
                bd.forward(h_r, resid)
                stamps.append(bd.trace_summary())
                bd.trace = False
                tp4.barrier()
            keys = [k for k in stamps[0] if isinstance(stamps[0][k], float)]
            med = {k: statistics.median(s_[k] for s_ in stamps if k in s_) for k in keys}
            t = torch.tensor([med[k] for k in keys], dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=tp4.GLOO)
            mx = dict(zip(keys, t.tolist()))
            info = {k: stamps[0][k] for k in stamps[0] if not isinstance(stamps[0][k], float)}
            out.setdefault(str(M), {})[key] = {"per_step": v[0], "b2b": v[1], "rank0": med, "max_over_ranks": mx, "info": info}
            if rank == 0:
                print(f"M={M:6d} {key:30s} chain per-step {v[0]:.3f}  b2b {v[1]:.3f} ms", flush=True)
                print("    stamps rank0 (us): " + ", ".join(f"{k} {med[k]:.1f}" for k in keys), flush=True)
                print("    stamps max ranks:  " + ", ".join(f"{k} {mx[k]:.1f}" for k in keys), flush=True)
                print(f"    info: {info}", flush=True)
            del bd
            torch.cuda.synchronize(dev); tp4.barrier()
            torch.cuda.empty_cache()
    return out if rank == 0 else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--tokens", default="1024,4096")
    ap.add_argument("--configs", default="4:1024:4:1,4:1024:4:2")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--port", type=int, default=29960)
    ap.add_argument("--json", default="build/p6_s2_stages.json")
    ap.add_argument("--nodrain", type=int, default=0, help="also time down7_nodrain (dbg 32) and down7_local")
    ap.add_argument("--sweep", default=None, choices=["down", "qkv", "trace"])
    ap.add_argument("--swizzles", default="1,2,4,8")
    ap.add_argument("--tails", default="", help="sweep down: extra swizzle-1 runs with these producer tail_cols (comma list)")
    ap.add_argument("--qkv-n", default="2560,14336")
    ap.add_argument("--qkv128", default="2:2", help="QKV raster:swizzle of the 128x128 tile at shape B (trace sweep)")
    ap.add_argument("--fusion-configs", default="none:256:4:4,pdl:256:4:4,role:256:4:4")
    ap.add_argument("--trace-iters", type=int, default=10)
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    configs = [tuple(int(v) for v in c.split(":")) for c in a.configs.split(",")]
    sw = {"swizzles": [int(v) for v in a.swizzles.split(",")], "qkv_n": [int(v) for v in a.qkv_n.split(",")],
          "qkv128": tuple(int(v) for v in a.qkv128.split(":")), "tails": [int(t) for t in a.tails.split(",") if t], "fusion_configs": a.fusion_configs.split(","), "trace_iters": a.trace_iters}
    res, bad = tp4.launch(worker, a.world, a.port, tokens, configs, a.iters, a.warmup, a.nodrain, a.sweep, sw)
    if bad:
        print("stages failed")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"world": a.world, "iters": a.iters, "warmup": a.warmup, "configs": a.configs, "results": res[0]}, f, indent=1)
    print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
