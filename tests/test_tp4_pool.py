"""CtappBoundaryPool test: varying M, per-call weights, one pool.   python tests/test_tp4_pool.py --world 2 --port 29611 [--protocol v5]

--protocol v5 (world 4): the pool must serve the whole sequence from ONE instance (max_M 8192)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch  # noqa: E402

from ctapp import tp4  # noqa: E402
from tests.test_tp4 import check, rel, sync, FAILS  # noqa: E402


def _run(rank, world, dev, gn, protocol="v3"):
    Kr, N2r = tp4.K_FFN // world, tp4.N2 // world
    pool = tp4.CtappBoundaryPool(Kr, N2r, rank, world, gn, dev, min_M=512, protocol=protocol, max_M=8192)
    seen = set()
    check(rank, "get(200) -> None", pool.get(200) is None)
    check(rank, "get(256) with min_M=512 -> None", pool.get(256) is None)
    wsets = {}

    def weights(seed):
        if seed not in wsets:
            Wd, gamma, Wq = tp4.make_weights(dev, seed=seed)
            Wd_r, _, Wqf_r = tp4.shard_weights(Wd, Wq, gamma, rank, world)
            wsets[seed] = (Wd, gamma, Wq, Wd_r, Wqf_r)
        return wsets[seed]

    def run(M, seed, tag, n=1, verify=True):
        bd = pool.get(M)
        seen.add(id(bd))
        Wd, gamma, Wq, Wd_r, Wqf_r = weights(seed)
        h, resid = tp4.make_inputs(M, dev, seed=seed + 100)
        h_r = h[:, rank * Kr:(rank + 1) * Kr].contiguous()
        for _ in range(n):
            y = bd.forward(h_r, resid, Wd_r, Wqf_r)
        if not verify:
            return bd
        torch.cuda.synchronize(dev)
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        ref_r = out_ref[:, rank * N2r:(rank + 1) * N2r]
        cx, co = rel(bd.x(), x_ref), rel(y, ref_r)
        check(rank, f"{tag} M={M} seed={seed}: x rel", cx < 1e-2, f"{cx:.2e}")
        check(rank, f"{tag} M={M} seed={seed}: out rel", co < 2e-2, f"{co:.2e}")
        sync(dev)
        return bd

    seq = [(4096, 3), (1024, 2), (4096, 2), (8192, 1), (4096, 1)]
    seed = 0
    for M, n in seq:
        for i in range(n):
            seed += 1
            run(M, seed, f"seq fwd {i + 1}/{n}")
    # back-to-back burst: alternating M and weights, no sync between forwards
    items = []
    for i in range(20):
        M, s = (4096, 11) if i % 2 == 0 else (1024, 12)
        bd = pool.get(M)
        seen.add(id(bd))
        Wd, gamma, Wq, Wd_r, Wqf_r = weights(s)
        h, resid = tp4.make_inputs(M, dev, seed=s + 100)
        items.append((bd, M, s, h[:, rank * Kr:(rank + 1) * Kr].contiguous(), resid, h))
    outs = []
    for bd, M, s, h_r, resid, h in items:
        _, _, _, Wd_r, Wqf_r = wsets[s]
        y = bd.forward(h_r, resid, Wd_r, Wqf_r)
        outs.append((y.clone(), bd.x().clone()))   # stream-ordered copies (v5: one instance, buffers reused by every M)
    torch.cuda.synchronize(dev)
    # the last forward of each M (items[-2:]) is checked
    for (bd, M, s, h_r, resid, h), (y, x) in zip(items[-2:], outs[-2:]):
        Wd, gamma, Wq, _, _ = wsets[s]
        x_ref, out_ref = tp4.one_gpu_reference(h, Wd, resid, gamma, Wq)
        ref_r = out_ref[:, rank * N2r:(rank + 1) * N2r]
        cx, co = rel(x, x_ref), rel(y, ref_r)
        check(rank, f"burst20 M={M}: x rel", cx < 1e-2, f"{cx:.2e}")
        check(rank, f"burst20 M={M}: out rel", co < 2e-2, f"{co:.2e}")
    sync(dev)
    if protocol == "v5":
        check(rank, "v5 pool: one instance for every M", len(seen) == 1, f"{len(seen)} distinct instances")
    return len(FAILS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--protocol", default="v3", choices=["v3", "v5"])
    a = ap.parse_args()
    res, bad = tp4.launch(_run, a.world, a.port, a.protocol)
    ok = (not bad) and all(r == 0 for r in res)
    print("ALL PASS" if ok else f"FAILED (results {res}, process error: {bad})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
