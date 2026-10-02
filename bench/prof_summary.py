"""Summarise a vLLM torch-profiler rank-0 trace: kernel list of the middle timed forward, per-layer tables.
Usage: python bench/prof_summary.py build/prof/{stock,ctapp} [--full]"""
import glob, gzip, json, sys, re

def load(d):
    t = json.load(gzip.open(glob.glob(f"{d}/rank0*.json.gz")[0]))
    return t["traceEvents"]

def short(n):
    n = re.sub(r"\(.*", "", n) if not n.startswith("void") else n
    return n[:90]

def steps(ev):
    ga = sorted([e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("execute_context_1(4096)")], key=lambda e: e["ts"])
    ca = sorted([e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context_1(4096)") and e["dur"] > 5000], key=lambda e: e["ts"])
    return ga, ca

def main():
    d = sys.argv[1]
    ev = load(d)
    ga, ca = steps(ev)
    print(f"{d}: gpu annotations {[round(e['dur']) for e in ga]}  cpu annotations {[round(e['dur']) for e in ca]}")
    ks = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    ga_big = [e for e in ga if e["dur"] > 10000]
    # choose the middle GPU step: pick GPU annotations of the full forward (those spanning > 10 ms), middle by ts
    full = [e for e in ga if e["dur"] > 10000]
    mid = sorted(full, key=lambda e: e["ts"])
    mid = mid[len(mid) // 2] if mid else None
    # kernels inside the middle step window: use the kernel set between boundaries of the middle step
    t0, t1 = mid["ts"], mid["ts"] + mid["dur"]
    sk = [k for k in ks if k["ts"] >= t0 - 1 and k["ts"] + k["dur"] <= t1 + 50]
    print(f"middle step window {mid['dur']:.0f} us, {len(sk)} kernels")
    prev_end = {}
    rows = []
    last_end_all = None
    for k in sk:
        st = k["tid"]
        gap = k["ts"] - last_end_all if last_end_all is not None else 0.0
        rows.append((k["ts"] - t0, k["dur"], gap, st, k["name"], k.get("args", {})))
        last_end_all = max(last_end_all or 0, k["ts"] + k["dur"])
    full = "--full" in sys.argv
    for r in rows:
        if full:
            a = r[5]
            print(f"{r[0]:9.1f} dur {r[1]:8.1f} gap {r[2]:7.1f} s{r[3]:2d} grid{a.get('grid')} blk{a.get('block')} {short(r[4])}")
    return rows, sk, t0, t1

if __name__ == "__main__":
    main()


def classify(n):
    if "tp4_reduce3" in n: return "reducer"
    if "GemmUniversalCtapp" in n: return "ctapp_gemm"
    if "nccl" in n: return "nccl_allreduce"
    if "cutlass::device_kernel<flash" in n or "FlashAttn" in n: return "attn"
    if "nvjet" in n: return "gemm"
    if "act_and_mul" in n: return "act_mul"
    if "rotary" in n: return "rope"
    if "reshape_and_cache" in n: return "kv_write"
    if "enable_if" in n or "rms_norm" in n: return "rmsnorm"
    return "other"

def layer_report(d):
    rows, sk, t0, t1 = main()
    # layer period = o_proj start to next o_proj start (o_proj = gemm with grid [2,66] preceded by 'other' fill and attn)
    idx = [i for i, r in enumerate(rows) if classify(r[4]) == "gemm" and "256x128" in r[4] and "coopA" in r[4] and rows[i - 3][4].find("FlashAttn") >= 0 or
           (classify(r[4]) == "gemm" and "256x128" in r[4] and i >= 2 and classify(rows[i - 2][4]) == "attn")]
    print("o_proj kernel indices (layer starts):", idx)
    for a, b in zip(idx[:-1], idx[1:]):
        seg = rows[a:b]
        start = seg[0][0]
        end_prev = max(r[0] + r[1] for r in seg)
        nxt = rows[b][0]
        wall = nxt - start
        # union of busy time over all streams
        iv = sorted((r[0], r[0] + r[1]) for r in seg)
        busy, cur_s, cur_e = 0.0, None, None
        for s, e in iv:
            if cur_e is None or s > cur_e:
                if cur_e is not None: busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        main_sum = sum(r[1] for r in seg if r[3] == 19)
        print(f"layer window [{start:.0f},{nxt:.0f}] wall {wall:.1f}us  main-stream kernel sum {main_sum:.1f}  union busy {busy:.1f}  idle {wall-busy:.1f}")
        for r in seg:
            c = classify(r[4])
            if c != "other":
                print(f"    {c:15s} s{r[3]} start {r[0]-start:8.1f} dur {r[1]:8.1f}  {r[4][:50] if c in ('gemm','ctapp_gemm') else ''} grid{r[5].get('grid')}")
