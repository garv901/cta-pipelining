# Phase 2b: per-CTA %globaltimer timeline of the CTAPP pipeline. Run:
#   timeout 3000 env CUDA_VISIBLE_DEVICES=3,0 TMPDIR=/mnt/storage/garv901/cta-pipelining/build python3 -u bench/timeline.py [--knob NAME]
# (torch dev 0 = physical GPU 3 = producer A, dev 1 = physical GPU 0 = consumer B). Needs a clean GPU 0/3 (build/wait.sh).
import os, subprocess, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctapp.ext import ROOT, load
from ctapp.timing import measure
from ctapp.pipeline import Pipeline
from ctapp.methods import Base, Ctapp, N, K, BF

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)
KNOB = sys.argv[sys.argv.index("--knob") + 1] if "--knob" in sys.argv else ""
CFG, TILE = 7, 128
FENCE = 3 if KNOB == "relaxed" else 2 if KNOB == "nofence" else 1            # nofence: UNSAFE knob, measures the fence cost only
GROUPS = (2, 4) if KNOB == "g24" else (1, 8)     # producer tile orders
SB = "tile" if "--tile" in sys.argv else "rowpanel"   # --tile: old per-consumer-tile scoreboard
TAG = "" if SB == "tile" else "rowpanel_"
REPS, WARMUP = 10, 5
i64 = dict(dtype=torch.int64)


def smi():
    return subprocess.run("nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv; nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv",
                          shell=True, capture_output=True, text=True).stdout


def timer_resolution():
    out = []
    for d in (0, 1):
        t = torch.zeros(20000, device=f"cuda:{d}", **i64); ext.timer_reads(t); torch.cuda.synchronize(d)
        dt = np.diff(t.cpu().numpy()); nz = dt[dt > 0]
        vals, cnt = np.unique(nz, return_counts=True)
        out.append((d, int(nz.min()), int(np.median(nz)), int(vals[cnt.argmax()]), float((dt == 0).mean())))
    return out


def clock_offset(n=200, trials=5):
    """offset = tB - (tA1+tA2)/2 from the lowest-RTT ping-pongs (A = dev 0, B = dev 1). Returns (offset_ns, min_rtt_ns, median_rtt_ns)."""
    offs, rtts = [], []
    for _ in range(trials):
        fa, fb = torch.zeros(1, device="cuda:0", dtype=torch.int32), torch.zeros(1, device="cuda:1", dtype=torch.int32)
        oa, ob = torch.zeros(2 * n, device="cuda:0", **i64), torch.zeros(n, device="cuda:1", **i64)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        ext.pingpong(fb, fa, ob, n, 0)  # B's kernel first (its flag lives on B, it writes A's flag)
        ext.pingpong(fa, fb, oa, n, 1)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        a = oa.cpu().numpy().reshape(n, 2); b = ob.cpu().numpy()
        rtt = a[:, 1] - a[:, 0]; off = b - (a[:, 0] + a[:, 1]) / 2
        keep = np.argsort(rtt)[1:11]  # skip sample 0 (start-up), take the 10 lowest RTTs
        offs.append(off[keep].mean()); rtts.append(rtt[keep]); rtts.append(rtt[1:])
    lo = np.concatenate(rtts[0::2]); allr = np.concatenate(rtts[1::2])
    return float(np.mean(offs)), float(np.std(offs)), float(lo.min()), float(np.median(allr))


class Timeline(Base):
    """kind: pipe = producer on A + consumer on B (as fig5 Ctapp); cons = consumer alone on B, pre-filled queue; prod = producer alone on A."""

    def __init__(self, M, X, W1, W2, g, kind):
        self.pl = Pipeline(M, N, K, CFG, CFG, devA=0, devB=1, fence=FENCE, group_rows=g, scoreboard=SB)
        self.X, self.W1, self.W2, self.kind = X, W1, W2, kind
        pl = self.pl
        self.sA = torch.zeros(pl.src_a.numel() * 8, device="cuda:0", **i64); self.sB = torch.zeros(pl.src_b.numel() * 8, device="cuda:1", **i64)
        self.gA, self.gB = torch.zeros(1, device="cuda:0", **i64), torch.zeros(1, device="cuda:1", **i64)
        self.snaps, self.pending = [], False
        self.Y1p = torch.empty(M, N, device="cuda:0", dtype=BF) if KNOB == "localy1" and kind == "prod" else pl.Y1  # localy1: producer stores Y1 locally

    def snap(self):
        if self.pending:
            self.snaps.append(dict(gA=self.gA.item(), gB=self.gB.item(), sA=self.sA.cpu().numpy().reshape(-1, 8), sB=self.sB.cpu().numpy().reshape(-1, 8)))
        self.pending = False

    def prepare(self):
        self.snap()
        pl = self.pl; pl.reset()
        if self.kind == "cons":
            with torch.cuda.device(1):
                pl.src_b.copy_(torch.arange(pl.src_b.numel(), device="cuda:1", dtype=torch.int32)); pl.tail_b.fill_(pl.src_b.numel())
        self.pending = True

    def prod(self):
        pl = self.pl
        with torch.cuda.device(0):
            ext.stamp_now(self.gA)
            ext.ctapp_gemm(self.X, self.W1, self.Y1p, pl.cfg1, pl.src_a, pl.head_a, pl.tail_a, pl.dep_off, pl.dep_cons, pl.sb, pl.src_b,
                           pl.head_b, pl.tail_b, pl.tn1, 0, pl.fence, self.sA, tiles_n2=pl.tn2, rowpanel=pl.rowpanel)

    def cons(self):
        pl = self.pl
        with torch.cuda.device(1):
            ext.stamp_now(self.gB)
            ext.ctapp_gemm(pl.Y1, self.W2, pl.Y2, pl.cfg2, pl.src_b, pl.head_b, pl.tail_b, None, None, None, None, None, None, pl.tn2, 0, 0, self.sB)

    def enqueue(self):
        if self.kind == "prod": self.prod()
        elif self.kind == "cons": self.cons()
        elif KNOB == "consfirst": self.cons(); self.prod()
        else: self.prod(); self.cons()


def run_case(M, g, kind, X, W1, W2):
    m = Timeline(M, X, W1, W2, g, kind); times = []
    r = measure(m, reps=REPS, warmup=WARMUP, devs=(1,) if kind == "cons" else (0, 1) if kind == "pipe" else (0,), times=times)
    m.snap()
    snaps = m.snaps[1 + WARMUP:]
    assert len(snaps) == REPS == len(times), (len(snaps), len(times))
    i = sorted(range(REPS), key=lambda j: times[j])[REPS // 2]  # median rep
    return m, snaps[i], times[i], r


def us(x): return np.asarray(x, dtype=np.float64) / 1e3


def analyse(m, s, harness, off, kind):
    """All times in us relative to gate stamp on A (B clock mapped to A by offset). Returns dict."""
    pl = m.pl; d = {}
    t0 = s["gA"] if kind != "cons" else s["gB"] - off
    A = s["sA"].astype(np.float64) if kind != "cons" else None
    B = s["sB"].astype(np.float64) if kind != "prod" else None
    if B is not None: B = B.copy(); B[:, :5] -= off
    rel = lambda x: (x - t0) / 1e3
    d["harness"] = harness
    if A is not None:
        d["A_first_entry"] = rel(A[:, 0].min()); d["A_last_k4"] = rel(A[:, 4].max()); d["A_span"] = d["A_last_k4"] - d["A_first_entry"]
        d["A_main"] = us(A[:, 2] - A[:, 1]).mean(); d["A_store"] = us(A[:, 3] - A[:, 2]).mean(); d["A_sig"] = us(A[:, 4] - A[:, 3]).mean()
        d["A_entry_lag"] = us(A[:, 1] - A[:, 0]).mean()
        ev = np.concatenate([A[:, 0], A[:, 4]]); sg = np.concatenate([np.ones(len(A)), -np.ones(len(A))])
        d["A_max_conc"] = int(np.cumsum(sg[np.lexsort((-sg, ev))]).max())
        d["A_slots"] = ext.occupancy(CFG) * torch.cuda.get_device_properties(0).multi_processor_count
        # k0 and k1 differ by prologue; wave = order of tile start
        tn = pl.tn1
        rows = (A[:, 5] // tn).astype(int); done = np.array([A[rows == r, 4].max() for r in range(rows.max() + 1)])
        d["row0_done"] = rel(done[0]); d["first_row_done"] = rel(done.min()); d["last_row_done"] = rel(done.max())
        d["rows_done"] = us(np.sort(done) - t0)
    if B is not None:
        d["B_first_entry"] = rel(B[:, 0].min()); d["B_first_acq"] = rel(B[:, 1].min()); d["B_last_k4"] = rel(B[:, 4].max())
        d["B_span"] = d["B_last_k4"] - d["B_first_entry"]
        wait = us(B[:, 1] - B[:, 0]); d["B_wait_sum"] = wait.sum(); d["B_wait_pct"] = np.percentile(wait, [10, 50, 90, 100])
        d["B_main"] = us(B[:, 2] - B[:, 1]).mean(); d["B_store"] = us(B[:, 3] - B[:, 2]).mean()
        d["B_main_med"] = np.median(us(B[:, 2] - B[:, 1]))
        d["B_max_conc"] = None
    if A is not None and B is not None:
        d["gate_skew"] = (s["gB"] - off - s["gA"]) / 1e3          # B's gate stamp minus A's
        d["entry_skew"] = d["B_first_entry"] - d["A_first_entry"]
        # readiness of consumer tile = when its producer row panel completed (last k4); latency = acquire - ready
        rowdone = {int(r): A[A[:, 5] // pl.tn1 == r, 4].max() for r in np.unique((A[:, 5] // pl.tn1).astype(int))}
        ready = np.array([rowdone[int(t // pl.tn2)] for t in B[:, 5]])
        lat = us(B[:, 1] - ready); d["sig_lat_all"] = lat
        waiting = B[:, 0] < ready  # CTA was already spinning when the tile became ready
        d["sig_lat_wait"] = lat[waiting]
        d["tail"] = d["B_last_k4"] - d["A_last_k4"]
        d["B_busy_after_A"] = us(np.clip(B[:, 4] - A[:, 4].max(), 0, None)).sum()
        d["B_first_after_row"] = d["B_first_acq"] - d["first_row_done"]
        d["e2e"] = d["B_last_k4"]
    return d


def gantt(name, title, sA, sB, off, t0):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    cols = dict(wait="#bbbbbb", main="#0072B2", store="#E69F00", sig="#D55E00")
    for ax, s, lab, o in ((axs[0], sA, "GPU A (producer)", 0), (axs[1], sB, "GPU B (consumer)", off)):
        s = s.astype(np.float64).copy(); s[:, :5] = (s[:, :5] - o - t0) / 1e3
        lanes = {}
        for i in np.argsort(s[:, 0]):
            sm = int(s[i, 6]); L = lanes.setdefault(sm, [])
            for j, end in enumerate(L):
                if end <= s[i, 0]: L[j] = s[i, 4]; break
            else: L.append(s[i, 4]); j = len(L) - 1
            y = sm + 0.5 * j
            for a, b, c in ((0, 1, "wait"), (1, 2, "main"), (2, 3, "store"), (3, 4, "sig")):
                ax.barh(y, s[i, b] - s[i, a], left=s[i, a], height=0.45, color=cols[c], align="edge", linewidth=0)
        ax.set_ylabel(f"{lab}: SM id"); ax.set_ylim(0, 133)
    axs[1].set_xlabel("time since gate open on A (us)")
    axs[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()], labels=["wait (entry to tile acquired)", "mainloop", "epilogue store", "signal"], loc="upper right")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(f"{ROOT}/results/{name}", dpi=110); plt.close(fig)


def main():
    print(smi(), flush=True)
    smi0 = smi()
    res = timer_resolution(); print("timer", res)
    o0 = clock_offset(); print("offset0", o0, flush=True)
    rows, cases = [], {}
    for M in (1024, 4096):
        torch.manual_seed(0)
        X = torch.randn(M, K, device="cuda:0", dtype=BF); W1 = torch.randn(N, K, device="cuda:0", dtype=BF) / K**0.5
        W2 = torch.randn(N, K, device="cuda:1", dtype=BF) / K**0.5
        off = o0[0]
        m, s, h, _ = run_case(M, 1, "cons", X, W1, W2); cases[(M, "cons")] = (m, s, analyse(m, s, h, off, "cons"))
        for g in GROUPS:
            if not KNOB:
                ms = []; measure(Ctapp(M, X, W1, W2, CFG, CFG, g, scoreboard=SB), reps=REPS, warmup=WARMUP, times=ms)  # same run with stamps off
                print(M, g, "stamps off", np.median(ms), flush=True); cases[(M, g, "off")] = (None, None, dict(harness=np.median(ms)))
            for kind in ("prod", "pipe"):
                m, s, h, _ = run_case(M, g, kind, X, W1, W2); cases[(M, g, kind)] = (m, s, analyse(m, s, h, off, kind)); print(M, g, kind, h, flush=True)
            if (M, g) in ((1024, 1), (4096, 8)) and not KNOB:
                m, s, _ = cases[(M, g, "pipe")]; gantt(f"timeline_{TAG}M{M}.png", f"CTAPP cfg7/7 {SB} M={M} group_rows={g} (median rep, harness {cases[(M, g, 'pipe')][2]['harness']:.0f} us)", s["sA"], s["sB"], off, s["gA"])
    o1 = clock_offset(); print("offset1", o1)
    smi1 = smi()
    import pickle; pickle.dump(dict(res=res, o0=o0, o1=o1, smi0=smi0, smi1=smi1, cases={k: v[2] for k, v in cases.items()}, raw={k: v[1] for k, v in cases.items() if v[1]}), open(f"{ROOT}/build/timeline_{TAG}{KNOB or 'main'}.pkl", "wb"))


if __name__ == "__main__":
    main()
