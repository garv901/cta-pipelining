# Builds results/timeline.md from build/timeline*.pkl written by bench/timeline.py (base run + --knob runs).
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ctapp.ext import ROOT

L = lambda k: pickle.load(open(f"{ROOT}/build/timeline{k}.pkl", "rb"))
base, knobs = L(""), {k: L("_" + k) for k in ("nofence", "localy1", "g24")}
c = base["cases"]; o0, o1 = base["o0"], base["o1"]
out = []; w = out.append
f = lambda x, n=1: f"{x:.{n}f}"

w("# Phase 2b: small-M timeline of CTA-pipelining (cfg7/7 = 128x128x64, fence=1, 2 CTAs/SM = 264 slots)\n")
w("Producer = physical GPU 3 (A), consumer = physical GPU 0 (B). Times in us relative to the gate stamp on A (1-thread kernel right after the gate wait); B stamps mapped to A's clock with the measured offset. Median rep of 10 (5 warmup), gated harness identical to fig5.\n")
w("## Clock and timer\n")
w("| item | value |\n|---|---|")
w("| globaltimer tick (20000 consecutive reads, dev A and B) | " + "; ".join(f"dev{d}: min nonzero delta {mn} ns, median {md} ns, mode {mo} ns, {z*100:.0f}% of consecutive reads identical" for d, mn, md, mo, z in base["res"]) + " |")
w(f"| A-to-B globaltimer offset (tB - tA), start / end of run | {o0[0]/1e9:.6f} s / {o1[0]/1e9:.6f} s (drift {(o1[0]-o0[0])/1e3:.1f} us over the run; std of the 5 trials {o0[1]:.0f} / {o1[1]:.0f} ns) |")
w(f"| ping-pong RTT via peer flags (200 round trips x 5) | min {o0[2]/1e3:.2f} us, median {o0[3]/1e3:.2f} us (one-way flag latency about {o0[2]/2e3:.1f} us) |")
w("| harness stamps-on vs stamps-off (same session) | " + ", ".join(f"M={M} g{g}: {c[(M,g,'pipe')]['harness']:.0f} vs {c[(M,g,'off')]['harness']:.0f}" for M, g in ((1024, 1), (1024, 8), (4096, 1), (4096, 8))) + " us |\n")

cases = [(1024, 1), (1024, 8), (4096, 1), (4096, 8)]
hdr = "| metric (us) | " + " | ".join(f"M={M} g{g}" for M, g in cases) + " |\n|---|" + "---|" * len(cases)
def row(name, fn):
    w(f"| {name} | " + " | ".join(fn(c[(M, g, 'pipe')], c[(M, g, 'prod')], c[(M, 'cons')], M, g) for M, g in cases) + " |")
w("## Decomposition (pipelined run)\n"); w(hdr)
row("harness latency (stamps on)", lambda p, a, b, M, g: f(p["harness"], 0))
row("**1. start skew**: B gate stamp minus A", lambda p, a, b, M, g: f(p["gate_skew"]))
row("first CTA entry B minus first CTA entry A", lambda p, a, b, M, g: f(p["entry_skew"]))
row("**2. producer**: first CTA entry", lambda p, a, b, M, g: f(p["A_first_entry"]))
row("last producer k4 (all signalled)", lambda p, a, b, M, g: f(p["A_last_k4"]))
row("producer span (first entry to last k4)", lambda p, a, b, M, g: f(p["A_span"]))
row("per-CTA mainloop k2-k1 (mean)", lambda p, a, b, M, g: f(p["A_main"]))
row("per-CTA store k3-k2 (mean)", lambda p, a, b, M, g: f(p["A_store"]))
row("per-CTA signal k4-k3 (mean)", lambda p, a, b, M, g: f(p["A_sig"]))
row("tiles / slots / waves; max CTAs alive", lambda p, a, b, M, g: f"{M//128*64} / {p['A_slots']} / {-(-M//128*64//p['A_slots'])}; {p['A_max_conc']}")
row("**3. ramp**: first row panel complete (min over rows)", lambda p, a, b, M, g: f(p["first_row_done"]))
row("row 0 complete", lambda p, a, b, M, g: f(p["row0_done"]))
row("first consumer tile acquired (min k1 on B)", lambda p, a, b, M, g: f(p["B_first_acq"]))
row("signal latency, row ready to acquire, waiting CTAs (median / max)", lambda p, a, b, M, g: f"{np.median(p['sig_lat_wait']):.1f} / {p['sig_lat_wait'].max():.1f}")
row("**4. consumer**: total wait sum(k1-k0), CTA-us", lambda p, a, b, M, g: f(p["B_wait_sum"], 0))
row("wait p10/p50/p90/max per CTA", lambda p, a, b, M, g: "/".join(f(x, 0) for x in p["B_wait_pct"]))
row("mainloop per tile, pipelined (mean / median)", lambda p, a, b, M, g: f"{p['B_main']:.1f} / {p['B_main_med']:.1f}")
row("mainloop per tile, standalone (a) (mean / median)", lambda p, a, b, M, g: f"{b['B_main']:.1f} / {b['B_main_med']:.1f}")
row("consumer store per tile (pipelined / standalone)", lambda p, a, b, M, g: f"{p['B_store']:.1f} / {b['B_store']:.1f}")
row("**5. tail**: last consumer k4 minus last producer k4", lambda p, a, b, M, g: f(p["tail"]))
row("consumer work done after producer's last k4, CTA-us / 264 slots", lambda p, a, b, M, g: f(p["B_busy_after_A"] / 264))
row("**6. end-to-end from stamps** (last consumer k4)", lambda p, a, b, M, g: f(p["e2e"]))
row("harness minus stamps e2e (join, event/gate offsets)", lambda p, a, b, M, g: f(p["harness"] - p["e2e"]))
w("")
w("Standalone references: (a) consumer alone on B, pre-filled queue, no deps: span " + ", ".join(f"M={M}: {c[(M,'cons')]['B_span']:.0f}" for M in (1024, 4096)) +
  " us (harness " + ", ".join(f"{c[(M,'cons')]['harness']:.0f}" for M in (1024, 4096)) + "). (b) producer alone on A with signalling: harness " +
  ", ".join(f"M={M} g{g}: {c[(M,g,'prod')]['harness']:.0f}" for M, g in cases) + " us.\n")

w("## Time budget (harness = sum of parts, us)\n")
w("| component | " + " | ".join(f"M={M} g{g}" for M, g in cases) + " |\n|---|" + "---|" * len(cases))
lp = {(M, g): knobs["localy1"]["cases"][(M, g, "prod")] for M, g in cases}
def bud(name, fn): w(f"| {name} | " + " | ".join(fn(c[(M, g, 'pipe')], c[(M, 'cons')], lp[(M, g)]) for M, g in cases) + " |")
bud("start (gate to first producer CTA)", lambda p, b, l: f(p["A_first_entry"]))
bud("compute-equivalent: standalone consumer span (a)", lambda p, b, l: f(b["B_span"]))
bud("producer: signalling excess (span with local Y1 + signalling, minus (a))", lambda p, b, l: f(l["A_span"] - b["B_span"]))
bud("producer: remote-store excess (span minus span with local Y1)", lambda p, b, l: f(p["A_span"] - l["A_span"]))
bud("tail after producer end", lambda p, b, l: f(p["tail"]))
bud("join / unexplained (harness - stamps e2e)", lambda p, b, l: f(p["harness"] - p["e2e"]))
bud("**sum = harness**", lambda p, b, l: f(p["A_first_entry"] + b["B_span"] + (p["A_span"] - b["B_span"]) + p["tail"] + p["harness"] - p["e2e"]))
w("\n(Local-Y1 spans come from the `localy1` knob run, a separate run of the producer alone, so the split between the two producer excess rows has run-to-run noise of a few us.)\n")

w("## Knob experiments (harness us; span = producer first entry to last k4)\n")
w("| run | " + " | ".join(f"M={M} g{g}" for M, g in cases) + " |\n|---|" + "---|" * len(cases))
w("| baseline: pipe / producer alone | " + " | ".join(f"{c[(M,g,'pipe')]['harness']:.0f} / {c[(M,g,'prod')]['harness']:.0f}" for M, g in cases) + " |")
kn = knobs["nofence"]["cases"]
w("| nofence (fence=2, UNSAFE, measurement only): pipe / producer alone | " + " | ".join(f"{kn[(M,g,'pipe')]['harness']:.0f} / {kn[(M,g,'prod')]['harness']:.0f}" for M, g in cases) + " |")
w("| nofence: per-CTA signal | " + " | ".join(f"{kn[(M,g,'prod')]['A_sig']:.1f}" for M, g in cases) + " |")
kn = knobs["localy1"]["cases"]
w("| localy1 (producer alone, Y1 stored locally, signalling kept): harness | " + " | ".join(f"{kn[(M,g,'prod')]['harness']:.0f}" for M, g in cases) + " |")
w("| localy1: per-CTA store / signal | " + " | ".join(f"{kn[(M,g,'prod')]['A_store']:.1f} / {kn[(M,g,'prod')]['A_sig']:.1f}" for M, g in cases) + " |")
kn = knobs["g24"]["cases"]
w("| g24 (producer groups of 2 and 4 tile-rows): pipe harness, M=1024 g2 / g4; M=4096 g2 / g4 | " + f"{kn[(1024,2,'pipe')]['harness']:.0f} / {kn[(1024,4,'pipe')]['harness']:.0f} | | {kn[(4096,2,'pipe')]['harness']:.0f} / {kn[(4096,4,'pipe')]['harness']:.0f} | |")
w("")

# signal time vs concurrency (from raw stamps of the producer-alone runs)
w("## Signal time vs number of producer CTAs signalling at the same moment\n")
w("Per producer CTA: concurrency = number of CTAs whose [k3,k4] window contains the midpoint of this CTA's window. Median k4-k3 (us), producer alone, baseline run and localy1 run (no NVLink data stores).\n")
w("| case | run | 1-10 | 11-30 | 31-80 | 81+ |\n|---|---|---|---|---|---|")
for M, g in ((1024, 1), (4096, 1), (4096, 8)):
    for name, d in (("base", base), ("localy1", knobs["localy1"]), ("nofence", knobs["nofence"])):
        A = d["raw"][(M, g, "prod")]["sA"].astype(float) / 1e3; sig = A[:, 4] - A[:, 3]; mid = (A[:, 3] + A[:, 4]) / 2
        conc = np.array([((A[:, 3] <= m) & (A[:, 4] >= m)).sum() for m in mid])
        cells = [f"{np.median(sig[(conc >= lo) & (conc <= hi)]):.1f} (n={((conc >= lo) & (conc <= hi)).sum()})" if ((conc >= lo) & (conc <= hi)).any() else "-" for lo, hi in ((1, 10), (11, 30), (31, 80), (81, 999))]
        w(f"| M={M} g{g} | {name} | " + " | ".join(cells) + " |")
w("\n## Environment\n")
w("nvidia-smi before the base run:\n```\n" + base["smi0"] + "```\nafter:\n```\n" + base["smi1"] + "```")
w("""
Notes: the `localy1` knob only changes the producer-alone runs (its pipe rows are unchanged code and not used). Monitoring: a 1 s nvidia-smi sampler during each run showed only this job on GPUs 0/3, except brief pids on GPU 3 at start-up (see build/tl_*.log.mon); one localy1 run with such a pid present gave a producer-alone time of 266 us (M=1024 g1) and 1073 us (M=4096 g1) instead of 299 / 1226 in the three other runs, so treat localy1 as +-10% noise-prone.

## Diagnosis

| hypothesis | verdict | numbers |
|---|---|---|
| H1 start / launch skew | no | B gate stamp vs A: -1.7 to +4.6 us; first CTA entry skew the same; consumer is idle until the first row panel is done (>= 187 us) anyway. Not on the critical path. |
| H2 ramp (first row panel late) | not the cause of the excess | first row panel completes at 187 us (one producer wave incl. store + signal), consumer acquires it 1 us later (median signal latency 1 us at g1). The consumer only needs 232 us of work against a 341 us producer, so B is starved, not backlogged; the ramp does not extend the end time. At g8 the first row is done at 333 us (M=1024) / 372 us (M=4096), which is the reason g8 is worse at small M. |
| H3 consumer slower while overlapped | no | mainloop per tile pipelined vs standalone: 105 vs 104 (M=1024 g1), 107 vs 104 (g8), 108 vs 112 and 114 vs 112 (M=4096). Store 5.9 vs 5.9 us. |
| H4 producer slowed by signalling and remote stores | YES, main cause on the producer | per-CTA store + signal = 54 us at M=1024 (vs 6 us store on the consumer) and 43 us at M=4096. Producer span is 341 us vs 232 us for the same kernel with no protocol at M=1024, and 1251 vs 965 us at M=4096. |
| H5 tail after producer ends | YES, second cause | 114 us (M=1024 g1) = one full consumer wave (mainloop 105 + store 6): all rows of the last producer wave complete within about 2 us of each other, so 256 consumer tiles start together at the very end. 164 us at M=4096 g1; 240 / 283 us with g8. |

Where the producer excess goes:
* Signalling (k3 to k4, mean 29 / 24 us) is the largest part and is a contention cost, not a fence or link cost: an isolated CTA signals in 4 to 8 us, but when 80+ CTAs signal at once (the normal case, because CTAs run in lock-step waves) it takes 28 to 44 us, also with local Y1 (no link traffic). Removing the fence (fence=2) saves only 7 to 8 us per CTA. Each producer tile does tiles_n2 = 64 device-scope acq_rel fetch_sub on the scoreboard counters of its row panel, and all 64 producer tiles of a row hit the same 64 counters (256 contiguous bytes), so a wave issues about 17k atomics on about 8 cache lines. Not verified by a direct A/B change (would need a coalesced counter layout, beyond the 3-knob budget).
* Remote store (k2 to k3, 25 us vs 6 us local) costs about 50 us at M=1024, where both waves run in lock-step, so the 8.6 MB per wave burst (about 125 GB/s = the measured P2P store bandwidth) is not overlapped with any mainloop; only 15-30 us at M=4096 where waves de-synchronise and some drain hides behind other CTAs' mainloops.
* Because the tile count per row panel is 64 and a wave holds 4.1 row panels, dependency completion is bunched at wave boundaries (rows_done at M=1024 g1: 187 x4, 344 x4).
""")
open(f"{ROOT}/results/timeline.md", "w").write("\n".join(out) + "\n")
