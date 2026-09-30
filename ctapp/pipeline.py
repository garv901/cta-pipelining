import torch
from ctapp.ext import load

EMPTY = -1


def build_deps(M, N1, BM1, BN1, BM2, BN2, rowpanel=True):
    """CSR dependency array over producer tiles (row-major ids) + consumer scoreboard init. Assumes N2 = N1.
    rowpanel=False: deps/counters per consumer tile. rowpanel=True: per consumer row panel (every tile of a panel has the same
    deps); counters are padded to 32 ints (one 128 B line) each."""
    tm1, tn1 = -(-M // BM1), -(-N1 // BN1)
    tm2, tn2 = -(-M // BM2), -(-N1 // BN2)
    m1 = torch.arange(tm1)
    lo = torch.div(m1 * BM1, BM2, rounding_mode="floor")
    hi = torch.div(torch.clamp((m1 + 1) * BM1, max=M) - 1, BM2, rounding_mode="floor")
    cnt_m = hi - lo + 1                                        # overlapping consumer row panels per producer row panel
    per = 1 if rowpanel else tn2                               # consumers per overlapping row panel
    deg = (cnt_m * per).repeat_interleave(tn1)                 # per producer tile (m1-major, n1 minor)
    dep_offsets = torch.zeros(tm1 * tn1 + 1, dtype=torch.int64)
    dep_offsets[1:] = deg.cumsum(0)
    # consumers of tile (m1, n1): m2 in [lo, hi] (and n2 in [0, tn2) per tile) -> m2 (row panel) or m2 * tn2 + n2
    tile_m1 = m1.repeat_interleave(tn1)
    idx = torch.arange(int(dep_offsets[-1])) - dep_offsets[:-1].repeat_interleave(deg)
    rows = lo[tile_m1].repeat_interleave(deg) + torch.div(idx, per, rounding_mode="floor")
    dep_consumers = rows if rowpanel else rows * tn2 + idx % tn2
    init = torch.zeros(tm2 if rowpanel else tm2 * tn2, dtype=torch.int64)
    init.scatter_add_(0, dep_consumers, torch.ones_like(dep_consumers))
    assert int(deg.sum()) == int(init.sum())
    if rowpanel:
        padded = torch.zeros(tm2 * 32, dtype=torch.int64); padded[::32] = init; init = padded
    return dep_offsets.int(), dep_consumers.int(), init.int()


class Pipeline:
    """2-layer CTA-pipelined run: producer GEMM on devA writes Y1 into devB memory, consumer GEMM on devB."""

    def __init__(self, M, N, K, cfg1, cfg2, devA=0, devB=1, skip_wait=0, fence=0, group_rows=0, scoreboard="rowpanel"):
        self.ext = load()
        self.ext.enable_peer_access(devA, devB); self.ext.enable_peer_access(devB, devA)
        self.M, self.cfg1, self.cfg2, self.skip_wait, self.fence = M, cfg1, cfg2, skip_wait, fence
        self.rowpanel = int(scoreboard == "rowpanel"); assert scoreboard in ("rowpanel", "tile")
        self.a, self.b = torch.device("cuda", devA), torch.device("cuda", devB)
        BM1, BN1 = self.tile(cfg1); BM2, BN2 = self.tile(cfg2)
        self.tn1, self.tn2 = -(-N // BN1), -(-N // BN2)
        P, C = -(-M // BM1) * self.tn1, -(-M // BM2) * self.tn2
        off, cons, sb = build_deps(M, N, BM1, BN1, BM2, BN2, bool(self.rowpanel))
        i32 = dict(dtype=torch.int32)
        self.src_a = torch.arange(P, device=self.a, **i32)
        if group_rows:  # producer tile order: groups of group_rows tile-rows, n-major inside a group (row-major = L2-hostile, see results/overhead.md)
            t = torch.arange(P); m, n = t // self.tn1, t % self.tn1
            self.src_a = torch.argsort(((m // group_rows) * self.tn1 + n) * group_rows + m % group_rows).to(self.a).int()
        self.head_a = torch.zeros(1, device=self.a, **i32); self.tail_a = torch.full((1,), P, device=self.a, **i32)
        self.dep_off, self.dep_cons = off.to(self.a), cons.to(self.a)
        self.sb_pristine = sb.to(self.a); self.sb = self.sb_pristine.clone()
        self.src_b = torch.full((C,), EMPTY, device=self.b, **i32)
        self.head_b = torch.zeros(1, device=self.b, **i32); self.tail_b = torch.zeros(1, device=self.b, **i32)
        self.Y1 = torch.empty(M, N, device=self.b, dtype=torch.bfloat16)
        self.Y2 = torch.empty(M, N, device=self.b, dtype=torch.bfloat16)

    def tile(self, cfg):
        s = self.ext.config_info(cfg).split("tile ")[1].split("x")
        return int(s[0]), int(s[1])

    def reset(self):
        with torch.cuda.device(self.a):
            self.head_a.zero_(); self.sb.copy_(self.sb_pristine)
        with torch.cuda.device(self.b):
            self.src_b.fill_(EMPTY); self.head_b.zero_(); self.tail_b.zero_()
            ev = torch.cuda.Event(); ev.record()
        # the producer must not write B's queue/Y1 before B has finished its reset (and any previous run)
        torch.cuda.current_stream(self.a).wait_event(ev)

    def run(self, X, W1, W2):
        self.reset()
        with torch.cuda.device(self.a):
            self.ext.ctapp_gemm(X, W1, self.Y1, self.cfg1, self.src_a, self.head_a, self.tail_a, self.dep_off, self.dep_cons,
                                self.sb, self.src_b, self.head_b, self.tail_b, self.tn1, 0, self.fence, tiles_n2=self.tn2, rowpanel=self.rowpanel)
        with torch.cuda.device(self.b):
            self.ext.ctapp_gemm(self.Y1, W2, self.Y2, self.cfg2, self.src_b, self.head_b, self.tail_b, None, None, None,
                                None, None, None, self.tn2, self.skip_wait, 0)
        return self.Y2
