import torch
from ctapp.ext import load
from ctapp.pipeline import Pipeline

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)  # without it cudaMemcpyPeerAsync stages through the host
BF = torch.bfloat16


def dev(i): return torch.device("cuda", i)


def dims(X, W1, W2, act):
    """-> K, N1 (Y1 width = W1 rows), H (width fed to GEMM2), N2. act='swiglu': W1 is (2I, K) = [gate; up], H = I."""
    assert act in (None, "swiglu")
    K, N1, N2 = X.shape[1], W1.shape[0], W2.shape[0]
    H = N1 // 2 if act == "swiglu" else N1
    assert W1.shape[1] == K and W2.shape[1] == H, (X.shape, W1.shape, W2.shape, act)
    return K, N1, H, N2


def swiglu(y, out=None):
    g, u = y.chunk(2, dim=1)
    return torch.mul(torch.nn.functional.silu(g), u, out=out)


class Base:
    def prepare(self): pass
    def result(self): return self.Y2


class OneGpuCublas(Base):
    def __init__(self, M, X, W1, W2, act=None):  # X, W1, W2 on device 0
        self.X, self.W1, self.W2, self.act = X, W1, W2, act
        K, N1, H, N2 = dims(X, W1, W2, act)
        self.Y1 = torch.empty(M, N1, device=dev(0), dtype=BF); self.Y2 = torch.empty(M, N2, device=dev(0), dtype=BF)

    def enqueue(self):
        with torch.cuda.device(0):
            torch.mm(self.X, self.W1.t(), out=self.Y1)
            torch.mm(self.Y1 if self.act is None else swiglu(self.Y1), self.W2.t(), out=self.Y2)


class OneGpuTma(OneGpuCublas):
    def __init__(self, M, X, W1, W2, cfg, act=None):
        assert act is None; super().__init__(M, X, W1, W2); self.cfg = cfg

    def enqueue(self):
        with torch.cuda.device(0):
            ext.gemm_tma(self.X, self.W1, self.Y1, self.cfg); ext.gemm_tma(self.Y1, self.W2, self.Y2, self.cfg)


class MicroBatchCublas(Base):
    """X, W1 on A (dev 0); W2 on B (dev 1). chunk == M is the plain sequential 2-GPU pipeline."""

    def __init__(self, M, X, W1, W2, chunk, act=None):
        self.M, self.chunk, self.X, self.W1, self.W2, self.act = M, min(chunk, M), X, W1, W2, act
        K, N1, H, N2 = dims(X, W1, W2, act)
        self.Y1a = torch.empty(M, N1, device=dev(0), dtype=BF)  # swiglu: the (gate, up) tensor is what crosses the link (2I wide)
        self.Y1b = torch.empty(M, N1, device=dev(1), dtype=BF)
        self.Y2 = torch.empty(M, N2, device=dev(1), dtype=BF)
        self.sA, self.sB = torch.cuda.current_stream(0), torch.cuda.current_stream(1)
        with torch.cuda.device(0): self.sC = torch.cuda.Stream()
        self.n = -(-M // self.chunk)
        self.evA = [torch.cuda.Event() for _ in range(self.n)]; self.evC = [torch.cuda.Event() for _ in range(self.n)]

    def gemm(self, x, w, y):
        torch.mm(x, w.t(), out=y)

    def enqueue(self):
        c = self.chunk
        for i in range(self.n):
            sl = slice(i * c, min((i + 1) * c, self.M))
            with torch.cuda.device(0):
                with torch.cuda.stream(self.sA):
                    self.gemm(self.X[sl], self.W1, self.Y1a[sl]); self.evA[i].record(self.sA)
                with torch.cuda.stream(self.sC):
                    self.sC.wait_event(self.evA[i]); ext.peer_copy(self.Y1b[sl], self.Y1a[sl]); self.evC[i].record(self.sC)
            with torch.cuda.device(1), torch.cuda.stream(self.sB):
                self.sB.wait_event(self.evC[i])
                self.gemm(self.Y1b[sl] if self.act is None else swiglu(self.Y1b[sl]), self.W2, self.Y2[sl])


class MicroBatchTma(MicroBatchCublas):
    def __init__(self, M, X, W1, W2, chunk, cfg, act=None):
        assert act is None; super().__init__(M, X, W1, W2, chunk); self.cfg = cfg

    def gemm(self, x, w, y):
        ext.gemm_tma(x, w, y, self.cfg)


class TensorParallel2(Base):
    """Megatron: W1 rows split (swiglu: gate and up each split, per-rank shard = [gate_shard; up_shard]), W2 columns split, all-reduce
    of the (M, N2) partials. X replicated on both GPUs. Inputs X, W1 on dev0, W2 on dev1."""
    def __init__(self, M, X, W1, W2, act=None):
        K, N1, H, N2 = dims(X, W1, W2, act)
        assert H % 2 == 0, "sharded dim must be even"
        h = H // 2; self.act = act
        W2 = W2.to(dev(0))
        self.X = [X, X.to(dev(1))]
        if act == "swiglu":
            gate, up = W1[:H], W1[H:]
            self.W1 = [torch.cat([gate[g * h:(g + 1) * h], up[g * h:(g + 1) * h]]).to(dev(g)) for g in range(2)]  # (2h, K)
        else:
            self.W1 = [W1[g * h:(g + 1) * h].contiguous().to(dev(g)) for g in range(2)]     # (h, K)
        self.W2 = [W2[:, g * h:(g + 1) * h].contiguous().to(dev(g)) for g in range(2)]  # (N2, h)
        w = self.W1[0].shape[0]
        self.Y1 = [torch.empty(M, w, device=dev(g), dtype=BF) for g in range(2)]
        self.P = [torch.empty(M, N2, device=dev(g), dtype=BF) for g in range(2)]
        self.s = [torch.cuda.current_stream(g) for g in range(2)]
        for _ in range(3): self.enqueue()  # warm NCCL
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def enqueue(self):
        for g in range(2):
            with torch.cuda.device(g):
                torch.mm(self.X[g], self.W1[g].t(), out=self.Y1[g])
                torch.mm(self.Y1[g] if self.act is None else swiglu(self.Y1[g]), self.W2[g].t(), out=self.P[g])
        torch.cuda.nccl.all_reduce(self.P, streams=self.s)

    def result(self): return self.P[1]


class Ctapp(Base):
    def __init__(self, M, X, W1, W2, cfg_p, cfg_c, group_rows, fence=1, scoreboard="rowpanel", act=None):
        assert act is None
        K, N1, H, N2 = dims(X, W1, W2, act)
        self.pl = Pipeline(M, K, N1, N2, cfg_p, cfg_c, devA=0, devB=1, fence=fence, group_rows=group_rows, scoreboard=scoreboard)
        self.X, self.W1, self.W2 = X, W1, W2

    def prepare(self): self.pl.reset()

    def enqueue(self):
        pl = self.pl
        with torch.cuda.device(0):
            pl.ext.ctapp_gemm(self.X, self.W1, pl.Y1, pl.cfg1, pl.src_a, pl.head_a, pl.tail_a, pl.dep_off, pl.dep_cons, pl.sb, pl.src_b,
                              pl.head_b, pl.tail_b, pl.tn1, 0, pl.fence, tiles_n2=pl.tn2, rowpanel=pl.rowpanel)
        with torch.cuda.device(1):
            pl.ext.ctapp_gemm(pl.Y1, self.W2, pl.Y2, pl.cfg2, pl.src_b, pl.head_b, pl.tail_b, None, None, None, None, None, None, pl.tn2, 0, 0)

    def result(self): return self.pl.Y2
