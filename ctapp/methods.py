import torch
from ctapp.ext import load
from ctapp.pipeline import Pipeline

ext = load()
ext.enable_peer_access(0, 1); ext.enable_peer_access(1, 0)  # without it cudaMemcpyPeerAsync stages through the host
N = K = 8192
BF = torch.bfloat16


def dev(i): return torch.device("cuda", i)


class Base:
    def prepare(self): pass
    def result(self): return self.Y2


class OneGpuCublas(Base):
    def __init__(self, M, X, W1, W2):  # X, W1, W2 on device 0
        self.X, self.W1, self.W2 = X, W1, W2
        self.Y1 = torch.empty(M, N, device=dev(0), dtype=BF); self.Y2 = torch.empty_like(self.Y1)

    def enqueue(self):
        with torch.cuda.device(0):
            torch.mm(self.X, self.W1.t(), out=self.Y1); torch.mm(self.Y1, self.W2.t(), out=self.Y2)


class OneGpuTma(OneGpuCublas):
    def __init__(self, M, X, W1, W2, cfg):
        super().__init__(M, X, W1, W2); self.cfg = cfg

    def enqueue(self):
        with torch.cuda.device(0):
            ext.gemm_tma(self.X, self.W1, self.Y1, self.cfg); ext.gemm_tma(self.Y1, self.W2, self.Y2, self.cfg)


class MicroBatchCublas(Base):
    """X, W1 on A (dev 0); W2 on B (dev 1). chunk == M is the plain sequential 2-GPU pipeline."""

    def __init__(self, M, X, W1, W2, chunk):
        self.M, self.chunk, self.X, self.W1, self.W2 = M, min(chunk, M), X, W1, W2
        self.Y1a = torch.empty(M, N, device=dev(0), dtype=BF)
        self.Y1b = torch.empty(M, N, device=dev(1), dtype=BF)
        self.Y2 = torch.empty(M, N, device=dev(1), dtype=BF)
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
                self.sB.wait_event(self.evC[i]); self.gemm(self.Y1b[sl], self.W2, self.Y2[sl])


class MicroBatchTma(MicroBatchCublas):
    def __init__(self, M, X, W1, W2, chunk, cfg):
        super().__init__(M, X, W1, W2, chunk); self.cfg = cfg

    def gemm(self, x, w, y):
        ext.gemm_tma(x, w, y, self.cfg)


class TensorParallel2(Base):
    """W1 rows split, W2 columns split (contiguous shards). X replicated on both GPUs. Inputs X, W1 on dev0, W2 on dev1."""
    def __init__(self, M, X, W1, W2):
        h = N // 2
        W2 = W2.to(dev(0))
        self.X = [X, X.to(dev(1))]
        self.W1 = [W1[g * h:(g + 1) * h].contiguous().to(dev(g)) for g in range(2)]     # (N/2, K)
        self.W2 = [W2[:, g * h:(g + 1) * h].contiguous().to(dev(g)) for g in range(2)]  # (N, N/2)
        self.Y1 = [torch.empty(M, h, device=dev(g), dtype=BF) for g in range(2)]
        self.P = [torch.empty(M, N, device=dev(g), dtype=BF) for g in range(2)]
        self.s = [torch.cuda.current_stream(g) for g in range(2)]
        for _ in range(3): self.enqueue()  # warm NCCL
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def enqueue(self):
        for g in range(2):
            with torch.cuda.device(g):
                torch.mm(self.X[g], self.W1[g].t(), out=self.Y1[g]); torch.mm(self.Y1[g], self.W2[g].t(), out=self.P[g])
        torch.cuda.nccl.all_reduce(self.P, streams=self.s)

    def result(self): return self.P[1]


class Ctapp(Base):
    def __init__(self, M, X, W1, W2, cfg_p, cfg_c, group_rows, fence=1):
        self.pl = Pipeline(M, N, K, cfg_p, cfg_c, devA=0, devB=1, fence=fence, group_rows=group_rows)
        self.X, self.W1, self.W2 = X, W1, W2

    def prepare(self): self.pl.reset()

    def enqueue(self):
        pl = self.pl
        with torch.cuda.device(0):
            pl.ext.ctapp_gemm(self.X, self.W1, pl.Y1, pl.cfg1, pl.src_a, pl.head_a, pl.tail_a, pl.dep_off, pl.dep_cons, pl.sb, pl.src_b,
                              pl.head_b, pl.tail_b, pl.tn1, 0, pl.fence)
        with torch.cuda.device(1):
            pl.ext.ctapp_gemm(pl.Y1, self.W2, pl.Y2, pl.cfg2, pl.src_b, pl.head_b, pl.tail_b, None, None, None, None, None, None, pl.tn2, 0, 0)

    def result(self): return self.pl.Y2
