// CTA-pipelining protocol (PLAN.md Phase 1): workqueue prologue and dependency-signalling epilogue.
#pragma once
#include <cuda/atomic>

struct WorkQueue { int* entries; int* head; int* tail; int cap; };  // entries init EMPTY = -1

struct CtappParams {
  unsigned long long* stamps;   // null = off (zero cost); else 8 u64 per CTA at [blockIdx.x*8 + k], globaltimer ns: k0 entry, k1 tile acquired, k2 mainloop done, k3 epilogue done, k4 signalled (= k3 if no dst), k5 tile id, k6 smid
  WorkQueue src;                // local to this kernel's GPU; always present
  int const* dep_offsets;       // CSR over this kernel's tiles (null if last kernel)
  int const* dep_consumers;
  int* scoreboard;              // local, one counter per downstream tile
  WorkQueue dst;                // pointers into the downstream GPU's memory (peer); unused if dep_offsets == null
  int tiles_n;                  // this kernel's N-tiles, to decode tile -> (m, n)
  int skip_wait;                // debug negative control: consumer takes tile = idx without waiting
  int tiles_n2;                 // downstream kernel's N-tiles (row-panel mode: entries published per panel)
  int rowpanel;                 // 0: scoreboard/dep_consumers per consumer tile; 1: per consumer row panel (counters padded to 32 ints = one 128 B line)
  int fence;                    // 0: every thread fence.sys + barrier (default); 1 (F1): barrier, then signalling threads fence.acq_rel.sys; 2 (F0): no fence, UNSAFE, measurement only; 3: additionally relaxed counter atomic, UNSAFE
};

// Thread 0 of the CTA records %globaltimer into slot k (no-op when stamps are off).
__device__ __forceinline__ void ctapp_stamp(CtappParams const& p, int k) {
  if (p.stamps && threadIdx.x == 0) {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamps[blockIdx.x * 8 + k] = t;
  }
}

// Returns the tile id this CTA works on. Must be called by all threads of the CTA.
// s_tile: one int of dynamic smem placed after the kernel's SharedStorage (a static __shared__ would shift the
// dynamic smem base and break the TMA swizzle alignment).
__device__ int ctapp_prologue(CtappParams const& p, int* s_tile) {
  if (threadIdx.x == 0) {
    int idx = atomicAdd(p.src.head, 1);
    int v = idx;
    if (!p.skip_wait) {
      cuda::atomic_ref<int, cuda::thread_scope_system> e(p.src.entries[idx % p.src.cap]);
      while ((v = e.load(cuda::memory_order_acquire)) == -1) {}
    }
    *s_tile = v;
    ctapp_stamp(p, 1);
    if (p.stamps) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); p.stamps[blockIdx.x * 8 + 5] = v; p.stamps[blockIdx.x * 8 + 6] = sm; }
  }
  __syncthreads();
  // TMA loads (async proxy) must observe data another GPU wrote with ordinary stores.
  asm volatile("fence.proxy.async.global;" ::: "memory");
  return *s_tile;
}

// Called after the collective epilogue stored the tile.
__device__ void ctapp_epilogue(CtappParams const& p, int tile) {
  if (p.fence == 0) __threadfence_system();
  __syncthreads();
  int begin = p.dep_offsets[tile], deg = p.dep_offsets[tile + 1] - begin;
  for (int i = threadIdx.x; i < deg; i += blockDim.x) {
    int c = p.dep_consumers[begin + i];
    if (p.fence == 1) cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
    int old = cuda::atomic_ref<int, cuda::thread_scope_device>(p.scoreboard[p.rowpanel ? c * 32 : c]).fetch_sub(1, p.fence == 3 ? cuda::memory_order_relaxed : cuda::memory_order_acq_rel);
    if (old == 1 && p.rowpanel) {  // c = consumer row panel: one remote atomic reserves tiles_n2 slots, one release fence covers all the relaxed entry stores
      int base = cuda::atomic_ref<int, cuda::thread_scope_system>(*p.dst.tail).fetch_add(p.tiles_n2, cuda::memory_order_relaxed);
      cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
      for (int j = 0; j < p.tiles_n2; j++)
        cuda::atomic_ref<int, cuda::thread_scope_system>(p.dst.entries[(base + j) % p.dst.cap]).store(c * p.tiles_n2 + j, cuda::memory_order_relaxed);
    } else if (old == 1) {
      int slot = cuda::atomic_ref<int, cuda::thread_scope_system>(*p.dst.tail).fetch_add(1, cuda::memory_order_relaxed);
      cuda::atomic_ref<int, cuda::thread_scope_system>(p.dst.entries[slot % p.dst.cap]).store(c, cuda::memory_order_release);
    }
  }
}
