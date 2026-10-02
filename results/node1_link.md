# S0.1 link characterisation: brev-tffbb7a2o, CUDA_VISIBLE_DEVICES=0,1,2,3 (4 GPUs; dev 0 = source)

```
	[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NV18	NV18	NV18	44-51,148-155	0		N/A
GPU1	NV18	 X 	NV18	NV18		1		N/A
GPU2	NV18	NV18	 X 	NV18		1		N/A
GPU3	NV18	NV18	NV18	 X 		1		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

Multicast (NVLS) support: torch `has_multicast_support` = [True, True, True, True], `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` = [1, 1, 1, 1]

## Bandwidth (256 MiB, median of 20 reps, 3 runs; spread = (max-min)/median)

| path | GB/s | spread % |
|---|---|---|
| SM 16 B stores, local HBM | 1386.0 | 0.1 |
| SM 16 B stores, dev0 -> dev1 | 369.0 | 0.1 |
| SM 16 B stores, dev0 -> dev2 | 369.1 | 0.0 |
| SM 16 B stores, dev0 -> dev3 | 369.0 | 0.1 |
| SM stores fan-out, dev0 -> 2 peers concurrently (aggregate egress) | 366.1 | 0.0 |
| SM stores fan-out, dev0 -> 3 peers concurrently (aggregate egress) | 368.1 | 0.0 |
| copy engine (cudaMemcpyPeerAsync), dev0 -> dev1 | 392.1 | 0.0 |

## Flag round trip (system-scope release/acquire ping-pong, 200 iterations x 5 trials)

| peer | min RTT us | median RTT us |
|---|---|---|
| dev1 | 4.32 | 4.58 |
| dev2 | 4.26 | 4.51 |
| dev3 | 4.42 | 4.64 |

## TMA-store D local vs peer (stock coop cfg0 128x256x64, N=K=8192, raster 1 swizzle 1; link bound = bytes / peer SM-store GB/s)

| M | cuBLAS us | local us | peer us | link bound us | peer/local | peer / max(local, link) | D equal |
|---|---|---|---|---|---|---|---|
| 1024 | 173 | 196 | 234 | 45 | 1.196 | 1.196 | True |
| 4096 | 732 | 709 | 805 | 182 | 1.136 | 1.136 | True |
| 16384 | 3264 | 3500 | 3557 | 727 | 1.016 | 1.016 | True |

## Model with the measured link (PLAN.md pen-and-paper, 700 TFLOP/s)

- Peer SM-store bandwidth used: 369.0 GB/s (1 peer); fan-out aggregate: 368.1 GB/s.
- Last-wave drain floor (132 x 128x256 bf16 tiles = 8.65 MB): **23.4 us** (was 69 us at 124 GB/s).

| M (paper square) | ideal us | CTAPP model us | model/ideal |
|---|---|---|---|
| 1024 | 196 | 259 | 1.321 |
| 2048 | 393 | 456 | 1.160 |
| 4096 | 785 | 848 | 1.080 |
| 8192 | 1571 | 1634 | 1.040 |
| 16384 | 3141 | 3204 | 1.020 |

70B FFN per-tile-reduce link ratio = (t-1) * (TF/BW) / inter (must be << 1 to hide under GEMM2):

- TP2 (1 peer): 0.07
- TP4 (3 peers, fan-out bw): 0.20
- TP8 (7 peers, fan-out bw): 0.46
