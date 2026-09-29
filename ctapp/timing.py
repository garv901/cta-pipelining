"""Gated latency measurement: all streams block on a host-memory flag (cuStreamWaitValue32), so host launch cost is outside the timed window."""
import statistics
import torch
from ctapp.ext import load

ext = load()
ext.gate_alloc()


def measure(method, reps=20, warmup=5, devs=(0, 1)):
    """method.prepare() runs before the gate; method.enqueue() is timed. Returns (median, p10, p90) in us. devs[0] is GPU A (timeline owner)."""
    streams = [torch.cuda.current_stream(d) for d in devs]
    # ungated run first: lazy CUDA module loading inside a gated window deadlocks (launch waits on the blocked stream)
    method.prepare(); method.enqueue()
    for d in devs: torch.cuda.synchronize(d)
    ts = []
    for i in range(warmup + reps):
        method.prepare()
        ext.gate_set(0)
        for d, s in zip(devs, streams):
            with torch.cuda.device(d), torch.cuda.stream(s): ext.gate_wait()
        with torch.cuda.device(devs[0]):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(streams[0])
        method.enqueue()
        for d, s in zip(devs[1:], streams[1:]):
            with torch.cuda.device(d): ev = torch.cuda.Event(); ev.record(s)
            streams[0].wait_event(ev)
        with torch.cuda.device(devs[0]): end.record(streams[0])
        ext.gate_set(1)
        for d in devs: torch.cuda.synchronize(d)
        if i >= warmup: ts.append(start.elapsed_time(end) * 1e3)
    q = statistics.quantiles(ts, n=10, method="inclusive")
    return statistics.median(ts), q[0], q[-1]
