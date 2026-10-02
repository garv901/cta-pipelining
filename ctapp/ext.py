import os
CUDA = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")  # must match torch's CUDA build
os.environ["PATH"] = f"{CUDA}/bin:" + os.environ["PATH"]
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
from torch.utils.cpp_extension import load as _load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    os.makedirs(f"{ROOT}/build/ext", exist_ok=True)
    return _load(
        name="gemm_tma_ext",
        sources=[f"{ROOT}/csrc/gemm_tma.cu", f"{ROOT}/csrc/gemm_coop.cu"],
        extra_include_paths=[f"{ROOT}/csrc", f"{ROOT}/third_party/cutlass/include", f"{ROOT}/third_party/cutlass/tools/util/include"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", f"-L{CUDA}/lib64/stubs"],
        build_directory=f"{ROOT}/build/ext",
        verbose=True,
    )


def load_tp4():
    """Phase-4 module (csrc/gemm_tp4.cu): TP-t down-proj -> RMSNorm -> QKV with CTA-pipelined NVLS reduction + multimem probes."""
    os.makedirs(f"{ROOT}/build/ext_tp4", exist_ok=True)
    return _load(
        name="ctapp_tp4_ext",
        sources=[f"{ROOT}/csrc/gemm_tp4.cu"],
        extra_include_paths=[f"{ROOT}/csrc", f"{ROOT}/third_party/cutlass/include", f"{ROOT}/third_party/cutlass/tools/util/include"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG", "-Xptxas", "-v"],   # -v: register/spill report in the build log
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", f"-L{CUDA}/lib64/stubs"],
        build_directory=f"{ROOT}/build/ext_tp4",
        verbose=True,
    )
