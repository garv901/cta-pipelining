import os
import torch
CUDA = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")  # must match torch's CUDA build
os.environ["PATH"] = f"{CUDA}/bin:" + os.environ["PATH"]
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
from torch.utils.cpp_extension import load as _load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_TV = torch.__version__.replace("+", "_")
BD = f"{ROOT}/build/ext_{_TV}"
BD4 = f"{ROOT}/build/ext_tp4_{_TV}"


def load():
    os.makedirs(BD, exist_ok=True)
    return _load(
        name="gemm_tma_ext",
        sources=[f"{ROOT}/csrc/gemm_tma.cu", f"{ROOT}/csrc/gemm_coop.cu"],
        extra_include_paths=[f"{ROOT}/csrc", f"{ROOT}/third_party/cutlass/include", f"{ROOT}/third_party/cutlass/tools/util/include"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", f"-L{CUDA}/lib64/stubs"],
        build_directory=BD,
        verbose=True,
    )


def load_tp4():
    """Phase-4 module (csrc/gemm_tp4.cu): TP-t down-proj -> RMSNorm -> QKV with CTA-pipelined NVLS reduction + multimem probes."""
    os.makedirs(BD4, exist_ok=True)
    return _load(
        name="ctapp_tp4_ext",
        sources=[f"{ROOT}/csrc/gemm_tp4.cu"],
        extra_include_paths=[f"{ROOT}/csrc", f"{ROOT}/third_party/cutlass/include", f"{ROOT}/third_party/cutlass/tools/util/include"],
        # -v: register/spill report in the build log. CUTLASS_ENABLE_GDC_FOR_SM90: compile the griddepcontrol (PDL) PTX into the GEMMs
        # (cutlass/arch/grid_dependency_control.h); a no-op for kernels launched without the PDL attribute.
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-DNDEBUG", "-DCUTLASS_ENABLE_GDC_FOR_SM90=1", "-Xptxas", "-v"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=["-lcuda", f"-L{CUDA}/lib64/stubs"],
        build_directory=BD4,
        verbose=True,
    )
