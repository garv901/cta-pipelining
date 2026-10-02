"""vLLM general plugin. Enable with CTAPP_VLLM=1 (default: off, stock vLLM)."""
import logging
import os
import sys

logger = logging.getLogger("ctapp_vllm")
_logged = False


def register():
    global _logged
    on = os.environ.get("CTAPP_VLLM", "0") == "1"
    if not _logged:
        _logged = True
        logger.warning("ctapp_vllm plugin loaded in pid %d: CTAPP_VLLM=%s -> %s", os.getpid(),
                       os.environ.get("CTAPP_VLLM", "0"), "CTA-pipelined boundary ENABLED" if on else "stock model (disabled)")
    if not on:
        return
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.environ.setdefault("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES", "0")
    from vllm.model_executor.models.registry import ModelRegistry
    ModelRegistry.register_model("LlamaForCausalLM", "ctapp_vllm.model:CtappLlamaForCausalLM")
