"""Llama with both TP all-reduce boundaries of each layer replaced by ctapp.tp4.CtappBoundary (CTA-pipelined NVLS reduction):
  A: o_proj -> all-reduce -> residual add -> post_attention_layernorm -> gate_up_proj   (same layer)
  B: down_proj -> all-reduce -> residual add -> next input_layernorm -> next qkv_proj   (layers 0..78 -> 1..79)
Env CTAPP_BOUNDARY = both (default) | down (B only) | oproj (A only).
Env CTAPP_PROTO = v5 (default; Phase 6 panel-ownership unicast protocol, ONE instance per boundary sized CTAPP_MAX_M serves
every M <= CTAPP_MAX_M with M % 128 == 0) | v3 (Phase 5 multimem protocol, one instance per M).
v5 only: CTAPP_FUSION = none | pdl (default) | pdl1 | role; CTAPP_STEAL = 0 | 1 (unset: A 1, B 0; set: both boundaries);
CTAPP_QKV_TILE = 256 (default) | 128 (boundary B consumer tile; A always 256); CTAPP_R default 4 (v3: per-M rule).
CTAPP_MAX_M default 8192 (larger M -> stock path). Eager only (enforce_eager): no CUDA graphs.
RMSNorm gammas are folded IN PLACE into the consumer weights (gate_up, qkv) at setup, so every stock-path norm in this
forward uses a ones-weight RMSNorm (self._ones_norm). Call model._refold() (sets _folded=False) after re-initialising weights."""
import copy
import logging
import os
from itertools import islice

import torch

from vllm.distributed import get_pp_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size, get_tp_group
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel

logger = logging.getLogger("ctapp_vllm")


class CtappLlamaModel(LlamaModel):
    def __init__(self, *, vllm_config, prefix="", layer_type=LlamaDecoderLayer):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        self._ctapp_enabled = (get_tensor_model_parallel_world_size() == 4 and get_pp_group().world_size == 1
                               and self.config.hidden_size == 8192 and abs(self.config.rms_norm_eps - 1e-5) < 1e-12)
        self._proto = os.environ.get("CTAPP_PROTO", "v5")
        assert self._proto in ("v3", "v5"), self._proto
        self._max_M = int(os.environ.get("CTAPP_MAX_M", "8192"))
        self._min_M = int(os.environ.get("CTAPP_MIN_M", "512"))
        self._R = int(os.environ["CTAPP_R"]) if os.environ.get("CTAPP_R") else None
        self._check = os.environ.get("CTAPP_CHECK", "0") == "1"
        self._mode = os.environ.get("CTAPP_BOUNDARY", "both")
        assert self._mode in ("both", "down", "oproj"), self._mode
        self._pool = None      # boundary B
        self._pool_a = None    # boundary A
        self._folded = False
        self._printed_ck = False
        self._seen_M = set()
        if not self._ctapp_enabled:
            logger.warning("ctapp_vllm: boundary disabled for this model/config (needs TP4, PP1, hidden 8192, eps 1e-5)")

    # -- lazy setup (after weights are loaded)
    def _setup(self, device):
        from ctapp.tp4 import CtappBoundaryPool
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        self._fold(layers)
        l0 = layers[0]
        Kr, N2r = l0.mlp.down_proj.weight.shape[1], l0.self_attn.qkv_proj.weight.shape[0]
        KrA, N2rA = l0.self_attn.o_proj.weight.shape[1], l0.mlp.gate_up_proj.weight.shape[0]
        gn = get_tp_group().device_group.group_name
        rk, ws = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        kwa, kwb = {}, {}
        b_tile = 256
        if self._proto == "v5":   # Phase 6 S3 recommendation: B = pdl, tile 256, no steal; A = pdl + steal (tail auto, R=4 both)
            fusion = os.environ.get("CTAPP_FUSION", "pdl")
            st = os.environ.get("CTAPP_STEAL")
            b_tile = int(os.environ.get("CTAPP_QKV_TILE", "256"))
            kwb = dict(protocol="v5", max_M=self._max_M, fusion=fusion, qkv_tile=b_tile, steal=bool(int(st)) if st else False)
            kwa = dict(protocol="v5", max_M=self._max_M, fusion=fusion, qkv_tile=256, steal=bool(int(st)) if st else True)
        self._pool = CtappBoundaryPool(Kr, N2r, rk, ws, gn, device, R=self._R, min_M=self._min_M, max_instances=6,
                                       qkv_raster=int(os.environ.get("CTAPP_B_RASTER", 2)),
                                       qkv_swizzle=int(os.environ.get("CTAPP_B_SWIZZLE", 2 if b_tile == 128 else 1)), **kwb)
        self._pool_a = CtappBoundaryPool(KrA, N2rA, rk, ws, gn, device, R=self._R, min_M=self._min_M, max_instances=6,
                                         qkv_raster=int(os.environ.get("CTAPP_A_RASTER", 1)),
                                         qkv_swizzle=int(os.environ.get("CTAPP_A_SWIZZLE", 2)), **kwa)
        logger.warning("ctapp_vllm rank %d: setup done (mode=%s proto=%s max_M=%d B: Kr=%d N2r=%d %s, A: Kr=%d N2r=%d %s, R=%s group=%s)",
                       rk, self._mode, self._proto, self._max_M, Kr, N2r, kwb, KrA, N2rA, kwa, self._R, gn)

    @torch.no_grad()
    def _fold(self, layers=None):
        """Fold gammas in place: gate_up *= post_attention_layernorm.weight, qkv *= input_layernorm.weight (fp32 math, bf16
        store, chunked over rows). Also (re)builds the ones-weight norm used by all stock-path norms. Must run exactly once
        per weight (re-)initialisation."""
        layers = layers or list(islice(self.layers, self.start_layer, self.end_layer))
        for l in layers:
            for W, gam in ((l.mlp.gate_up_proj.weight, l.post_attention_layernorm.weight),
                           (l.self_attn.qkv_proj.weight, l.input_layernorm.weight)):
                g32 = gam.float()[None, :]
                for r0 in range(0, W.shape[0], 2048):
                    W[r0:r0 + 2048] = (W[r0:r0 + 2048].float() * g32).to(W.dtype)
        if getattr(self, "_ones_norm", None) is None:
            on = copy.deepcopy(layers[0].input_layernorm)
            for p_ in on.parameters():
                p_.requires_grad_(False)
            object.__setattr__(self, "_ones_norm", on)    # bypass nn.Module registration (not a checkpoint weight)
        self._ones_norm.weight.fill_(1.0)
        self._folded = True

    def _refold(self):
        self._folded = False

    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds=None, **extra_layer_kwargs):
        if not self._ctapp_enabled:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **extra_layer_kwargs)
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        M = hidden_states.shape[0]
        if self._pool is None:
            self._setup(hidden_states.device)
        elif not self._folded:
            self._fold()
        on = self._ones_norm
        use_a, use_b = self._mode in ("both", "oproj"), self._mode in ("both", "down")
        rank = get_tensor_model_parallel_rank()
        if self._check and rank == 0 and not self._printed_ck:
            w = self.layers[self.start_layer].mlp.down_proj.weight
            logger.warning("CTAPP_CKSUM layers[0].down_proj.weight rank0 shape=%s sum=%.6e abssum=%.6e", tuple(w.shape),
                           w.double().sum().item(), w.double().abs().sum().item())
            self._printed_ck = True
        # Instance creation is collective and allocates device memory (implicit device syncs). Quiesce every rank's GPU around
        # each creation so no spinning reducer of an earlier forward is outstanding while a peer is still in (slow) creation
        # (observed once as cudaErrorIllegalAddress at the first M=12288 forward without this). Order A then B on all ranks.
        pools = {}
        for name, pool, use in (("A", self._pool_a, use_a), ("B", self._pool, use_b)):
            if not use or M > self._max_M:
                pools[name] = None
                continue
            ok_M = M % 128 == 0 and M >= self._min_M
            new_M = ok_M and (not pool._inst if self._proto == "v5" else M not in pool._inst)   # v5: one max_M instance
            if new_M:
                torch.cuda.synchronize()
                get_tp_group().barrier()
            pools[name] = pool.get(M)
            if new_M:
                torch.cuda.synchronize()
                get_tp_group().barrier()
                if rank == 0:
                    logger.warning("ctapp_vllm created boundary %s instance for M=%d (proto %s, max_M %d)", name, M, self._proto,
                                   self._max_M)
        ba, b = pools["A"], pools["B"]
        if rank == 0 and (M not in self._seen_M or os.environ.get("CTAPP_LOG_M") == "1"):
            self._seen_M.add(M)
            logger.warning("ctapp_vllm forward M=%d -> A:%s B:%s", M, ba is not None, b is not None)
        if self._check and rank == 0 and ba is None and b is None:
            logger.warning("CTAPP_CHECK M=%d -> stock fallback (no boundary)", M)

        def rel(x, y):
            x, y = x.float(), y.float()
            return ((x - y).norm() / y.norm()).item()

        qkv_pre = None
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            attn = layer.self_attn
            if qkv_pre is None:
                if residual is None:
                    residual = hidden_states
                    h = on(hidden_states)
                else:
                    h, residual = on(hidden_states, residual)
                qkv, _ = attn.qkv_proj(h)
            else:
                qkv = qkv_pre            # residual was already set to boundary B's x()
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q, k = attn.rotary_emb(positions, q, k)
            a = attn.attn(q, k, v)
            mlp = layer.mlp
            if ba is not None:
                if not a.is_contiguous():
                    a = a.contiguous()
                ref = None
                if self._check:   # stock-path result first (boundary overwrites buffers); same collectives on all ranks
                    hh, _ = attn.o_proj(a)
                    hh, res2 = on(hh, residual.clone())
                    g_ref, _ = mlp.gate_up_proj(hh)
                    ref = (g_ref, res2)
                g = ba.forward(a, residual, Wd=attn.o_proj.weight, Wq=mlp.gate_up_proj.weight)
                residual = ba.x()
                if ref is not None:
                    e_g, e_x = rel(g, ref[0]), rel(residual, ref[1])
                    if rank == 0:
                        logger.warning("CTAPP_CHECK M=%d layer %d A: g rel err %.3e  x rel err %.3e", M, i, e_g, e_x)
            else:
                hidden_states, _ = attn.o_proj(a)                      # stock all-reduce
                hidden_states, residual = on(hidden_states, residual)
                g, _ = mlp.gate_up_proj(hidden_states)
            h_r = maybe_fused_act_quant(mlp.act_fn, g, mlp.down_proj)
            if b is not None and nxt is not None:
                ref = None
                if self._check:
                    y, _ = mlp.down_proj(h_r)
                    h2, res2 = on(y, residual.clone())
                    qkv_ref, _ = nxt.self_attn.qkv_proj(h2)
                    ref = (qkv_ref, res2)
                qkv_pre = b.forward(h_r, residual, Wd=mlp.down_proj.weight, Wq=nxt.self_attn.qkv_proj.weight)
                residual = b.x()
                hidden_states = None
                if ref is not None:
                    e_q, e_x = rel(qkv_pre, ref[0]), rel(residual, ref[1])
                    if rank == 0:
                        logger.warning("CTAPP_CHECK M=%d layer %d->%d B: qkv rel err %.3e  x rel err %.3e", M, i, i + 1, e_q, e_x)
            else:
                hidden_states, _ = mlp.down_proj(h_r)
                qkv_pre = None
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class CtappLlamaForCausalLM(LlamaForCausalLM):
    def _init_model(self, vllm_config, prefix="", layer_type=LlamaDecoderLayer):
        return CtappLlamaModel(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
