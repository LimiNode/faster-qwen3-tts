"""Prefill compile compatibility islands for product-shaped opt-in modes."""

from __future__ import annotations

import types
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa

import qwen_tts.core.models.modeling_qwen3_tts as qwen_modeling


SUPPORTED_PREFILL_COMPILE_COMPAT_MODES = {
    "none",
    "strict_bf16_sdpa_v1",
}
PREFILL_COMPILE_COMPAT_METADATA_VERSION = 1


@torch.library.custom_op("faster_qwen3_tts::strict_add", mutates_args=())
def _strict_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left + right


@_strict_add.register_fake
def _(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(left)


@torch.library.custom_op("faster_qwen3_tts::strict_rmsnorm", mutates_args=())
def _strict_rmsnorm(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = tensor.dtype
    values = tensor.to(torch.float32)
    variance = values.pow(2).mean(-1, keepdim=True)
    values = values * torch.rsqrt(variance + eps)
    return weight * values.to(input_dtype)


@_strict_rmsnorm.register_fake
def _(tensor: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(tensor)


@torch.library.custom_op("faster_qwen3_tts::strict_mul", mutates_args=())
def _strict_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return left * right


@_strict_mul.register_fake
def _(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(left)


@torch.library.custom_op("faster_qwen3_tts::strict_sdpa", mutates_args=())
def _strict_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    is_causal: bool,
    enable_gqa: bool,
) -> torch.Tensor:
    sdpa_kwargs = {}
    key_states = key
    value_states = value
    if enable_gqa:
        sdpa_kwargs["enable_gqa"] = True
    else:
        key_states = qwen_modeling.repeat_kv(key, num_key_value_groups)
        value_states = qwen_modeling.repeat_kv(value, num_key_value_groups)
    output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=None,
        dropout_p=0.0,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    return output.transpose(1, 2).contiguous()


@_strict_sdpa.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    num_key_value_groups: int,
    is_causal: bool,
    enable_gqa: bool,
) -> torch.Tensor:
    return torch.empty(
        (query.shape[0], query.shape[2], query.shape[1], query.shape[3]),
        dtype=query.dtype,
        device=query.device,
    )


def normalize_prefill_compile_compat_mode(mode: str) -> str:
    normalized = str(mode or "none").strip().lower()
    if normalized not in SUPPORTED_PREFILL_COMPILE_COMPAT_MODES:
        raise ValueError(
            f"Unsupported prefill_compile_compat_mode {mode!r}. "
            f"Expected one of {sorted(SUPPORTED_PREFILL_COMPILE_COMPAT_MODES)}."
        )
    return normalized


def configure_prefill_compile_compat(talker: Any, mode: str) -> dict[str, Any]:
    """Declare and, when requested, install immutable Talker compat mode."""
    mode = normalize_prefill_compile_compat_mode(mode)
    declared = getattr(
        talker,
        "_faster_qwen3_tts_prefill_compile_compat_declared_mode",
        None,
    )
    if declared is not None and declared != mode:
        raise RuntimeError(
            "Talker prefill compile compatibility mode is immutable after load: "
            f"{declared!r} != {mode!r}"
        )
    if declared == mode:
        return prefill_compile_compat_metadata(talker)

    if mode == "none":
        talker._faster_qwen3_tts_prefill_compile_compat_declared_mode = mode
        talker._faster_qwen3_tts_prefill_compile_compat_mode = mode
        talker._faster_qwen3_tts_prefill_compile_compat_patched_modules = {}
        talker._faster_qwen3_tts_prefill_compile_compat_validated_modules = {}
        return prefill_compile_compat_metadata(talker)

    targets = _collect_prefill_compile_compat_targets(talker)
    patched = _validated_prefill_compile_compat_counts(targets)
    talker._faster_qwen3_tts_prefill_compile_compat_declared_mode = mode
    talker._faster_qwen3_tts_prefill_compile_compat_mode = mode
    talker._faster_qwen3_tts_prefill_compile_compat_patched_modules = {}
    talker._faster_qwen3_tts_prefill_compile_compat_validated_modules = dict(patched)
    return prefill_compile_compat_metadata(talker)


def ensure_prefill_compile_compat(talker: Any, requested_mode: str) -> dict[str, Any]:
    """Validate that a request uses the Talker's immutable compat mode."""
    requested_mode = normalize_prefill_compile_compat_mode(requested_mode)
    declared = getattr(
        talker,
        "_faster_qwen3_tts_prefill_compile_compat_declared_mode",
        None,
    )
    if declared is not None and declared != requested_mode:
        raise RuntimeError(
            "Requested prefill compile compatibility mode does not match the "
            f"loaded model: requested {requested_mode!r}, loaded {declared!r}."
        )
    if declared is not None:
        return prefill_compile_compat_metadata(talker)

    # Low-level diagnostic entry points may operate on an unconfigured Talker.
    # Keep them lazy, but never allow a patched Talker to pretend it is "none".
    applied = getattr(talker, "_faster_qwen3_tts_prefill_compile_compat_mode", None)
    if applied not in (None, requested_mode):
        raise RuntimeError(
            "Talker already has a different prefill compile compatibility mode: "
            f"{applied!r} != {requested_mode!r}"
        )
    if requested_mode == "none":
        return {
            "prefill_compile_compat_mode": requested_mode,
            "prefill_compile_compat_applied": False,
            "prefill_compile_compat_reused": False,
            "prefill_compile_compat_patched_modules": {},
        }
    return apply_prefill_compile_compat(talker, requested_mode)


def prefill_compile_compat_metadata(talker: Any) -> dict[str, Any]:
    declared = getattr(
        talker,
        "_faster_qwen3_tts_prefill_compile_compat_declared_mode",
        None,
    )
    mode = getattr(
        talker,
        "_faster_qwen3_tts_prefill_compile_compat_mode",
        declared or "none",
    )
    patched = dict(
        getattr(
            talker,
            "_faster_qwen3_tts_prefill_compile_compat_patched_modules",
            {},
        )
    )
    validated = dict(
        getattr(
            talker,
            "_faster_qwen3_tts_prefill_compile_compat_validated_modules",
            patched,
        )
    )
    return {
        "prefill_compile_compat_metadata_version": (
            PREFILL_COMPILE_COMPAT_METADATA_VERSION
        ),
        "prefill_compile_compat_declared_mode": (
            normalize_prefill_compile_compat_mode(declared)
            if declared is not None
            else None
        ),
        "prefill_compile_compat_mode": normalize_prefill_compile_compat_mode(mode),
        "prefill_compile_compat_applied": bool(patched),
        "prefill_compile_compat_reused": bool(patched),
        "prefill_compile_compat_patched_modules": patched,
        "prefill_compile_compat_validated_modules": validated,
    }


@contextmanager
def prefill_compile_compat_context(
    talker: Any,
    mode: str,
) -> Iterator[dict[str, Any]]:
    """Temporarily install strict forwards for a compiled prefill call."""
    mode = normalize_prefill_compile_compat_mode(mode)
    if mode == "none":
        yield prefill_compile_compat_metadata(talker)
        return

    declared = getattr(
        talker,
        "_faster_qwen3_tts_prefill_compile_compat_declared_mode",
        None,
    )
    if declared is not None and declared != mode:
        raise RuntimeError(
            "Requested prefill compile compatibility mode does not match the "
            f"loaded model: requested {mode!r}, loaded {declared!r}."
        )
    if getattr(talker, "_faster_qwen3_tts_prefill_compile_compat_active", False):
        yield prefill_compile_compat_metadata(talker)
        return

    targets = _collect_prefill_compile_compat_targets(talker)
    patched = _validated_prefill_compile_compat_counts(targets)
    originals = []
    try:
        for module in targets["rmsnorm"]:
            originals.append((module, module.forward))
            module.forward = types.MethodType(_strict_rmsnorm_forward, module)
        for module in targets["mlp"]:
            originals.append((module, module.forward))
            module.forward = types.MethodType(_strict_mlp_forward, module)
        for module in targets["attention"]:
            originals.append((module, module.forward))
            module.forward = types.MethodType(_strict_attention_forward, module)

        if declared is None:
            talker._faster_qwen3_tts_prefill_compile_compat_declared_mode = mode
        talker._faster_qwen3_tts_prefill_compile_compat_mode = mode
        talker._faster_qwen3_tts_prefill_compile_compat_patched_modules = dict(patched)
        talker._faster_qwen3_tts_prefill_compile_compat_validated_modules = dict(patched)
        talker._faster_qwen3_tts_prefill_compile_compat_active = True
        yield prefill_compile_compat_metadata(talker)
    finally:
        for module, original_forward in reversed(originals):
            module.forward = original_forward
        talker._faster_qwen3_tts_prefill_compile_compat_patched_modules = {}
        talker._faster_qwen3_tts_prefill_compile_compat_active = False


def apply_prefill_compile_compat(talker: Any, mode: str) -> dict[str, Any]:
    """Install instance-local compatibility forwards for compiled prefill."""
    mode = normalize_prefill_compile_compat_mode(mode)
    if mode == "none":
        return {
            "prefill_compile_compat_mode": mode,
            "prefill_compile_compat_applied": False,
            "prefill_compile_compat_reused": False,
            "prefill_compile_compat_patched_modules": {},
        }

    if getattr(talker, "_faster_qwen3_tts_prefill_compile_compat_mode", None) == mode:
        return prefill_compile_compat_metadata(talker)

    if getattr(talker, "_faster_qwen3_tts_prefill_compile_compat_mode", None) not in (None, mode):
        raise RuntimeError("Talker already has a different prefill compile compatibility mode")

    targets = _collect_prefill_compile_compat_targets(talker)
    patched = _validated_prefill_compile_compat_counts(targets)

    for module in targets["rmsnorm"]:
        module.forward = types.MethodType(_strict_rmsnorm_forward, module)
    for module in targets["mlp"]:
        module.forward = types.MethodType(_strict_mlp_forward, module)
    for module in targets["attention"]:
        module.forward = types.MethodType(_strict_attention_forward, module)

    talker._faster_qwen3_tts_prefill_compile_compat_mode = mode
    talker._faster_qwen3_tts_prefill_compile_compat_patched_modules = dict(patched)
    talker._faster_qwen3_tts_prefill_compile_compat_validated_modules = dict(patched)
    return {
        "prefill_compile_compat_mode": mode,
        "prefill_compile_compat_applied": True,
        "prefill_compile_compat_reused": False,
        "prefill_compile_compat_patched_modules": dict(patched),
        "prefill_compile_compat_validated_modules": dict(patched),
    }


def _collect_prefill_compile_compat_targets(talker: Any) -> dict[str, list[Any]]:
    targets: dict[str, list[Any]] = {"rmsnorm": [], "mlp": [], "attention": []}
    for module in talker.modules():
        class_name = type(module).__name__
        if class_name == "Qwen3TTSRMSNorm":
            targets["rmsnorm"].append(module)
        elif class_name == "Qwen3TTSTalkerTextMLP":
            targets["mlp"].append(module)
        elif class_name in {"Qwen3TTSAttention", "Qwen3TTSTalkerAttention"}:
            targets["attention"].append(module)
    return targets


def _validated_prefill_compile_compat_counts(
    targets: dict[str, list[Any]],
) -> dict[str, int]:
    patched = {name: len(modules) for name, modules in targets.items()}
    if patched["rmsnorm"] == 0 or patched["mlp"] == 0 or patched["attention"] == 0:
        raise RuntimeError(f"Incomplete prefill compile compatibility patch: {patched!r}")
    return patched


def validate_strict_bf16_sdpa_v1(
    *,
    prefill_backend: str,
    prefill_mask_mode: str,
    talker_input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    input_metadata: Optional[dict],
) -> None:
    if prefill_backend not in {"compile_inductor_default", "compile_reduce_overhead"}:
        raise ValueError(
            "strict_bf16_sdpa_v1 requires prefill_backend in "
            "{'compile_inductor_default', 'compile_reduce_overhead'}."
        )
    if prefill_mask_mode != "skip":
        raise ValueError("strict_bf16_sdpa_v1 requires verified prefill mask skip.")
    if talker_input_embeds.dtype != torch.bfloat16:
        raise ValueError("strict_bf16_sdpa_v1 requires bfloat16 prefill inputs.")
    if talker_input_embeds.shape[0] != 1:
        raise ValueError("strict_bf16_sdpa_v1 requires batch size 1.")
    if attention_mask is not None:
        raise ValueError("strict_bf16_sdpa_v1 requires absent attention_mask after verified skip.")
    metadata = input_metadata or {}
    if metadata.get("prefill_attn_implementation") != "sdpa":
        raise ValueError("strict_bf16_sdpa_v1 requires SDPA attention metadata.")
    if metadata.get("prefill_has_sliding_window") is True:
        raise ValueError("strict_bf16_sdpa_v1 does not support sliding-window prefill.")


def _strict_rmsnorm_forward(self, hidden_states):
    return _strict_rmsnorm(hidden_states, self.weight, self.variance_epsilon)


def _strict_mlp_forward(self, x):
    return self.down_proj(_strict_mul(self.act_fn(self.gate_proj(x)), self.up_proj(x)))


def _strict_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if attention_mask is not None:
        return _call_original_attention(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )
    if self.training or kwargs.get("output_attentions", False):
        return _call_original_attention(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _strict_apply_rope(self, query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    is_causal = query_states.shape[2] > 1 and getattr(self, "is_causal", True)
    attn_output = _strict_sdpa(
        query_states,
        key_states,
        value_states,
        self.scaling,
        self.num_key_value_groups,
        is_causal,
        use_gqa_in_sdpa(attention_mask, key_states),
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def _call_original_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    if hasattr(self, "rope_scaling"):
        query_rope = qwen_modeling.apply_multimodal_rotary_pos_emb
    else:
        query_rope = qwen_modeling.apply_rotary_pos_emb
    raise RuntimeError(
        "strict_bf16_sdpa_v1 attention fallback is unsupported in compiled prefill; "
        f"attention_mask={attention_mask is not None}, training={self.training}, rope={query_rope.__name__}."
    )


def _strict_apply_rope(self, q, k, cos, sin):
    if hasattr(self, "rope_scaling"):
        return _strict_apply_multimodal_rope(
            q,
            k,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
            self.rope_scaling["interleaved"],
        )
    return _strict_apply_plain_rope(q, k, cos, sin)


def _strict_apply_plain_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _strict_add(q * cos, qwen_modeling.rotate_half(q) * sin)
    k_embed = _strict_add(k * cos, qwen_modeling.rotate_half(k) * sin)
    return q_embed, k_embed


def _strict_apply_multimodal_rope(
    q,
    k,
    cos,
    sin,
    mrope_section,
    mrope_interleaved=False,
    unsqueeze_dim=1,
):
    if mrope_interleaved:
        dim = cos.shape[-1]
        modality_count = len(mrope_section)
        cos_half = _apply_interleaved_rope(cos[..., : dim // 2], mrope_section, modality_count)
        sin_half = _apply_interleaved_rope(sin[..., : dim // 2], mrope_section, modality_count)
        cos = torch.cat([cos_half] * 2, dim=-1).unsqueeze(unsqueeze_dim)
        sin = torch.cat([sin_half] * 2, dim=-1).unsqueeze(unsqueeze_dim)
    else:
        sections = mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)
    q_embed = _strict_add(q * cos, qwen_modeling.rotate_half(q) * sin)
    k_embed = _strict_add(k * cos, qwen_modeling.rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_interleaved_rope(x, mrope_section, modality_count):
    x_t = x[0].clone()
    index_ranges = []
    for index, section in enumerate(mrope_section[1:], 1):
        index_ranges.append((index, section * modality_count))
    for begin, end in index_ranges:
        x_t[..., begin:end:modality_count] = x[begin, ..., begin:end:modality_count]
    return x_t
