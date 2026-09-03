"""Opt-in numerical diagnostics and CMP 50HX precision boundaries.

The default import is a strict no-op. ``QTB_FASTER_EAGER_DIAGNOSTIC=1``
deliberately bypasses internal CUDA Graph replay and is therefore unsuitable
for performance comparison. The graph-compatible precision controls remain
separate explicit opt-ins so production callers cannot activate a numerical
policy accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_ENV_ENABLED = "QTB_FASTER_EAGER_DIAGNOSTIC"
_ENV_TRACE = "QTB_FASTER_DIAGNOSTIC_TRACE_PATH"
_ENV_MLP_FP32_ISLAND = "QTB_FASTER_MLP_FP32_ISLAND"
_ENV_RESIDUAL_CARRIER_FP32 = "QTB_FASTER_RESIDUAL_CARRIER_FP32"
_ENV_GRAPH_RESIDUAL_CARRIER_FP32 = "QTB_FASTER_GRAPH_RESIDUAL_CARRIER_FP32"
_ENV_GRAPH_CARRIER_PROOF = "QTB_FASTER_GRAPH_CARRIER_PROOF_PATH"
_ENV_MLP_NARROW_GATE_UP_FP16 = "QTB_FASTER_MLP_NARROW_GATE_UP_FP16"
_ENV_MLP_FUSED_GATE_UP = "QTB_FASTER_MLP_FUSED_GATE_UP"
_ENV_GRAPH_FINITE_CHECKER = "QTB_FASTER_GRAPH_FINITE_CHECKER"
_ENV_GRAPH_FINITE_PROOF = "QTB_FASTER_GRAPH_FINITE_PROOF_PATH"
_ENV_START_REQUEST = "QTB_FASTER_DIAGNOSTIC_START_REQUEST"
_installed = False
_active_predictor: Any | None = None
_active_graph_finite_predictor: Any | None = None
_request_index = 0


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLED) == "1"


def _mlp_fp32_island_enabled() -> bool:
    return os.environ.get(_ENV_MLP_FP32_ISLAND) == "1"


def _residual_carrier_fp32_enabled() -> bool:
    return os.environ.get(_ENV_RESIDUAL_CARRIER_FP32) == "1"


def _graph_residual_carrier_fp32_enabled() -> bool:
    return os.environ.get(_ENV_GRAPH_RESIDUAL_CARRIER_FP32) == "1"


def _mlp_narrow_gate_up_fp16_enabled() -> bool:
    return os.environ.get(_ENV_MLP_NARROW_GATE_UP_FP16) == "1"


def _mlp_fused_gate_up_enabled() -> bool:
    return os.environ.get(_ENV_MLP_FUSED_GATE_UP) == "1"


def _graph_finite_checker_enabled() -> bool:
    return os.environ.get(_ENV_GRAPH_FINITE_CHECKER) == "1"


def _append_graph_carrier_proof(value: dict[str, Any]) -> None:
    """Write one capture-time provenance record, never a request-time trace."""

    proof_path = os.environ.get(_ENV_GRAPH_CARRIER_PROOF)
    if not proof_path:
        return
    with Path(proof_path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _append_graph_finite_proof(value: dict[str, Any]) -> None:
    """Write one compact request-level finite-check result after completion."""

    proof_path = os.environ.get(_ENV_GRAPH_FINITE_PROOF)
    if not proof_path:
        return
    with Path(proof_path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


_GRAPH_FINITE_COMPONENTS = (
    "layer2_gate_fp16",
    "layer2_up_fp16",
    "layer2_gate_fp32",
    "layer2_up_fp32",
    "layer2_silu_times_up_fp32",
    "layer2_down_fp32",
    "layer2_output_fp16",
    "layer2_residual_fp32",
    "rmsnorm_fp32",
    "normalized_branch_fp16",
    "predictor_logits",
    "predictor_probabilities",
)
_GRAPH_FINITE_COMPONENT_INDEX = {
    component: index for index, component in enumerate(_GRAPH_FINITE_COMPONENTS)
}


def _install_graph_finite_checker(owner: Any) -> None:
    """Allocate replay-safe, device-resident finite aggregates for one predictor.

    The observation operations are inserted while FasterQwen captures the
    predictor graph.  Replays therefore update only GPU tensors: they do not
    write traces, inspect values on the host, or allocate request-time buffers.
    """

    if getattr(owner, "_cmp50hx_graph_finite_checker_installed", False):
        return
    component_count = len(_GRAPH_FINITE_COMPONENTS)
    owner._cmp50hx_graph_finite_flags = torch.zeros(
        component_count, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_anomaly_flags = torch.zeros(
        component_count, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_counts = torch.zeros(
        component_count, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_max_abs = torch.zeros(
        component_count, dtype=torch.float32, device=owner.device
    )
    owner._cmp50hx_graph_finite_component_indices = torch.arange(
        component_count, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_replay_index = torch.zeros(
        (), dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_first_component = torch.full(
        (), -1, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_first_replay = torch.full(
        (), -1, dtype=torch.int32, device=owner.device
    )
    owner._cmp50hx_graph_finite_checker_installed = True


def _record_graph_checker_anomaly(
    owner: Any,
    component: str,
    anomaly: torch.Tensor,
) -> None:
    """Latch the first observed invalid boundary entirely on the GPU."""

    index = _GRAPH_FINITE_COMPONENT_INDEX[component]
    owner._cmp50hx_graph_finite_anomaly_flags[index].bitwise_or_(
        anomaly.to(torch.int32)
    )
    first_component = owner._cmp50hx_graph_finite_first_component
    record_first = torch.logical_and(anomaly, first_component.lt(0))
    component_index = owner._cmp50hx_graph_finite_component_indices[index]
    torch.where(record_first, component_index, first_component, out=first_component)
    first_replay = owner._cmp50hx_graph_finite_first_replay
    torch.where(
        record_first,
        owner._cmp50hx_graph_finite_replay_index,
        first_replay,
        out=first_replay,
    )


def _observe_graph_finite(owner: Any, component: str, value: torch.Tensor) -> None:
    """Record a finite flag, count, and maximum magnitude entirely on device."""

    if not getattr(owner, "_cmp50hx_graph_finite_checker_installed", False):
        return
    index = _GRAPH_FINITE_COMPONENT_INDEX[component]
    flags = owner._cmp50hx_graph_finite_flags[index]
    counts = owner._cmp50hx_graph_finite_counts[index]
    max_abs = owner._cmp50hx_graph_finite_max_abs[index]
    nonfinite = torch.logical_not(torch.isfinite(value).all())
    flags.bitwise_or_(nonfinite.to(torch.int32))
    _record_graph_checker_anomaly(owner, component, nonfinite)
    counts.add_(1)
    observed_max = torch.nan_to_num(
        value.abs(), nan=0.0, posinf=float("inf"), neginf=float("inf")
    ).amax()
    torch.maximum(max_abs, observed_max, out=max_abs)


def _reset_graph_finite_checker(owner: Any) -> None:
    """Reset request aggregates without synchronising with the GPU."""

    owner._cmp50hx_graph_finite_flags.zero_()
    owner._cmp50hx_graph_finite_anomaly_flags.zero_()
    owner._cmp50hx_graph_finite_counts.zero_()
    owner._cmp50hx_graph_finite_max_abs.zero_()
    owner._cmp50hx_graph_finite_replay_index.zero_()
    owner._cmp50hx_graph_finite_first_component.fill_(-1)
    owner._cmp50hx_graph_finite_first_replay.fill_(-1)


def _finalize_graph_finite_checker(owner: Any) -> None:
    """Synchronise once at request end and emit the checker proof."""

    flags = owner._cmp50hx_graph_finite_flags.detach().cpu().tolist()
    anomaly_flags = owner._cmp50hx_graph_finite_anomaly_flags.detach().cpu().tolist()
    counts = owner._cmp50hx_graph_finite_counts.detach().cpu().tolist()
    max_abs = owner._cmp50hx_graph_finite_max_abs.detach().cpu().tolist()
    components = {
        component: {
            "nonfinite_observed": bool(flags[index]),
            "anomaly_observed": bool(anomaly_flags[index]),
            "observations": int(counts[index]),
            "max_abs": float(max_abs[index]),
        }
        for index, component in enumerate(_GRAPH_FINITE_COMPONENTS)
    }
    first_component = int(owner._cmp50hx_graph_finite_first_component.detach().cpu())
    first_replay = int(owner._cmp50hx_graph_finite_first_replay.detach().cpu())
    _append_graph_finite_proof(
        {
            "event": "graph_finite_checker_request_complete",
            "request_index": int(getattr(owner, "_cmp50hx_request_index", 0)),
            "all_finite": not any(flags),
            "all_boundaries_valid": not any(anomaly_flags),
            "first_anomalous_boundary": (
                _GRAPH_FINITE_COMPONENTS[first_component]
                if first_component >= 0
                else None
            ),
            "first_anomalous_predictor_replay": first_replay if first_replay >= 0 else None,
            "observed_components": [
                component
                for index, component in enumerate(_GRAPH_FINITE_COMPONENTS)
                if counts[index]
            ],
            "components": components,
            "eager_numerical_trace": False,
            "host_syncs_during_predictor_replay": 0,
            "per_step_allocations": False,
        }
    )


def _diagnostic_start_request() -> int:
    value = os.environ.get(_ENV_START_REQUEST, "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def _install_layer2_mlp_fp32_island(owner: Any) -> None:
    """Replace only predictor layer 2 MLP with an FP32 compute island.

    The layer returns the model's original FP16 dtype, so the surrounding
    predictor remains FP16.  FP32 copies of the three weights are allocated
    before any FasterQwen CUDA graph capture and are reused by every replay.
    """

    layer_index = 2
    mlp = owner.pred_model.layers[layer_index].mlp
    if getattr(mlp, "_cmp50hx_fp32_island_installed", False):
        return

    mlp._cmp50hx_gate_weight_fp32 = mlp.gate_proj.weight.detach().float()
    mlp._cmp50hx_up_weight_fp32 = mlp.up_proj.weight.detach().float()
    mlp._cmp50hx_down_weight_fp32 = mlp.down_proj.weight.detach().float()
    down_bias = getattr(mlp.down_proj, "bias", None)
    mlp._cmp50hx_down_bias_fp32 = (
        down_bias.detach().float() if down_bias is not None else None
    )
    mlp._cmp50hx_precision_variant = "wide_gate_up_fp32"
    mlp._cmp50hx_gate_up_compute_dtype = "float32"
    mlp._cmp50hx_product_compute_dtype = "float32"
    mlp._cmp50hx_down_compute_dtype = "float32"

    def forward(x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        gate = F.linear(x_fp32, mlp._cmp50hx_gate_weight_fp32)
        up = F.linear(x_fp32, mlp._cmp50hx_up_weight_fp32)
        _observe_graph_finite(owner, "layer2_gate_fp32", gate)
        _observe_graph_finite(owner, "layer2_up_fp32", up)
        product = mlp.act_fn(gate) * up
        _observe_graph_finite(owner, "layer2_silu_times_up_fp32", product)
        down = F.linear(product, mlp._cmp50hx_down_weight_fp32, mlp._cmp50hx_down_bias_fp32)
        _observe_graph_finite(owner, "layer2_down_fp32", down)
        output = down.to(dtype=x.dtype)
        _observe_graph_finite(owner, "layer2_output_fp16", output)
        if owner._cmp50hx_diagnostic_active:
            records = owner._cmp50hx_diagnostic_records
            records.setdefault((layer_index, "gate_proj"), []).append(gate.detach())
            records.setdefault((layer_index, "up_proj"), []).append(up.detach())
            records.setdefault((layer_index, "down_proj"), []).append(down.detach())
        return output

    mlp.forward = forward
    mlp._cmp50hx_fp32_island_installed = True


def _install_layer2_mlp_narrow_gate_up_fp16(owner: Any) -> None:
    """Keep L2 gate/up GEMMs FP16 while retaining the proven FP32 tail.

    Gate and up projection outputs are copied into preallocated FP32 buffers.
    The SiLU product, down projection, residual carrier, and RMSNorm then keep
    their FP32 numerical boundary.  Fixed buffers are allocated before Faster
    captures its graph, so replay has no per-step FP32 allocation.
    """

    layer_index = 2
    mlp = owner.pred_model.layers[layer_index].mlp
    if getattr(mlp, "_cmp50hx_fp32_island_installed", False):
        return

    max_tokens = 2
    intermediate_size = mlp.gate_proj.weight.shape[0]
    hidden_size = mlp.down_proj.weight.shape[0]
    device = owner.device
    mlp._cmp50hx_down_weight_fp32 = mlp.down_proj.weight.detach().float()
    down_bias = getattr(mlp.down_proj, "bias", None)
    mlp._cmp50hx_down_bias_fp32 = (
        down_bias.detach().float() if down_bias is not None else None
    )
    mlp._cmp50hx_gate_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_up_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_product_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_down_fp32 = torch.empty(
        (max_tokens, hidden_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_output_fp16 = torch.empty(
        (1, max_tokens, hidden_size), dtype=torch.float16, device=device
    )
    mlp._cmp50hx_precision_variant = "narrow_gate_up_fp16"
    mlp._cmp50hx_gate_up_compute_dtype = "float16"
    mlp._cmp50hx_product_compute_dtype = "float32"
    mlp._cmp50hx_down_compute_dtype = "float32"

    def forward(x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        gate_fp16 = F.linear(x, mlp.gate_proj.weight)
        up_fp16 = F.linear(x, mlp.up_proj.weight)
        _observe_graph_finite(owner, "layer2_gate_fp16", gate_fp16)
        _observe_graph_finite(owner, "layer2_up_fp16", up_fp16)
        gate_fp32 = mlp._cmp50hx_gate_fp32[:sequence_length]
        up_fp32 = mlp._cmp50hx_up_fp32[:sequence_length]
        product = mlp._cmp50hx_product_fp32[:sequence_length]
        down = mlp._cmp50hx_down_fp32[:sequence_length]
        output = mlp._cmp50hx_output_fp16[:, :sequence_length, :]
        gate_fp32.copy_(gate_fp16.reshape(sequence_length, -1))
        up_fp32.copy_(up_fp16.reshape(sequence_length, -1))
        torch.sigmoid(gate_fp32, out=product)
        product.mul_(gate_fp32)
        product.mul_(up_fp32)
        _observe_graph_finite(owner, "layer2_silu_times_up_fp32", product)
        torch.mm(product, mlp._cmp50hx_down_weight_fp32.t(), out=down)
        if mlp._cmp50hx_down_bias_fp32 is not None:
            down.add_(mlp._cmp50hx_down_bias_fp32)
        _observe_graph_finite(owner, "layer2_down_fp32", down)
        output.copy_(down.unsqueeze(0))
        _observe_graph_finite(owner, "layer2_output_fp16", output)
        return output

    mlp.forward = forward
    mlp._cmp50hx_fp32_island_installed = True


def _install_layer2_mlp_fused_gate_up(owner: Any) -> None:
    """Fuse FP16 gate/up projections while retaining the proven FP32 tail."""

    layer_index = 2
    mlp = owner.pred_model.layers[layer_index].mlp
    if getattr(mlp, "_cmp50hx_fp32_island_installed", False):
        return

    max_tokens = 2
    intermediate_size = mlp.gate_proj.weight.shape[0]
    hidden_size = mlp.down_proj.weight.shape[0]
    device = owner.device
    mlp._cmp50hx_fused_gate_up_weight = torch.cat(
        [mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0
    ).contiguous()
    mlp._cmp50hx_down_weight_fp32 = mlp.down_proj.weight.detach().float()
    down_bias = getattr(mlp.down_proj, "bias", None)
    mlp._cmp50hx_down_bias_fp32 = (
        down_bias.detach().float() if down_bias is not None else None
    )
    mlp._cmp50hx_gate_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_up_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_product_fp32 = torch.empty(
        (max_tokens, intermediate_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_down_fp32 = torch.empty(
        (max_tokens, hidden_size), dtype=torch.float32, device=device
    )
    mlp._cmp50hx_output_fp16 = torch.empty(
        (1, max_tokens, hidden_size), dtype=torch.float16, device=device
    )
    mlp._cmp50hx_precision_variant = "fused_gate_up_fp16_fp32_tail"
    mlp._cmp50hx_gate_up_compute_dtype = "float16"
    mlp._cmp50hx_product_compute_dtype = "float32"
    mlp._cmp50hx_down_compute_dtype = "float32"

    def forward(x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        fused = F.linear(x, mlp._cmp50hx_fused_gate_up_weight)
        gate_fp16 = fused[..., :intermediate_size]
        up_fp16 = fused[..., intermediate_size:]
        gate_fp32 = mlp._cmp50hx_gate_fp32[:sequence_length]
        up_fp32 = mlp._cmp50hx_up_fp32[:sequence_length]
        product = mlp._cmp50hx_product_fp32[:sequence_length]
        down = mlp._cmp50hx_down_fp32[:sequence_length]
        output = mlp._cmp50hx_output_fp16[:, :sequence_length, :]
        gate_fp32.copy_(gate_fp16.reshape(sequence_length, -1))
        up_fp32.copy_(up_fp16.reshape(sequence_length, -1))
        torch.sigmoid(gate_fp32, out=product)
        product.mul_(gate_fp32)
        product.mul_(up_fp32)
        torch.mm(product, mlp._cmp50hx_down_weight_fp32.t(), out=down)
        if mlp._cmp50hx_down_bias_fp32 is not None:
            down.add_(mlp._cmp50hx_down_bias_fp32)
        output.copy_(down.unsqueeze(0))
        return output

    mlp.forward = forward
    mlp._cmp50hx_fp32_island_installed = True


def _install_residual_carrier_fp32(owner: Any) -> None:
    """Keep only the predictor residual spine in FP32 for diagnostics.

    Attention and MLP branches still receive FP16 normalized activations. This
    eager-only hook remains diagnostic and is not used by the graph profile.
    """

    if getattr(owner, "_cmp50hx_residual_carrier_fp32_installed", False):
        return

    layer_index = 2
    layer = owner.pred_model.layers[layer_index]

    def replace_layer2_output(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        if not owner._cmp50hx_diagnostic_active:
            return None
        records = owner._cmp50hx_diagnostic_records
        residuals = records.get((layer_index, "post_attention_residual"), [])
        branches = records.get((layer_index, "mlp_output"), [])
        if not residuals or not branches:
            raise RuntimeError("CMP residual carrier lacks layer-2 operands")
        carried = residuals[-1].float() + branches[-1].float()
        records.setdefault((layer_index, "post_mlp_residual_fp32"), []).append(
            carried.detach()
        )
        if not isinstance(output, tuple):
            return carried
        return (carried, *output[1:])

    def fp16_normalized_branch(
        component: str,
        index: int,
    ) -> Callable:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not owner._cmp50hx_diagnostic_active:
                return None
            if not isinstance(output, torch.Tensor):
                return None
            normalized = output.to(dtype=torch.float16)
            owner._cmp50hx_diagnostic_records.setdefault(
                (index, component), []
            ).append(normalized.detach())
            return normalized

        return hook

    owner._cmp50hx_diagnostic_handles.append(
        layer.register_forward_hook(replace_layer2_output)
    )
    for index, next_layer in enumerate(owner.pred_model.layers[layer_index + 1 :], start=layer_index + 1):
        owner._cmp50hx_diagnostic_handles.append(
            next_layer.input_layernorm.register_forward_hook(
                fp16_normalized_branch("input_layernorm_output_fp16", index)
            )
        )
        owner._cmp50hx_diagnostic_handles.append(
            next_layer.post_attention_layernorm.register_forward_hook(
                fp16_normalized_branch(
                    "post_attention_layernorm_output_fp16", index
                )
            )
        )

    def final_norm_fp16(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        if owner._cmp50hx_diagnostic_active and isinstance(output, torch.Tensor):
            return output.to(dtype=torch.float16)
        return None

    owner._cmp50hx_diagnostic_handles.append(
        owner.pred_model.norm.register_forward_hook(final_norm_fp16)
    )
    owner._cmp50hx_diagnostic_handles.append(
        owner.pred_model.norm.register_forward_pre_hook(
            _record_module_input(owner, len(owner.pred_model.layers), "final_norm_input")
        )
    )
    owner._cmp50hx_diagnostic_handles.append(
        owner.pred_model.norm.register_forward_hook(
            _record_module_output(owner, len(owner.pred_model.layers), "final_norm_output")
        )
    )
    owner._cmp50hx_residual_carrier_fp32_installed = True


def _install_graph_residual_carrier_fp32(owner: Any) -> None:
    """Install a preallocated FP32 residual carrier that CUDA graphs can replay.

    Hooks execute while FasterQwen captures its fixed-shape predictor loop. The
    capture records the FP32 copies, addition, FP32 RMSNorm, and FP16 branch
    copies; graph replay then runs those recorded kernels without Python hooks,
    diagnostics, synchronisation, or tensor allocation on a measured request.
    """

    if getattr(owner, "_cmp50hx_graph_residual_carrier_installed", False):
        return

    layer_index = 2
    max_tokens = 2
    shape = (1, max_tokens, owner.hidden_size)
    device = owner.device
    layer = owner.pred_model.layers[layer_index]
    owner._cmp50hx_graph_layer2_residual_fp32 = torch.empty(
        shape, dtype=torch.float32, device=device
    )
    owner._cmp50hx_graph_layer2_branch_fp32 = torch.empty(
        shape, dtype=torch.float32, device=device
    )
    owner._cmp50hx_graph_layer2_carried_fp32 = torch.empty(
        shape, dtype=torch.float32, device=device
    )
    owner._cmp50hx_graph_layer2_residual: torch.Tensor | None = None
    owner._cmp50hx_graph_layer2_branch: torch.Tensor | None = None

    def buffer_view(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return buffer[:, : value.shape[1], :]

    def remember_layer2_residual(
        _module: Any,
        inputs: tuple[Any, ...],
    ) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            owner._cmp50hx_graph_layer2_residual = inputs[0]

    def remember_layer2_branch(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if isinstance(output, torch.Tensor):
            owner._cmp50hx_graph_layer2_branch = output

    def replace_layer2_output(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        residual = owner._cmp50hx_graph_layer2_residual
        branch = owner._cmp50hx_graph_layer2_branch
        if residual is None or branch is None:
            raise RuntimeError("CMP graph residual carrier lacks layer-2 operands")
        residual_fp32 = buffer_view(owner._cmp50hx_graph_layer2_residual_fp32, residual)
        branch_fp32 = buffer_view(owner._cmp50hx_graph_layer2_branch_fp32, branch)
        carried = buffer_view(owner._cmp50hx_graph_layer2_carried_fp32, branch)
        residual_fp32.copy_(residual)
        branch_fp32.copy_(branch)
        torch.add(residual_fp32, branch_fp32, out=carried)
        _observe_graph_finite(owner, "layer2_residual_fp32", carried)
        if not isinstance(output, tuple):
            return carried
        return (carried, *output[1:])

    def fp16_normalized_branch(index: int, component: str) -> Callable:
        buffer = torch.empty(shape, dtype=torch.float16, device=device)
        setattr(owner, f"_cmp50hx_graph_{component}_{index}_fp16", buffer)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if not isinstance(output, torch.Tensor):
                return None
            _observe_graph_finite(owner, "rmsnorm_fp32", output)
            normalized = buffer_view(buffer, output)
            normalized.copy_(output)
            _observe_graph_finite(owner, "normalized_branch_fp16", normalized)
            return normalized

        return hook

    owner._cmp50hx_diagnostic_handles.append(
        layer.post_attention_layernorm.register_forward_pre_hook(
            remember_layer2_residual
        )
    )
    owner._cmp50hx_diagnostic_handles.append(
        layer.mlp.register_forward_hook(remember_layer2_branch)
    )
    owner._cmp50hx_diagnostic_handles.append(
        layer.register_forward_hook(replace_layer2_output)
    )
    for index, next_layer in enumerate(
        owner.pred_model.layers[layer_index + 1 :], start=layer_index + 1
    ):
        owner._cmp50hx_diagnostic_handles.append(
            next_layer.input_layernorm.register_forward_hook(
                fp16_normalized_branch(index, "input_layernorm")
            )
        )
        owner._cmp50hx_diagnostic_handles.append(
            next_layer.post_attention_layernorm.register_forward_hook(
                fp16_normalized_branch(index, "post_attention_layernorm")
            )
        )

    final_norm_fp16 = torch.empty(shape, dtype=torch.float16, device=device)
    owner._cmp50hx_graph_final_norm_fp16 = final_norm_fp16

    def final_norm_hook(
        _module: Any,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        if not isinstance(output, torch.Tensor):
            return None
        _observe_graph_finite(owner, "rmsnorm_fp32", output)
        normalized = buffer_view(final_norm_fp16, output)
        normalized.copy_(output)
        _observe_graph_finite(owner, "normalized_branch_fp16", normalized)
        return normalized

    owner._cmp50hx_diagnostic_handles.append(
        owner.pred_model.norm.register_forward_hook(final_norm_hook)
    )
    owner._cmp50hx_graph_residual_carrier_installed = True


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    sanitized = torch.nan_to_num(value, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(finite.all().item()),
        "nan_count": int(torch.isnan(value).sum().item()),
        "positive_inf_count": int(torch.isposinf(value).sum().item()),
        "negative_inf_count": int(torch.isneginf(value).sum().item()),
        "max_abs": float(sanitized.abs().max().item()),
    }


def _probability_summary(value: torch.Tensor) -> dict[str, Any]:
    """Describe the multinomial contract without changing its input."""

    summary = _tensor_summary(value)
    sums = value.sum(dim=-1)
    nonnegative = value >= 0
    summary.update(
        {
            "all_nonnegative": bool(nonnegative.all().item()),
            "sum": [float(item) for item in sums.flatten().tolist()],
            "all_sums_positive": bool((sums > 0).all().item()),
        }
    )
    return summary


def _append_trace(value: dict[str, Any]) -> None:
    trace_path = os.environ.get(_ENV_TRACE)
    if not trace_path:
        return
    with Path(trace_path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _cache_sequence_length(graph: Any) -> int | None:
    get_length = getattr(getattr(graph, "static_cache", None), "get_seq_length", None)
    if not callable(get_length):
        return None
    value = get_length()
    if value is None:
        return None
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _record_projection(owner: Any, layer_index: int, component: str) -> Callable:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if owner._cmp50hx_diagnostic_active and isinstance(output, torch.Tensor):
            owner._cmp50hx_diagnostic_records.setdefault((layer_index, component), []).append(
                output.detach()
            )

    return hook


def _record_mlp_input(owner: Any, layer_index: int) -> Callable:
    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if (
            owner._cmp50hx_diagnostic_active
            and inputs
            and isinstance(inputs[0], torch.Tensor)
        ):
            owner._cmp50hx_diagnostic_records.setdefault(
                (layer_index, "mlp_input"), []
            ).append(inputs[0].detach())

    return hook


def _record_module_input(owner: Any, layer_index: int, component: str) -> Callable:
    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if (
            owner._cmp50hx_diagnostic_active
            and inputs
            and isinstance(inputs[0], torch.Tensor)
        ):
            owner._cmp50hx_diagnostic_records.setdefault(
                (layer_index, component), []
            ).append(inputs[0].detach())

    return hook


def _record_module_output(owner: Any, layer_index: int, component: str) -> Callable:
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not owner._cmp50hx_diagnostic_active:
            return
        value = output[0] if isinstance(output, tuple) else output
        if isinstance(value, torch.Tensor):
            owner._cmp50hx_diagnostic_records.setdefault(
                (layer_index, component), []
            ).append(value.detach())

    return hook


def _scalar_text(value: torch.Tensor) -> float | str:
    scalar = float(value.item())
    if math.isnan(scalar):
        return "nan"
    if scalar == float("inf"):
        return "+inf"
    if scalar == float("-inf"):
        return "-inf"
    return scalar


def _residual_add_summary(
    residual: torch.Tensor,
    branch: torch.Tensor,
    result: torch.Tensor,
    next_layernorm: Any,
) -> dict[str, Any]:
    """Describe one observed residual addition at the result's largest entry."""

    flat_result = result.detach().abs().reshape(-1)
    max_index = int(torch.nan_to_num(flat_result, nan=float("inf")).argmax().item())
    residual_flat = residual.detach().reshape(-1)
    branch_flat = branch.detach().reshape(-1)
    result_flat = result.detach().reshape(-1)
    fp32_sum = residual.float() + branch.float()
    fp32_norm = next_layernorm(fp32_sum)
    fp16_norm = fp32_norm.to(dtype=branch.dtype)
    return {
        "residual": _tensor_summary(residual),
        "branch": _tensor_summary(branch),
        "result": _tensor_summary(result),
        "max_abs_flat_index": max_index,
        "operands_at_result_max_abs": {
            "residual": _scalar_text(residual_flat[max_index]),
            "branch": _scalar_text(branch_flat[max_index]),
            "result": _scalar_text(result_flat[max_index]),
        },
        "fp32_add_then_next_input_layernorm_counterfactual": {
            "fp32_sum": _tensor_summary(fp32_sum),
            "layernorm_output": _tensor_summary(fp32_norm),
            "layernorm_output_fp16": _tensor_summary(fp16_norm),
        },
    }


def _summarize_predictor(owner: Any) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    products: list[dict[str, Any]] = []
    residual_additions: list[dict[str, Any]] = []
    first_nonfinite: dict[str, Any] | None = None
    layers = owner.pred_model.layers

    for layer_index, layer in enumerate(layers):
        by_component: dict[str, list[torch.Tensor]] = {}
        for component in (
            "input_layernorm_input",
            "input_layernorm_output",
            "attention_output",
            "post_attention_residual",
            "post_attention_layernorm_output",
            "mlp_input",
            "gate_proj",
            "up_proj",
            "down_proj",
            "mlp_output",
            "post_mlp_residual_fp32",
            "input_layernorm_output_fp16",
            "post_attention_layernorm_output_fp16",
        ):
            values = owner._cmp50hx_diagnostic_records.get((layer_index, component), [])
            by_component[component] = values
            for forward_index, value in enumerate(values):
                summary = _tensor_summary(value)
                entry = {
                    "layer": layer_index,
                    "component": component,
                    "forward_index": forward_index,
                    **summary,
                }
                components.append(entry)
                if first_nonfinite is None and not summary["finite"]:
                    first_nonfinite = entry

        for forward_index, (gate, up) in enumerate(
            zip(by_component["gate_proj"], by_component["up_proj"])
        ):
            product = layer.mlp.act_fn(gate) * up
            summary = _tensor_summary(product)
            entry = {
                "layer": layer_index,
                "component": "silu_gate_times_up",
                "forward_index": forward_index,
                **summary,
            }
            products.append(entry)
            if first_nonfinite is None and not summary["finite"]:
                first_nonfinite = entry

        residuals = owner._cmp50hx_diagnostic_records.get(
            (layer_index, "post_attention_residual"), []
        )
        branches = owner._cmp50hx_diagnostic_records.get(
            (layer_index, "mlp_output"), []
        )
        if layer_index + 1 < len(layers):
            results = owner._cmp50hx_diagnostic_records.get(
                (layer_index + 1, "input_layernorm_input"), []
            )
            next_layernorm = layers[layer_index + 1].input_layernorm
        else:
            results = owner._cmp50hx_diagnostic_records.get(
                (len(layers), "final_norm_input"), []
            )
            next_layernorm = owner.pred_model.norm
        for forward_index, (residual, branch, result) in enumerate(
            zip(residuals, branches, results)
        ):
            residual_additions.append(
                {
                    "layer": layer_index,
                    "component": "post_mlp_residual_add",
                    "forward_index": forward_index,
                    **_residual_add_summary(
                        residual,
                        branch,
                        result,
                        next_layernorm,
                    ),
                }
            )

    final_norm_index = len(layers)
    for component in ("final_norm_input", "final_norm_output"):
        values = owner._cmp50hx_diagnostic_records.get((final_norm_index, component), [])
        for forward_index, value in enumerate(values):
            summary = _tensor_summary(value)
            entry = {
                "layer": None,
                "component": component,
                "forward_index": forward_index,
                **summary,
            }
            components.append(entry)
            if first_nonfinite is None and not summary["finite"]:
                first_nonfinite = entry

    return {
        "component_observations": components,
        "gated_mlp_product_observations": products,
        "residual_add_observations": residual_additions,
        "first_nonfinite": first_nonfinite,
    }


def _diagnostic_context(owner: Any) -> dict[str, int]:
    return {
        "request_index": int(getattr(owner, "_cmp50hx_request_index", 0)),
        "codec_step": int(getattr(owner, "_cmp50hx_codec_step", 0)),
    }


def _record_sampling_failure(
    owner: Any,
    boundary: dict[str, Any],
    *,
    reason: str,
) -> None:
    _append_trace(
        {
            "event": "faster_predictor_invalid_sampling_boundary",
            "reason": reason,
            **_diagnostic_context(owner),
            "sampling_boundary": boundary,
            "predictor": _summarize_predictor(owner),
        }
    )


def _guarded_predictor_sample(original: Callable) -> Callable:
    def guarded(logits: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        owner = _active_predictor
        if owner is not None:
            boundary = {
                **_diagnostic_context(owner),
                "sampling_index": len(owner._cmp50hx_diagnostic_sampling_boundaries),
                "logits": _tensor_summary(logits),
            }
            owner._cmp50hx_diagnostic_sampling_boundaries.append(boundary)
            if not boundary["logits"]["finite"]:
                _record_sampling_failure(owner, boundary, reason="nonfinite_logits")
                raise RuntimeError(
                    "CMP FasterQwen diagnostic stopped before torch.multinomial: "
                    "code-predictor logits are non-finite"
                )

            original_multinomial = torch.multinomial

            def guarded_multinomial(
                probabilities: torch.Tensor,
                *multinomial_args: Any,
                **multinomial_kwargs: Any,
            ) -> torch.Tensor:
                probability = _probability_summary(probabilities)
                boundary["probabilities"] = probability
                valid = (
                    probability["finite"]
                    and probability["all_nonnegative"]
                    and probability["all_sums_positive"]
                )
                if not valid:
                    _record_sampling_failure(
                        owner,
                        boundary,
                        reason="invalid_probabilities",
                    )
                    raise RuntimeError(
                        "CMP FasterQwen diagnostic stopped before torch.multinomial: "
                        "probabilities are not finite, nonnegative, and positive-sum"
                    )
                return original_multinomial(
                    probabilities,
                    *multinomial_args,
                    **multinomial_kwargs,
                )

            torch.multinomial = guarded_multinomial
            try:
                return original(logits, *args, **kwargs)
            finally:
                torch.multinomial = original_multinomial
        return original(logits, *args, **kwargs)

    return guarded


def _finite_checked_predictor_sample(original: Callable) -> Callable:
    """Capture predictor aggregates and repair invalid multinomial input.

    The repair is diagnostic-only.  It prevents a known invalid probability row
    from poisoning the CUDA context before the captured checker can report the
    first anomalous boundary at request completion.
    """

    def checked(logits: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        owner = _active_graph_finite_predictor
        if owner is None:
            return original(logits, *args, **kwargs)
        _observe_graph_finite(owner, "predictor_logits", logits)
        original_multinomial = torch.multinomial

        def checked_multinomial(
            probabilities: torch.Tensor,
            *multinomial_args: Any,
            **multinomial_kwargs: Any,
        ) -> torch.Tensor:
            _observe_graph_finite(owner, "predictor_probabilities", probabilities)
            finite = torch.isfinite(probabilities).all(dim=-1, keepdim=True)
            nonnegative = (probabilities >= 0).all(dim=-1, keepdim=True)
            positive_sum = (probabilities.sum(dim=-1, keepdim=True) > 0)
            valid_rows = torch.logical_and(
                torch.logical_and(finite, nonnegative), positive_sum
            )
            _record_graph_checker_anomaly(
                owner,
                "predictor_probabilities",
                torch.logical_not(valid_rows.all()),
            )
            repaired = torch.nan_to_num(
                probabilities, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
            repaired_sum = repaired.sum(dim=-1, keepdim=True)
            fallback = torch.ones_like(repaired)
            repaired = torch.where(repaired_sum > 0, repaired, fallback)
            repaired = repaired / repaired.sum(dim=-1, keepdim=True)
            safe_probabilities = torch.where(valid_rows, probabilities, repaired)
            return original_multinomial(
                safe_probabilities, *multinomial_args, **multinomial_kwargs
            )

        torch.multinomial = checked_multinomial
        try:
            return original(logits, *args, **kwargs)
        finally:
            torch.multinomial = original_multinomial

    return checked


def install() -> None:
    """Install opt-in eager diagnostics and graph-compatible finite checking."""

    global _installed
    if _installed or not (
        _enabled()
        or _mlp_fp32_island_enabled()
        or _residual_carrier_fp32_enabled()
        or _graph_residual_carrier_fp32_enabled()
        or _mlp_fused_gate_up_enabled()
        or _graph_finite_checker_enabled()
    ):
        return

    from . import predictor_graph as predictor_graph_module
    from . import streaming as streaming_module
    from . import talker_graph as talker_graph_module

    predictor_class = predictor_graph_module.PredictorGraph
    talker_class = talker_graph_module.TalkerGraph
    original_predictor_init = predictor_class.__init__
    original_predictor_capture = predictor_class.capture
    original_predictor_run = predictor_class.run
    original_talker_capture = talker_class.capture
    original_sample = predictor_graph_module.sample_logits
    original_stream = streaming_module.fast_generate_streaming

    def predictor_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_predictor_init(self, *args, **kwargs)
        self._cmp50hx_diagnostic_active = False
        self._cmp50hx_diagnostic_records: dict[tuple[int, str], list[torch.Tensor]] = {}
        self._cmp50hx_diagnostic_sampling_boundaries: list[dict[str, Any]] = []
        self._cmp50hx_diagnostic_handles = []
        self._cmp50hx_request_index = 0
        self._cmp50hx_codec_step = 0
        if _graph_finite_checker_enabled():
            _install_graph_finite_checker(self)
        for layer_index, layer in enumerate(self.pred_model.layers):
            self._cmp50hx_diagnostic_handles.append(
                layer.input_layernorm.register_forward_pre_hook(
                    _record_module_input(self, layer_index, "input_layernorm_input")
                )
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.input_layernorm.register_forward_hook(
                    _record_module_output(self, layer_index, "input_layernorm_output")
                )
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.self_attn.register_forward_hook(
                    _record_module_output(self, layer_index, "attention_output")
                )
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    _record_module_input(self, layer_index, "post_attention_residual")
                )
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    _record_module_output(
                        self, layer_index, "post_attention_layernorm_output"
                    )
                )
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.mlp.register_forward_pre_hook(_record_mlp_input(self, layer_index))
            )
            self._cmp50hx_diagnostic_handles.append(
                layer.mlp.register_forward_hook(
                    _record_module_output(self, layer_index, "mlp_output")
                )
            )
            for component in ("gate_proj", "up_proj", "down_proj"):
                module = getattr(layer.mlp, component)
                self._cmp50hx_diagnostic_handles.append(
                    module.register_forward_hook(_record_projection(self, layer_index, component))
                )
        if _mlp_fused_gate_up_enabled():
            _install_layer2_mlp_fused_gate_up(self)
        elif _mlp_fp32_island_enabled() and _mlp_narrow_gate_up_fp16_enabled():
            _install_layer2_mlp_narrow_gate_up_fp16(self)
        elif _mlp_fp32_island_enabled():
            _install_layer2_mlp_fp32_island(self)
        if _residual_carrier_fp32_enabled():
            _install_residual_carrier_fp32(self)
        if _graph_residual_carrier_fp32_enabled():
            _install_graph_residual_carrier_fp32(self)

    def graph_predictor_capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _active_graph_finite_predictor
        if _graph_finite_checker_enabled():
            _active_graph_finite_predictor = self
        try:
            result = original_predictor_capture(self, *args, **kwargs)
        finally:
            _active_graph_finite_predictor = None
        layer2_mlp = self.pred_model.layers[2].mlp
        _append_graph_carrier_proof(
            {
                "event": "predictor_graph_capture",
                "carrier_active": bool(
                    getattr(self, "_cmp50hx_graph_residual_carrier_installed", False)
                ),
                "captured": bool(self.captured),
                "residual_dtype": "float32",
                "norm_dtype": "float32",
                "normalized_branch_dtype": "float16",
                "preallocated_static_buffers": True,
                "finite_checker_active": bool(
                    getattr(self, "_cmp50hx_graph_finite_checker_installed", False)
                ),
                "finite_checker_components": list(_GRAPH_FINITE_COMPONENTS)
                if _graph_finite_checker_enabled()
                else [],
                "layer2_mlp_variant": getattr(
                    layer2_mlp, "_cmp50hx_precision_variant", "unknown"
                ),
                "layer2_gate_up_dtype": getattr(
                    layer2_mlp, "_cmp50hx_gate_up_compute_dtype", "unknown"
                ),
                "layer2_product_dtype": getattr(
                    layer2_mlp, "_cmp50hx_product_compute_dtype", "unknown"
                ),
                "layer2_down_dtype": getattr(
                    layer2_mlp, "_cmp50hx_down_compute_dtype", "unknown"
                ),
            }
        )
        return result

    def graph_talker_capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_talker_capture(self, *args, **kwargs)
        _append_graph_carrier_proof(
            {
                "event": "talker_graph_capture",
                "captured": bool(self.captured),
            }
        )
        return result

    @torch.inference_mode()
    def predictor_capture(self: Any, num_warmup: int = 3) -> None:
        print(
            "CMP diagnostic: FasterQwen predictor CUDA graph capture bypassed; "
            "eager-only numerical trace."
        )
        self._init_cache_layers()
        self._build_attention_masks()
        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize()
        self.captured = True

    @torch.inference_mode()
    def predictor_run(self: Any, pred_input: torch.Tensor) -> torch.Tensor:
        global _active_predictor
        if self._cmp50hx_request_index < _diagnostic_start_request():
            return original_predictor_run(self, pred_input)
        self.input_buf.copy_(pred_input)
        self.static_cache.reset()
        self._cmp50hx_diagnostic_active = True
        self._cmp50hx_diagnostic_records = {}
        self._cmp50hx_diagnostic_sampling_boundaries = []
        _active_predictor = self
        try:
            result = self._full_loop()
            torch.cuda.synchronize()
            _append_trace(
                {
                    "event": "faster_predictor_eager_run_complete",
                    **_diagnostic_context(self),
                    "sampling_boundaries": self._cmp50hx_diagnostic_sampling_boundaries,
                    "predictor": _summarize_predictor(self),
                }
            )
            self._cmp50hx_codec_step += 1
            return result.clone()
        finally:
            _active_predictor = None
            self._cmp50hx_diagnostic_active = False

    @torch.inference_mode()
    def graph_finite_predictor_run(
        self: Any,
        pred_input: torch.Tensor,
    ) -> torch.Tensor:
        self._cmp50hx_graph_finite_replay_index.add_(1)
        return original_predictor_run(self, pred_input)

    @torch.inference_mode()
    def talker_capture(self: Any, prefill_len: int = 100, num_warmup: int = 3) -> None:
        print(
            "CMP diagnostic: FasterQwen talker CUDA graph capture bypassed; "
            "eager-only numerical trace."
        )
        self._init_cache_layers()
        self._build_attention_masks()
        self.cache_position[0] = prefill_len
        self._set_attention_mask(prefill_len)
        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()
        self.captured = True

    @torch.inference_mode()
    def talker_run(self: Any, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self._set_attention_mask(position)
        delta = self.rope_deltas + self.cache_position[0].to(self.rope_deltas.dtype)
        self.position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))
        self._decode_step()
        return self.output_buf

    def traced_fast_generate_streaming(*args: Any, **kwargs: Any) -> Any:
        global _request_index
        predictor = kwargs.get("predictor_graph")
        if predictor is None and len(args) > 6:
            predictor = args[6]
        if predictor is None:
            return original_stream(*args, **kwargs)

        _request_index += 1
        predictor._cmp50hx_request_index = _request_index
        predictor._cmp50hx_codec_step = 0
        if _enabled():
            talker = kwargs.get("talker_graph")
            rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()
            _append_trace(
                {
                    "event": "faster_request_start",
                    "request_index": _request_index,
                    "torch_initial_seed": int(torch.initial_seed()),
                    "torch_cpu_rng_sha256": _tensor_sha256(rng_state),
                    "torch_cuda_rng_sha256": _tensor_sha256(cuda_rng_state),
                    "predictor_static_cache_sequence_length": _cache_sequence_length(
                        predictor
                    ),
                    "talker_static_cache_sequence_length": _cache_sequence_length(talker),
                }
            )
        stream = original_stream(*args, **kwargs)
        if not _graph_finite_checker_enabled():
            return stream

        def checked_stream() -> Any:
            _reset_graph_finite_checker(predictor)
            completed = False
            try:
                yield from stream
                completed = True
            finally:
                if completed:
                    _finalize_graph_finite_checker(predictor)

        return checked_stream()

    predictor_class.__init__ = predictor_init
    if _graph_residual_carrier_fp32_enabled() or _graph_finite_checker_enabled():
        predictor_class.capture = graph_predictor_capture
        talker_class.capture = graph_talker_capture
    if _graph_finite_checker_enabled():
        predictor_class.run = graph_finite_predictor_run
        predictor_graph_module.sample_logits = _finite_checked_predictor_sample(
            original_sample
        )
        streaming_module.fast_generate_streaming = traced_fast_generate_streaming
    if _enabled():
        predictor_class.run = predictor_run
        predictor_graph_module.sample_logits = _guarded_predictor_sample(original_sample)
        streaming_module.fast_generate_streaming = traced_fast_generate_streaming
    _installed = True
