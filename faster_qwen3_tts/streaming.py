#!/usr/bin/env python3
"""
Streaming generation with CUDA graphs for both predictor and talker.

Yields codec ID chunks during generation instead of collecting all at once.
CUDA graph usage is identical to non-streaming — same per-step performance.
"""

import math
import threading
import time
from collections import OrderedDict
from typing import Callable, Generator, Optional, Tuple

import torch
import torch.nn.functional as F

from .generate import get_eos_tracker, get_fused_codec_embeddings
from .prefill_compat import (
    ensure_prefill_compile_compat,
    normalize_prefill_compile_compat_mode,
    prefill_compile_compat_context,
    validate_strict_bf16_sdpa_v1,
)
from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, build_suppress_mask, sample_logits
from .talker_graph import TalkerGraph


_PREFILL_BACKENDS = {
    "eager",
    "compile_backend_eager",
    "compile_backend_aot_eager",
    "compile_default",
    "compile_inductor_default",
    "compile_inductor_graphbreak",
    "compile_reduce_overhead",
}
_PREFILL_BACKEND_ALIASES = {
    "compile_default": "compile_inductor_default",
}
_PREFILL_COMPILE_CACHE_MAX_ENTRIES = 64
_PREFILL_COMPILE_CACHE = OrderedDict()
_PREFILL_COMPILE_ERRORS = OrderedDict()
_PREFILL_COMPILE_CACHE_LOCK = threading.RLock()


class UnsupportedPrefillConfiguration(ValueError):
    """Raised when a requested prefill backend/mask combination is unsafe."""


class _CudaNvtxRange:
    def __init__(self, name: str, enabled: bool):
        self._name = name
        self._enabled = enabled

    def __enter__(self):
        if self._enabled:
            torch.cuda.nvtx.range_push(self._name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._enabled:
            torch.cuda.nvtx.range_pop()
        return False


def _normalize_prefill_backend(prefill_backend: str) -> str:
    backend = str(prefill_backend or "eager").strip().lower()
    backend = _PREFILL_BACKEND_ALIASES.get(backend, backend)
    if backend not in _PREFILL_BACKENDS:
        raise ValueError(
            f"Unsupported prefill_backend {prefill_backend!r}. "
            f"Expected one of {sorted(_PREFILL_BACKENDS)}."
        )
    return backend


def _run_talker_prefill(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    *,
    prefill_backend: str,
    prefill_mask_mode: str = "auto",
    prefill_compile_compat_mode: str = "none",
    input_metadata: Optional[dict] = None,
):
    prefill_backend = _normalize_prefill_backend(prefill_backend)
    prefill_mask_mode = _normalize_prefill_mask_mode(prefill_mask_mode)
    prefill_compile_compat_mode = normalize_prefill_compile_compat_mode(
        prefill_compile_compat_mode
    )
    _validate_prefill_configuration(prefill_backend, prefill_mask_mode)
    skip_prefill_causal_mask = prefill_mask_mode == "skip"
    prefill_attention_mask = None if skip_prefill_causal_mask else attention_mask
    _validate_prefill_compile_compat_configuration(
        prefill_backend=prefill_backend,
        prefill_mask_mode=prefill_mask_mode,
        prefill_compile_compat_mode=prefill_compile_compat_mode,
        talker_input_embeds=talker_input_embeds,
        attention_mask=prefill_attention_mask,
        input_metadata=input_metadata,
    )
    cache_stats = prefill_compile_cache_stats()
    profile = {
        "prefill_backend_requested": prefill_backend,
        "prefill_backend_used": "eager",
        "prefill_compile_fallback": False,
        "prefill_compile_error": None,
        "prefill_mask_mode": prefill_mask_mode,
        "prefill_skip_causal_mask": skip_prefill_causal_mask,
        "prefill_compile_compat_mode": prefill_compile_compat_mode,
        "prefill_compile_cache_hit": False,
        "prefill_compile_cache_entries": cache_stats["entries"],
        "prefill_compile_cache_max_entries": cache_stats["max_entries"],
        "prefill_compile_wall_ms": 0.0,
    }
    if prefill_backend == "eager":
        return (
            _talker_prefill_eager(
                talker,
                talker_input_embeds,
                prefill_attention_mask,
                trailing_text_hiddens,
                tts_pad_embed,
                skip_prefill_causal_mask=skip_prefill_causal_mask,
            ),
            profile,
        )

    cache_key = _prefill_compile_cache_key(
        talker,
        talker_input_embeds,
        prefill_attention_mask,
        trailing_text_hiddens,
        tts_pad_embed,
        prefill_backend,
        prefill_mask_mode,
        prefill_compile_compat_mode,
    )
    cached_error = _prefill_compile_error(cache_key)
    if cached_error is not None:
        profile["prefill_compile_fallback"] = True
        profile["prefill_compile_error"] = cached_error
        return (
            _talker_prefill_eager(
                talker,
                talker_input_embeds,
                prefill_attention_mask,
                trailing_text_hiddens,
                tts_pad_embed,
                skip_prefill_causal_mask=skip_prefill_causal_mask,
            ),
            profile,
        )

    try:
        profile.update(
            ensure_prefill_compile_compat(talker, prefill_compile_compat_mode)
        )
        with prefill_compile_compat_context(
            talker,
            prefill_compile_compat_mode,
        ) as metadata:
            profile.update(metadata)
            compiled = _prefill_compile_cache_get(cache_key)
            profile["prefill_compile_cache_hit"] = compiled is not None
            if compiled is None:
                compile_started = time.perf_counter()
                compiled = _compile_talker_prefill(talker, prefill_backend)
                profile["prefill_compile_wall_ms"] = round(
                    (time.perf_counter() - compile_started) * 1000.0,
                    3,
                )
                _prefill_compile_cache_store(cache_key, compiled)
            cache_stats = prefill_compile_cache_stats()
            profile["prefill_compile_cache_entries"] = cache_stats["entries"]
            profile["prefill_compile_cache_max_entries"] = cache_stats["max_entries"]
            out = compiled(
                talker_input_embeds,
                prefill_attention_mask,
                trailing_text_hiddens,
                tts_pad_embed,
                skip_prefill_causal_mask,
            )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        _prefill_compile_error_store(cache_key, message)
        profile["prefill_compile_fallback"] = True
        profile["prefill_compile_error"] = message
        return (
            _talker_prefill_eager(
                talker,
                talker_input_embeds,
                prefill_attention_mask,
                trailing_text_hiddens,
                tts_pad_embed,
                skip_prefill_causal_mask=skip_prefill_causal_mask,
            ),
            profile,
        )

    profile["prefill_backend_used"] = prefill_backend
    return out, profile


def prefill_compile_cache_stats() -> dict[str, int]:
    with _PREFILL_COMPILE_CACHE_LOCK:
        return {
            "entries": len(_PREFILL_COMPILE_CACHE),
            "errors": len(_PREFILL_COMPILE_ERRORS),
            "max_entries": _PREFILL_COMPILE_CACHE_MAX_ENTRIES,
        }


def clear_prefill_compile_cache(talker=None) -> dict[str, int]:
    talker_id = id(talker) if talker is not None else None
    removed_entries = 0
    removed_errors = 0
    with _PREFILL_COMPILE_CACHE_LOCK:
        for key in list(_PREFILL_COMPILE_CACHE):
            if talker_id is None or _cache_key_talker_id(key) == talker_id:
                del _PREFILL_COMPILE_CACHE[key]
                removed_entries += 1
        for key in list(_PREFILL_COMPILE_ERRORS):
            if talker_id is None or _cache_key_talker_id(key) == talker_id:
                del _PREFILL_COMPILE_ERRORS[key]
                removed_errors += 1
    return {
        "removed_entries": removed_entries,
        "removed_errors": removed_errors,
        **prefill_compile_cache_stats(),
    }


def _prefill_compile_cache_get(cache_key: tuple):
    with _PREFILL_COMPILE_CACHE_LOCK:
        compiled = _PREFILL_COMPILE_CACHE.get(cache_key)
        if compiled is not None:
            _PREFILL_COMPILE_CACHE.move_to_end(cache_key)
        return compiled


def _prefill_compile_cache_store(cache_key: tuple, compiled) -> None:
    with _PREFILL_COMPILE_CACHE_LOCK:
        _PREFILL_COMPILE_CACHE[cache_key] = compiled
        _PREFILL_COMPILE_CACHE.move_to_end(cache_key)
        while len(_PREFILL_COMPILE_CACHE) > _PREFILL_COMPILE_CACHE_MAX_ENTRIES:
            _PREFILL_COMPILE_CACHE.popitem(last=False)


def _prefill_compile_error(cache_key: tuple) -> str | None:
    with _PREFILL_COMPILE_CACHE_LOCK:
        message = _PREFILL_COMPILE_ERRORS.get(cache_key)
        if message is not None:
            _PREFILL_COMPILE_ERRORS.move_to_end(cache_key)
        return message


def _prefill_compile_error_store(cache_key: tuple, message: str) -> None:
    with _PREFILL_COMPILE_CACHE_LOCK:
        _PREFILL_COMPILE_ERRORS[cache_key] = message
        _PREFILL_COMPILE_ERRORS.move_to_end(cache_key)
        while len(_PREFILL_COMPILE_ERRORS) > _PREFILL_COMPILE_CACHE_MAX_ENTRIES:
            _PREFILL_COMPILE_ERRORS.popitem(last=False)


def _cache_key_talker_id(cache_key: tuple) -> int | None:
    if not cache_key:
        return None
    value = cache_key[0]
    return value if isinstance(value, int) else None


def _normalize_prefill_mask_mode(prefill_mask_mode: str) -> str:
    mode = str(prefill_mask_mode or "auto").strip().lower()
    if mode not in {"auto", "skip", "explicit"}:
        raise ValueError(
            f"Unsupported prefill_mask_mode {prefill_mask_mode!r}. "
            "Expected 'auto', 'skip', or 'explicit'."
        )
    if mode == "auto":
        return "explicit"
    return mode


def _validate_prefill_configuration(prefill_backend: str, prefill_mask_mode: str) -> None:
    if prefill_backend == "eager":
        return
    if prefill_mask_mode != "skip":
        raise UnsupportedPrefillConfiguration(
            "Compiled prefill requires verified mask skip; use eager for explicit masks."
        )


def _validate_prefill_compile_compat_configuration(
    *,
    prefill_backend: str,
    prefill_mask_mode: str,
    prefill_compile_compat_mode: str,
    talker_input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    input_metadata: Optional[dict],
) -> None:
    if prefill_compile_compat_mode == "none":
        return
    validate_strict_bf16_sdpa_v1(
        prefill_backend=prefill_backend,
        prefill_mask_mode=prefill_mask_mode,
        talker_input_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        input_metadata=input_metadata,
    )


def select_prefill_mask_mode(input_metadata: Optional[dict]) -> str:
    if not input_metadata:
        return "explicit"
    if input_metadata.get("prefill_attention_mask_all_valid") is not True:
        return "explicit"
    if input_metadata.get("prefill_mask_decision_source") != "constructed_all_ones":
        return "explicit"
    if input_metadata.get("prefill_batch_size") != 1:
        return "explicit"
    if input_metadata.get("prefill_has_sliding_window") is True:
        return "explicit"
    if input_metadata.get("prefill_attn_implementation") not in {"eager", "sdpa"}:
        return "explicit"
    return "skip"


def _talker_prefill_eager(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    *,
    skip_prefill_causal_mask: bool = False,
):
    return talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
        skip_prefill_causal_mask=skip_prefill_causal_mask,
    )


def _compile_talker_prefill(talker, prefill_backend: str) -> Callable:
    backend = "inductor"
    mode = None
    if prefill_backend == "compile_backend_eager":
        backend = "eager"
    elif prefill_backend == "compile_backend_aot_eager":
        backend = "aot_eager"
    elif prefill_backend == "compile_reduce_overhead":
        mode = "reduce-overhead"
    fullgraph = prefill_backend != "compile_inductor_graphbreak"

    def prefill_fn(
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        trailing_text_hiddens: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        skip_prefill_causal_mask: bool,
    ):
        return _talker_prefill_eager(
            talker,
            inputs_embeds,
            attention_mask,
            trailing_text_hiddens,
            tts_pad_embed,
            skip_prefill_causal_mask=skip_prefill_causal_mask,
        )

    return torch.compile(
        prefill_fn,
        backend=backend,
        fullgraph=fullgraph,
        dynamic=False,
        mode=mode,
    )


def _prefill_compile_cache_key(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    prefill_backend: str,
    prefill_mask_mode: str,
    prefill_compile_compat_mode: str,
) -> tuple:
    return (
        id(talker),
        prefill_backend,
        prefill_mask_mode,
        prefill_compile_compat_mode,
        _tensor_signature(talker_input_embeds),
        _tensor_signature(attention_mask),
        _tensor_signature(trailing_text_hiddens),
        _tensor_signature(tts_pad_embed),
    )


def _tensor_signature(tensor: Optional[torch.Tensor]) -> tuple:
    if tensor is None:
        return ("none",)
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        bool(tensor.requires_grad),
    )


@torch.inference_mode()
def fast_generate_streaming(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    chunk_size: int = 12,
    input_metadata: Optional[dict] = None,
    profile_prefill: bool = False,
    profile_nvtx: bool = False,
    prefill_backend: str = "eager",
    prefill_mask_mode: str = "auto",
    prefill_compile_compat_mode: str = "none",
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Streaming autoregressive generation with CUDA-graphed predictor and talker.

    Yields (codec_chunk, timing_info) tuples every chunk_size steps.
    codec_chunk: [chunk_steps, 16] tensor of codec IDs.
    The final chunk may be shorter than chunk_size.
    """
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = build_suppress_mask(vocab_size, eos_id, device)
    eos_suppress_ids = torch.tensor([eos_id], dtype=torch.long, device=device)

    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    fused_codec_weights, fused_codec_offsets = get_fused_codec_embeddings(predictor)
    prefill_backend = _normalize_prefill_backend(prefill_backend)
    prefill_compile_compat_mode = normalize_prefill_compile_compat_mode(
        prefill_compile_compat_mode
    )
    if str(prefill_mask_mode or "auto").strip().lower() == "auto":
        prefill_mask_mode = select_prefill_mask_mode(input_metadata)
    else:
        prefill_mask_mode = _normalize_prefill_mask_mode(prefill_mask_mode)
    _validate_prefill_configuration(prefill_backend, prefill_mask_mode)

    # === PREFILL (still uses HF forward for variable-length prefill) ===
    t_start = time.perf_counter()
    prefill_profile = {}
    prefill_events = _PrefillEvents(device, profile_prefill)
    nvtx_enabled = profile_nvtx and device.type == "cuda"
    outer_nvtx_name = _profile_outer_nvtx_name(input_metadata)
    prefill_events.record("start")
    outer_nvtx_range = _CudaNvtxRange(
        outer_nvtx_name,
        nvtx_enabled and outer_nvtx_name is not None,
    )
    outer_nvtx_range.__enter__()

    forward_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_talker_forward", nvtx_enabled):
        out, backend_profile = _run_talker_prefill(
            talker,
            talker_input_embeds,
            attention_mask,
            trailing_text_hiddens,
            tts_pad_embed,
            prefill_backend=prefill_backend,
            prefill_mask_mode=prefill_mask_mode,
            prefill_compile_compat_mode=prefill_compile_compat_mode,
            input_metadata=input_metadata,
        )
    prefill_profile["talker_forward_launch_wall_ms"] = (
        time.perf_counter() - forward_started
    ) * 1000
    prefill_profile.update(backend_profile)
    prefill_events.record("after_forward")

    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step

    sample_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_first_sample", nvtx_enabled):
        logits = out.logits[:, -1, :]
        suppress_eos = min_new_tokens > 0
        token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )
    prefill_profile["first_sample_launch_wall_ms"] = (
        time.perf_counter() - sample_started
    ) * 1000
    prefill_events.record("after_sample")

    prefill_kv_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_static_cache", nvtx_enabled):
        prefill_len = talker_graph.prefill_kv(talker_past_kv)
    prefill_profile["prefill_kv_launch_wall_ms"] = (
        time.perf_counter() - prefill_kv_started
    ) * 1000
    prefill_events.record("after_prefill_kv")

    generation_state_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_generation_state", nvtx_enabled):
        rope_deltas = getattr(talker, "rope_deltas", None)
        talker_graph.set_generation_state(attention_mask, rope_deltas)
    prefill_profile["generation_state_wall_ms"] = (
        time.perf_counter() - generation_state_started
    ) * 1000
    prefill_events.record("after_generation_state")

    # Deferred EOS detection (see fast_generate): tokens are copied to a pinned
    # host buffer asynchronously and checked one iteration late so the CPU never
    # blocks on the GPU mid-step. Slot k%2 holds the token consumed by iteration
    # k. Chunk flushes additionally validate their tail entry (after the sync
    # they already perform) so an EOS overshoot is never yielded downstream.
    token_cpu, token_events = get_eos_tracker(device)
    token_cpu[0:1].copy_(token, non_blocking=True)
    token_events[0].record()
    prefill_events.record("before_sync")

    sync_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_final_sync", nvtx_enabled):
        torch.cuda.synchronize()
    outer_nvtx_range.__exit__(None, None, None)
    prefill_profile["prefill_sync_wait_ms"] = (
        time.perf_counter() - sync_started
    ) * 1000
    t_prefill = time.perf_counter() - t_start
    prefill_profile.update(
        _finalize_prefill_profile(
            prefill_events.elapsed_ms(profile_path="fast"),
            profile_prefill,
            profile_path="fast",
        )
    )

    # === DECODE LOOP — yield chunks ===
    chunk_buffer = []
    # Preallocated first-codebook history for repetition penalty across chunks;
    # rebuilding it with torch.stack over a growing list is O(n) launches per step.
    rep_history = None
    if repetition_penalty != 1.0:
        rep_history = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    total_steps = 0
    chunk_count = 0
    eos_found = False
    termination = {
        "termination_reason": "max_new_tokens",
        "hit_eos": False,
        "hit_max_new_tokens": False,
        "hit_max_seq_len": False,
        "terminal_token_id": None,
        "terminal_step_index": None,
        "generator_loop_iterations": 0,
        "generated_steps": 0,
        "emitted_steps": 0,
    }
    chunk_start = time.time()

    for step_idx in range(max_new_tokens):
        termination["generator_loop_iterations"] = step_idx + 1
        if step_idx > 0:
            prev_slot = (step_idx - 1) % 2
            token_events[prev_slot].synchronize()
            if int(token_cpu[prev_slot]) == eos_id:
                # Previous iteration consumed EOS — drop its output and stop.
                # The entry is always still in chunk_buffer: a flush validates
                # its tail before yielding, so an EOS entry never leaves it.
                chunk_buffer.pop()
                eos_found = True
                termination.update(
                    {
                        "termination_reason": "eos",
                        "hit_eos": True,
                        "terminal_token_id": eos_id,
                        "terminal_step_index": step_idx - 1,
                    }
                )
                break
        cur_slot = step_idx % 2
        if token_events[cur_slot].query() and int(token_cpu[cur_slot]) == eos_id:
            # This iteration's own token is already visible and is EOS — stop
            # before doing any work (no overshoot).
            eos_found = True
            termination.update(
                {
                    "termination_reason": "eos",
                    "hit_eos": True,
                    "terminal_token_id": eos_id,
                    "terminal_step_index": step_idx,
                }
            )
            break

        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = predictor_graph.run(pred_input)

        all_cb = torch.cat([token.view(1), codebook_token_ids])
        chunk_buffer.append(all_cb.detach())
        if rep_history is not None:
            rep_history[step_idx : step_idx + 1] = token

        # --- Build input embedding for talker ---
        # One fused gather over all 15 codebook tables; the cat+sum keeps the
        # exact reduction order of the previous per-table loop for parity.
        codebook_embeds = F.embedding(
            codebook_token_ids + fused_codec_offsets, fused_codec_weights
        ).unsqueeze(0)  # [1, 15, H]
        inputs_embeds = torch.cat((last_id_hidden, codebook_embeds), dim=1).sum(
            1, keepdim=True
        )

        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[
                :, gen_step
            ].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        # --- CUDA-Graphed Talker decode step ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            termination.update(
                {
                    "termination_reason": "max_seq_len",
                    "hit_max_seq_len": True,
                    "terminal_step_index": step_idx,
                }
            )
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)

        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

        if rep_history is not None:
            logits = apply_repetition_penalty(
                logits, rep_history[: step_idx + 1], repetition_penalty
            )

        suppress_eos = step_idx + 1 < min_new_tokens
        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )
        next_slot = (step_idx + 1) % 2
        token_cpu[next_slot : next_slot + 1].copy_(token, non_blocking=True)
        token_events[next_slot].record()
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

        # --- Yield chunk when buffer is full ---
        if len(chunk_buffer) >= chunk_size:
            torch.cuda.synchronize()
            # This chunk's tail entry hasn't been EOS-checked yet (that happens
            # at the top of the next iteration); validate it now that we're synced.
            if int(token_cpu[step_idx % 2]) == eos_id:
                chunk_buffer.pop()
                eos_found = True
                termination.update(
                    {
                        "termination_reason": "eos",
                        "hit_eos": True,
                        "terminal_token_id": eos_id,
                        "terminal_step_index": step_idx,
                    }
                )
                if not chunk_buffer:
                    break
            elif step_idx + 1 >= max_new_tokens:
                termination.update(
                    {
                        "termination_reason": "max_new_tokens",
                        "hit_max_new_tokens": True,
                        "terminal_step_index": step_idx,
                    }
                )
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)
            is_final_chunk = eos_found or step_idx + 1 >= max_new_tokens

            yield (
                torch.stack(chunk_buffer),
                _chunk_timing(
                    {
                        "chunk_index": chunk_count,
                        "chunk_steps": len(chunk_buffer),
                        "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                        "decode_ms": chunk_decode_time * 1000,
                        "total_steps_so_far": total_steps,
                        "is_final": is_final_chunk,
                        **_termination_telemetry(
                            termination,
                            generated_steps=total_steps,
                            emitted_steps=total_steps,
                            include=is_final_chunk,
                        ),
                    },
                    input_metadata=input_metadata,
                    prefill_profile=prefill_profile if chunk_count == 0 else None,
                ),
            )

            chunk_buffer = []
            if eos_found:
                break
            chunk_count += 1
            chunk_start = time.time()

    # --- Yield final partial chunk ---
    if chunk_buffer:
        torch.cuda.synchronize()
        if not eos_found:
            # Loop exited on budget/seq-len with the newest entry unchecked;
            # slots are keyed by global iteration index.
            tail_idx = total_steps + len(chunk_buffer) - 1
            if int(token_cpu[tail_idx % 2]) == eos_id:
                chunk_buffer.pop()
                eos_found = True
                termination.update(
                    {
                        "termination_reason": "eos",
                        "hit_eos": True,
                        "terminal_token_id": eos_id,
                        "terminal_step_index": tail_idx,
                    }
                )
            elif total_steps + len(chunk_buffer) >= max_new_tokens:
                termination.update(
                    {
                        "termination_reason": "max_new_tokens",
                        "hit_max_new_tokens": True,
                        "terminal_step_index": total_steps + len(chunk_buffer) - 1,
                    }
                )

    if chunk_buffer:
        chunk_decode_time = time.time() - chunk_start
        total_steps += len(chunk_buffer)

        yield (
            torch.stack(chunk_buffer),
            _chunk_timing(
                {
                    "chunk_index": chunk_count,
                    "chunk_steps": len(chunk_buffer),
                    "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                    "decode_ms": chunk_decode_time * 1000,
                    "total_steps_so_far": total_steps,
                    "is_final": True,
                    **_termination_telemetry(
                        termination,
                        generated_steps=total_steps,
                        emitted_steps=total_steps,
                        include=True,
                    ),
                },
                input_metadata=input_metadata,
                prefill_profile=prefill_profile if chunk_count == 0 else None,
            ),
        )


def _profile_outer_nvtx_name(input_metadata: Optional[dict]) -> Optional[str]:
    if not input_metadata:
        return None
    request_role = input_metadata.get("profile_request_role")
    if request_role == "first_user":
        return "qtb_profile_first_user_prefill"
    if request_role == "steady":
        return "qtb_profile_steady_prefill"
    return None


class _PrefillEvents:
    _KNOWN_EVENTS = (
        "start",
        "after_forward",
        "after_sample",
        "after_prefill_kv",
        "after_generation_state",
        "before_sync",
    )

    def __init__(self, device, enabled: bool):
        self._enabled = enabled and device.type == "cuda"
        self._events = {
            name: torch.cuda.Event(enable_timing=True)
            for name in self._KNOWN_EVENTS
        } if self._enabled else {}
        self._stream_ids = {}

    def record(self, name: str) -> None:
        if not self._enabled:
            return
        stream = torch.cuda.current_stream()
        event = self._events.get(name)
        if event is None:
            event = torch.cuda.Event(enable_timing=True)
            self._events[name] = event
        event.record(stream)
        self._stream_ids[name] = int(stream.cuda_stream)

    def elapsed_ms(self, *, profile_path: str) -> dict:
        if not self._enabled:
            return {}
        event_profile = {
            "prefill_total_gpu_ms": self._elapsed("start", "before_sync"),
            "talker_forward_gpu_ms": self._elapsed("start", "after_forward"),
            "first_sample_gpu_ms": self._elapsed("after_forward", "after_sample"),
            "prefill_total_gpu_stream_id": self._stream_id("start"),
            "talker_forward_gpu_stream_id": self._stream_id("start"),
            "first_sample_gpu_stream_id": self._stream_id("after_forward"),
            "generation_state_gpu_stream_id": self._stream_id("after_sample"),
            "prefill_to_sync_gpu_stream_id": self._stream_id(
                "after_generation_state"
            ),
        }
        if profile_path == "fast":
            event_profile.update(
                {
                    "prefill_kv_gpu_ms": self._elapsed(
                        "after_sample",
                        "after_prefill_kv",
                    ),
                    "generation_state_gpu_ms": self._elapsed(
                        "after_prefill_kv",
                        "after_generation_state",
                    ),
                    "prefill_kv_gpu_stream_id": self._stream_id("after_sample"),
                    "generation_state_gpu_stream_id": self._stream_id(
                        "after_prefill_kv"
                    ),
                }
            )
        else:
            event_profile.update(
                {
                    "prefill_kv_gpu_ms": None,
                    "prefill_kv_gpu_stream_id": None,
                    "generation_state_gpu_ms": self._elapsed(
                        "after_sample",
                        "after_generation_state",
                    ),
                }
            )
        event_profile["prefill_to_sync_gpu_ms"] = self._elapsed(
            "after_generation_state",
            "before_sync",
        )
        for name in self._events:
            event_profile[f"{name}_event_recorded"] = True
        return event_profile

    def _elapsed(self, start_name: str, end_name: str) -> float | None:
        start = self._events.get(start_name)
        end = self._events.get(end_name)
        if start is None or end is None:
            return None
        return float(start.elapsed_time(end))

    def _stream_id(self, name: str) -> int | None:
        return self._stream_ids.get(name)


def _finalize_prefill_profile(
    event_profile: dict,
    profile_prefill: bool,
    *,
    profile_path: str,
) -> dict:
    if not profile_prefill:
        return _disabled_prefill_profile(profile_path)

    profile = {
        "profile_schema_version": 3,
        "profile_prefill_enabled": bool(profile_prefill),
        "profile_status": "complete",
        "profile_path": profile_path,
    }
    profile.update(event_profile)

    if profile_path == "fast":
        component_keys = (
            "talker_forward_gpu_ms",
            "first_sample_gpu_ms",
            "prefill_kv_gpu_ms",
            "generation_state_gpu_ms",
            "prefill_to_sync_gpu_ms",
        )
        required_event_names = (
            "start",
            "after_forward",
            "after_sample",
            "after_prefill_kv",
            "after_generation_state",
            "before_sync",
        )
        stream_id_keys = (
            "prefill_total_gpu_stream_id",
            "talker_forward_gpu_stream_id",
            "first_sample_gpu_stream_id",
            "prefill_kv_gpu_stream_id",
            "generation_state_gpu_stream_id",
            "prefill_to_sync_gpu_stream_id",
        )
    elif profile_path == "parity":
        component_keys = (
            "talker_forward_gpu_ms",
            "first_sample_gpu_ms",
            "generation_state_gpu_ms",
            "prefill_to_sync_gpu_ms",
        )
        required_event_names = (
            "start",
            "after_forward",
            "after_sample",
            "after_generation_state",
            "before_sync",
        )
        stream_id_keys = (
            "prefill_total_gpu_stream_id",
            "talker_forward_gpu_stream_id",
            "first_sample_gpu_stream_id",
            "generation_state_gpu_stream_id",
            "prefill_to_sync_gpu_stream_id",
        )
    else:
        component_keys = ()
        required_event_names = ()
        stream_id_keys = ()

    for key in (
        "prefill_total_gpu_ms",
        "talker_forward_gpu_ms",
        "first_sample_gpu_ms",
        "prefill_kv_gpu_ms",
        "generation_state_gpu_ms",
        "prefill_to_sync_gpu_ms",
        "prefill_total_gpu_stream_id",
        "talker_forward_gpu_stream_id",
        "first_sample_gpu_stream_id",
        "prefill_kv_gpu_stream_id",
        "generation_state_gpu_stream_id",
        "prefill_to_sync_gpu_stream_id",
    ):
        profile.setdefault(key, None)
    for name in required_event_names:
        profile.setdefault(f"{name}_event_recorded", False)

    profile["events_complete"] = all(
        profile.get(f"{name}_event_recorded") is True
        for name in required_event_names
    )
    component_values = [profile.get(key) for key in component_keys]
    components_complete = all(
        isinstance(value, (int, float)) for value in component_values
    )
    profile["components_finite"] = components_complete and all(
        math.isfinite(float(value)) for value in component_values
    )
    profile["components_nonnegative"] = components_complete and all(
        float(value) >= 0.0 for value in component_values
    )
    stream_ids = [profile.get(key) for key in stream_id_keys]
    present_stream_ids = [
        int(value) for value in stream_ids if isinstance(value, int)
    ]
    profile["all_component_streams_equal"] = (
        len(present_stream_ids) == len(stream_id_keys)
        and len(set(present_stream_ids)) == 1
    )
    total = profile.get("prefill_total_gpu_ms")
    if components_complete and isinstance(total, (int, float)):
        component_sum = float(sum(component_values))
        profile["prefill_gpu_component_sum_ms"] = component_sum
        partition_error = float(total) - component_sum
        profile["prefill_gpu_partition_error_ms"] = partition_error
        profile["prefill_gpu_accounting_error_ms"] = partition_error
    else:
        profile["prefill_gpu_component_sum_ms"] = None
        profile["prefill_gpu_partition_error_ms"] = None
        profile["prefill_gpu_accounting_error_ms"] = None

    total_complete = (
        isinstance(total, (int, float))
        and math.isfinite(float(total))
        and float(total) >= 0.0
    )
    profile["profile_complete"] = bool(
        profile_prefill
        and profile["events_complete"]
        and total_complete
        and profile["components_finite"]
        and profile["components_nonnegative"]
        and profile["all_component_streams_equal"]
    )
    if not profile["profile_complete"]:
        profile["profile_status"] = "incomplete"
    return profile


def _disabled_prefill_profile(profile_path: str) -> dict:
    profile = {
        "profile_schema_version": 3,
        "profile_prefill_enabled": False,
        "profile_status": "disabled",
        "profile_path": profile_path,
    }
    for key in (
        "prefill_total_gpu_ms",
        "talker_forward_gpu_ms",
        "first_sample_gpu_ms",
        "prefill_kv_gpu_ms",
        "generation_state_gpu_ms",
        "prefill_to_sync_gpu_ms",
        "prefill_total_gpu_stream_id",
        "talker_forward_gpu_stream_id",
        "first_sample_gpu_stream_id",
        "prefill_kv_gpu_stream_id",
        "generation_state_gpu_stream_id",
        "prefill_to_sync_gpu_stream_id",
        "prefill_gpu_component_sum_ms",
        "prefill_gpu_partition_error_ms",
        "prefill_gpu_accounting_error_ms",
        "profile_complete",
        "events_complete",
        "components_finite",
        "components_nonnegative",
        "all_component_streams_equal",
    ):
        profile[key] = None
    return profile


def _chunk_timing(
    timing: dict,
    *,
    input_metadata: Optional[dict],
    prefill_profile: Optional[dict],
) -> dict:
    if input_metadata:
        timing.update(input_metadata)
    if prefill_profile:
        timing.update(prefill_profile)
    return timing


def _termination_telemetry(
    termination: dict,
    *,
    generated_steps: int,
    emitted_steps: int,
    include: bool,
) -> dict:
    if not include:
        return {}
    telemetry = dict(termination)
    telemetry["generated_steps"] = generated_steps
    telemetry["emitted_steps"] = emitted_steps
    return telemetry


@torch.inference_mode()
def parity_generate_streaming(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    chunk_size: int = 12,
    input_metadata: Optional[dict] = None,
    profile_prefill: bool = False,
    profile_nvtx: bool = False,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Streaming generation without CUDA graphs (dynamic cache).

    Yields (codec_chunk, timing_info) tuples every chunk_size steps.
    """
    # NOTE: This function intentionally mirrors fast_generate_streaming. The core
    # decode loop is duplicated so we can swap CUDA graphs/static cache for the
    # dynamic-cache path while keeping sampling/chunking identical. If you edit
    # the fast path, check parity_generate_streaming for matching changes.
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = build_suppress_mask(vocab_size, eos_id, device)
    eos_suppress_ids = torch.tensor([eos_id], dtype=torch.long, device=device)

    # === PREFILL ===
    t_start = time.perf_counter()
    prefill_profile = {}
    prefill_events = _PrefillEvents(device, profile_prefill)
    nvtx_enabled = profile_nvtx and device.type == "cuda"
    outer_nvtx_name = _profile_outer_nvtx_name(input_metadata)
    prefill_events.record("start")
    outer_nvtx_range = _CudaNvtxRange(
        outer_nvtx_name,
        nvtx_enabled and outer_nvtx_name is not None,
    )
    outer_nvtx_range.__enter__()

    forward_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_talker_forward", nvtx_enabled):
        out = talker.forward(
            inputs_embeds=talker_input_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=None,
            past_hidden=None,
            past_key_values=None,
        )
    prefill_profile["talker_forward_launch_wall_ms"] = (
        time.perf_counter() - forward_started
    ) * 1000
    prefill_events.record("after_forward")

    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step

    sample_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_first_sample", nvtx_enabled):
        logits = out.logits[:, -1, :]
        suppress_eos = min_new_tokens > 0
        token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )
    prefill_profile["first_sample_launch_wall_ms"] = (
        time.perf_counter() - sample_started
    ) * 1000
    prefill_events.record("after_sample")

    generation_state_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_generation_state", nvtx_enabled):
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
    prefill_profile["generation_state_wall_ms"] = (
        time.perf_counter() - generation_state_started
    ) * 1000
    prefill_events.record("after_generation_state")

    prefill_events.record("before_sync")
    sync_started = time.perf_counter()
    with _CudaNvtxRange("qtb_prefill_final_sync", nvtx_enabled):
        torch.cuda.synchronize()
    outer_nvtx_range.__exit__(None, None, None)
    prefill_profile["prefill_sync_wait_ms"] = (
        time.perf_counter() - sync_started
    ) * 1000
    t_prefill = time.perf_counter() - t_start
    prefill_profile.update(
        _finalize_prefill_profile(
            prefill_events.elapsed_ms(profile_path="parity"),
            profile_prefill,
            profile_path="parity",
        )
    )

    # === DECODE LOOP — yield chunks ===
    chunk_buffer = []
    all_first_tokens = []
    total_steps = 0
    chunk_count = 0
    chunk_start = time.time()

    for _ in range(max_new_tokens):
        if token.item() == eos_id:
            break

        cache_position = None
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )
            cache_position = torch.tensor(
                [attention_mask.shape[1] - 1], device=attention_mask.device
            )

        out = talker.forward(
            input_ids=token.view(1, 1),
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=gen_step,
            past_hidden=past_hidden,
            past_key_values=talker_past_kv,
            subtalker_dosample=do_sample,
            subtalker_top_k=top_k,
            subtalker_top_p=top_p,
            subtalker_temperature=temperature,
            cache_position=cache_position,
        )

        codec_ids = out.hidden_states[1]
        if codec_ids is None:
            break

        chunk_buffer.append(codec_ids.squeeze(0).detach())
        all_first_tokens.append(token.detach())

        logits = out.logits[:, -1, :]
        if repetition_penalty != 1.0 and all_first_tokens:
            history = torch.stack(all_first_tokens)
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = len(all_first_tokens) < min_new_tokens
        token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )

        talker_past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        if len(chunk_buffer) >= chunk_size:
            torch.cuda.synchronize()
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)

            yield (
                torch.stack(chunk_buffer),
                _chunk_timing(
                    {
                        "chunk_index": chunk_count,
                        "chunk_steps": len(chunk_buffer),
                        "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                        "decode_ms": chunk_decode_time * 1000,
                        "total_steps_so_far": total_steps,
                        "is_final": False,
                    },
                    input_metadata=input_metadata,
                    prefill_profile=prefill_profile if chunk_count == 0 else None,
                ),
            )

            chunk_buffer = []
            chunk_count += 1
            chunk_start = time.time()

    if chunk_buffer:
        torch.cuda.synchronize()
        chunk_decode_time = time.time() - chunk_start
        total_steps += len(chunk_buffer)

        yield (
            torch.stack(chunk_buffer),
            _chunk_timing(
                {
                    "chunk_index": chunk_count,
                    "chunk_steps": len(chunk_buffer),
                    "prefill_ms": t_prefill * 1000 if chunk_count == 0 else 0,
                    "decode_ms": chunk_decode_time * 1000,
                    "total_steps_so_far": total_steps,
                    "is_final": True,
                },
                input_metadata=input_metadata,
                prefill_profile=prefill_profile if chunk_count == 0 else None,
            ),
        )
