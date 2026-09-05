"""Shared sampling helpers for talker and predictor generation."""
from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F

_SUPPRESS_MASK_CACHE: dict = {}


def build_suppress_mask(vocab_size: int, eos_id: int, device) -> torch.Tensor:
    """Boolean mask over the suppressed special-token tail [vocab-1024, vocab), excluding EOS.

    Cached per (vocab_size, eos_id, device); callers must not mutate the result.
    """
    key = (vocab_size, eos_id, str(device))
    mask = _SUPPRESS_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        mask[max(0, vocab_size - 1024):] = True
        mask[eos_id] = False
        _SUPPRESS_MASK_CACHE[key] = mask
    return mask


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_history: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits in-place and return them.

    Args:
        logits: Tensor shaped [1, 1, vocab] or [1, vocab].
        token_history: 1-D tensor of previously generated token ids.
        repetition_penalty: HF-style repetition penalty (>1.0).
    """
    if repetition_penalty == 1.0 or token_history.numel() == 0:
        return logits
    # ``Tensor.unique()`` materializes a dynamically sized result and forces a
    # host synchronization on the sealed CMP runtime (about 28 ms per AR
    # frame).  A fixed-size GPU mask preserves the HF semantics for duplicate
    # history entries without a synchronization boundary.
    flat_logits = logits.reshape(-1)
    seen = torch.zeros_like(flat_logits, dtype=torch.bool)
    seen.scatter_(0, token_history, True)
    penalized = torch.where(
        flat_logits > 0,
        flat_logits / repetition_penalty,
        flat_logits * repetition_penalty,
    )
    flat_logits.copy_(torch.where(seen, penalized, flat_logits))
    return logits


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int] | torch.Tensor] = None,
    phase_timer: Any | None = None,
    phase_prefix: str = "",
) -> torch.Tensor:
    """Sample a token from logits.

    Mirrors HF order: suppress -> temperature -> top-k -> top-p -> sample.

    suppress_tokens may be a 1-D index tensor (preferred in hot loops — avoids a
    host-to-device copy per call) or any iterable of ints.
    """
    clone_suppress = _begin_phase(phase_timer, phase_prefix, "clone_suppress")
    logits = logits.clone()
    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens is not None:
        if not isinstance(suppress_tokens, torch.Tensor):
            suppress_tokens = torch.tensor(list(suppress_tokens), dtype=torch.long, device=logits.device)
        if suppress_tokens.numel() > 0:
            logits[..., suppress_tokens] = float("-inf")
    _end_phase(phase_timer, phase_prefix, "clone_suppress", clone_suppress)
    if not do_sample:
        argmax = _begin_phase(phase_timer, phase_prefix, "argmax")
        token = torch.argmax(logits, dim=-1)
        _end_phase(phase_timer, phase_prefix, "argmax", argmax)
        return token
    temperature_phase = _begin_phase(phase_timer, phase_prefix, "temperature")
    logits = logits / temperature
    _end_phase(phase_timer, phase_prefix, "temperature", temperature_phase)
    if top_k > 0:
        top_k_phase = _begin_phase(phase_timer, phase_prefix, "top_k")
        topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(
            logits < topk_vals[..., -1:],
            torch.full_like(logits, float("-inf")),
            logits,
        )
        _end_phase(phase_timer, phase_prefix, "top_k", top_k_phase)
    if top_p < 1.0:
        top_p_phase = _begin_phase(phase_timer, phase_prefix, "top_p")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)
        _end_phase(phase_timer, phase_prefix, "top_p", top_p_phase)
    softmax_phase = _begin_phase(phase_timer, phase_prefix, "softmax")
    probabilities = F.softmax(logits, dim=-1)
    _end_phase(phase_timer, phase_prefix, "softmax", softmax_phase)
    multinomial_phase = _begin_phase(phase_timer, phase_prefix, "multinomial")
    token = torch.multinomial(probabilities, 1).squeeze(-1)
    _end_phase(phase_timer, phase_prefix, "multinomial", multinomial_phase)
    return token


def _begin_phase(phase_timer: Any | None, prefix: str, name: str) -> Any | None:
    if phase_timer is None:
        return None
    return phase_timer.begin(_phase_name(prefix, name))


def _end_phase(
    phase_timer: Any | None,
    prefix: str,
    name: str,
    start: Any | None,
) -> None:
    if phase_timer is not None:
        phase_timer.end(_phase_name(prefix, name), start)


def _phase_name(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name
