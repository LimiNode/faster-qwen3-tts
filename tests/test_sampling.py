import types

import pytest
import torch

from faster_qwen3_tts.generate import fast_generate
from faster_qwen3_tts.model import FasterQwen3TTS
from faster_qwen3_tts.sampling import apply_repetition_penalty
from faster_qwen3_tts.sampling import sample_logits
from faster_qwen3_tts import streaming
from faster_qwen3_tts.streaming import _PREFILL_BACKENDS
from faster_qwen3_tts.streaming import _run_talker_prefill
from faster_qwen3_tts.streaming import UnsupportedPrefillConfiguration
from faster_qwen3_tts.streaming import select_prefill_mask_mode


def test_repetition_penalty_uses_all_history():
    logits = torch.zeros(1, 1, 10)
    logits[..., 7] = 1.0
    logits[..., 8] = -1.0

    others = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    history = [7] + [others[i % len(others)] for i in range(1, 60)]
    history = torch.tensor(history, dtype=torch.long)

    out = apply_repetition_penalty(logits.clone(), history, repetition_penalty=1.1)
    assert pytest.approx(out[0, 0, 7].item(), rel=1e-6) == 1.0 / 1.1
    assert pytest.approx(out[0, 0, 8].item(), rel=1e-6) == -1.0 * 1.1


def test_sample_logits_greedy_does_not_call_multinomial(monkeypatch):
    def fail_multinomial(*args, **kwargs):
        raise AssertionError("greedy sampling must not call torch.multinomial")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)

    logits = torch.tensor([[0.1, 1.0, 0.5]])
    token = sample_logits(
        logits,
        temperature=0.9,
        top_k=50,
        top_p=1.0,
        do_sample=False,
    )

    assert token.tolist() == [1]


def test_faster_wrapper_selects_greedy_predictor_graph():
    sampling_graph = types.SimpleNamespace(do_sample=True)
    greedy_graph = types.SimpleNamespace(do_sample=False)
    model = FasterQwen3TTS.__new__(FasterQwen3TTS)
    model.predictor_graph = sampling_graph
    model.predictor_graph_greedy = greedy_graph

    assert model._select_predictor_graph(True) is sampling_graph
    assert model._select_predictor_graph(False) is greedy_graph

    model.predictor_graph_greedy = None
    with pytest.raises(RuntimeError, match="Greedy PredictorGraph is unavailable"):
        model._select_predictor_graph(False)


def test_faster_wrapper_resets_partial_generation_graph_state():
    class ResetGraph:
        def __init__(self):
            self.reset_calls = []

        def reset(self, *args):
            self.reset_calls.append(args)

    model = FasterQwen3TTS.__new__(FasterQwen3TTS)
    model.talker_graph = ResetGraph()
    model.predictor_graph = ResetGraph()
    model.predictor_graph_greedy = ResetGraph()

    metadata = model.reset_after_partial_generation()

    assert model.talker_graph.reset_calls == [(0,)]
    assert model.predictor_graph.reset_calls == [()]
    assert model.predictor_graph_greedy.reset_calls == [()]
    assert metadata == {
        "reset_api_version": 1,
        "talker_graph_reset": True,
        "predictor_graphs_reset": 2,
        "compiled_prefill_cache_preserved": True,
        "cuda_graphs_preserved": True,
        "generation_mask_cache_preserved": True,
    }


def test_faster_wrapper_uses_loaded_prefill_compile_compat_mode_by_default():
    model = FasterQwen3TTS.__new__(FasterQwen3TTS)
    model.prefill_compile_compat_mode = "strict_bf16_sdpa_v1"

    assert model._resolve_prefill_compile_compat_mode(None) == "strict_bf16_sdpa_v1"
    assert (
        model._resolve_prefill_compile_compat_mode("strict_bf16_sdpa_v1")
        == "strict_bf16_sdpa_v1"
    )
    with pytest.raises(RuntimeError, match="immutable"):
        model._resolve_prefill_compile_compat_mode("none")


def test_faster_wrapper_can_make_prefill_backend_immutable():
    model = FasterQwen3TTS.__new__(FasterQwen3TTS)
    model.prefill_backend = "compile_reduce_overhead"

    assert model._resolve_prefill_backend(None) == "compile_reduce_overhead"
    assert (
        model._resolve_prefill_backend("compile_reduce_overhead")
        == "compile_reduce_overhead"
    )
    with pytest.raises(RuntimeError, match="immutable"):
        model._resolve_prefill_backend("compile_inductor_default")


def test_faster_wrapper_without_loaded_prefill_backend_preserves_request_override():
    model = FasterQwen3TTS.__new__(FasterQwen3TTS)
    model.prefill_backend = None

    assert model._resolve_prefill_backend(None) == "eager"
    assert (
        model._resolve_prefill_backend("compile_backend_eager")
        == "compile_backend_eager"
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_mask_decision_source": "constructed_all_ones",
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "eager",
            },
            "skip",
        ),
        (
            {
                "prefill_attention_mask_all_valid": False,
                "prefill_mask_decision_source": "constructed_left_padded",
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "eager",
            },
            "explicit",
        ),
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "eager",
            },
            "explicit",
        ),
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_mask_decision_source": "constructed_all_ones",
                "prefill_batch_size": 2,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "eager",
            },
            "explicit",
        ),
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_mask_decision_source": "constructed_all_ones",
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": True,
                "prefill_attn_implementation": "eager",
            },
            "explicit",
        ),
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_mask_decision_source": "constructed_all_ones",
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "sdpa",
            },
            "skip",
        ),
        (
            {
                "prefill_attention_mask_all_valid": True,
                "prefill_mask_decision_source": "constructed_all_ones",
                "prefill_batch_size": 1,
                "prefill_has_sliding_window": False,
                "prefill_attn_implementation": "flash_attention_2",
            },
            "explicit",
        ),
        (None, "explicit"),
        ({}, "explicit"),
    ],
)
def test_select_prefill_mask_mode_is_fail_closed(metadata, expected):
    assert select_prefill_mask_mode(metadata) == expected


def test_prefill_attention_mask_metadata_tracks_local_provenance():
    assert FasterQwen3TTS._prefill_attention_mask_metadata(torch.tensor([8])) == {
        "prefill_attention_mask_all_valid": True,
        "prefill_mask_decision_source": "constructed_all_ones",
    }
    assert FasterQwen3TTS._prefill_attention_mask_metadata(torch.tensor([5, 8])) == {
        "prefill_attention_mask_all_valid": False,
        "prefill_mask_decision_source": "constructed_left_padded",
    }
    assert FasterQwen3TTS._prefill_attention_mask_metadata(torch.tensor([])) == {
        "prefill_attention_mask_all_valid": False,
        "prefill_mask_decision_source": "unknown",
    }


def test_run_talker_prefill_passes_static_mask_mode():
    class DummyTalker:
        def __init__(self):
            self.seen = []
            self.masks = []

        def forward(self, **kwargs):
            self.seen.append(kwargs["skip_prefill_causal_mask"])
            self.masks.append(kwargs["attention_mask"])
            hidden = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                logits=torch.zeros(1, hidden.shape[1], 3),
                past_hidden=hidden[:, -1:, :],
                past_key_values=[],
                generation_step=0,
            )

    talker = DummyTalker()
    tie = torch.zeros(1, 2, 4)
    tam = torch.ones(1, 2, dtype=torch.long)
    tth = torch.zeros(1, 1, 4)
    tpe = torch.zeros(1, 1, 4)

    _run_talker_prefill(
        talker,
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="eager",
        prefill_mask_mode="skip",
    )
    _run_talker_prefill(
        talker,
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="eager",
        prefill_mask_mode="explicit",
    )

    assert talker.seen == [True, False]
    assert talker.masks[0] is None
    assert talker.masks[1] is tam


def _dummy_prefill_inputs():
    return (
        torch.zeros(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
        torch.zeros(1, 1, 4),
        torch.zeros(1, 1, 4),
    )


def test_run_talker_prefill_allows_eager_explicit_masks():
    class DummyTalker:
        def forward(self, **kwargs):
            hidden = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                logits=torch.zeros(1, hidden.shape[1], 3),
                past_hidden=hidden[:, -1:, :],
                past_key_values=[],
                generation_step=0,
            )

    talker = DummyTalker()
    tie, tam, tth, tpe = _dummy_prefill_inputs()

    _run_talker_prefill(
        talker,
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="eager",
        prefill_mask_mode="explicit",
    )


@pytest.mark.parametrize(
    "prefill_backend",
    sorted(backend for backend in _PREFILL_BACKENDS if backend != "eager"),
)
def test_run_talker_prefill_rejects_all_compiled_explicit_masks(prefill_backend):
    class DummyTalker:
        def forward(self, **kwargs):
            raise AssertionError("unsafe compiled explicit mask reached model execution")

    tie, tam, tth, tpe = _dummy_prefill_inputs()
    with pytest.raises(UnsupportedPrefillConfiguration):
        _run_talker_prefill(
            DummyTalker(),
            tie,
            tam,
            tth,
            tpe,
            prefill_backend=prefill_backend,
            prefill_mask_mode="explicit",
        )


def test_run_talker_prefill_allows_compiled_verified_skip_masks():
    class DummyTalker:
        def forward(self, **kwargs):
            hidden = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                logits=torch.zeros(1, hidden.shape[1], 3),
                past_hidden=hidden[:, -1:, :],
                past_key_values=[],
                generation_step=0,
            )

    tie, tam, tth, tpe = _dummy_prefill_inputs()
    _run_talker_prefill(
        DummyTalker(),
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="compile_backend_eager",
        prefill_mask_mode="skip",
    )


def test_run_talker_prefill_compiles_allowlisted_length(monkeypatch):
    class DummyTalker:
        def forward(self, **kwargs):
            hidden = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                logits=torch.zeros(1, hidden.shape[1], 3),
                past_hidden=hidden[:, -1:, :],
                past_key_values=[],
                generation_step=0,
            )

    compile_calls = []

    def fake_compile(talker, backend):
        compile_calls.append((talker, backend))

        def compiled(*args):
            return DummyTalker().forward(
                inputs_embeds=args[0],
                attention_mask=args[1],
                trailing_text_hidden=args[2],
                tts_pad_embed=args[3],
                skip_prefill_causal_mask=args[4],
            )

        return compiled

    monkeypatch.setattr(streaming, "_compile_talker_prefill", fake_compile)
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    _out, profile = _run_talker_prefill(
        DummyTalker(),
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="compile_backend_eager",
        prefill_mask_mode="skip",
        prefill_compile_lengths=[tie.shape[1]],
        prefill_compile_on_miss=False,
        prefill_unknown_shape_policy="eager",
    )

    assert len(compile_calls) == 1
    assert profile["prefill_backend_used"] == "compile_backend_eager"
    assert profile["prefill_shape_policy"] == "compiled_allowlist"
    assert profile["prefill_shape_allowlist_hit"] is True


def test_run_talker_prefill_unknown_length_uses_eager_before_compile(monkeypatch):
    class DummyTalker:
        def forward(self, **kwargs):
            hidden = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                logits=torch.zeros(1, hidden.shape[1], 3),
                past_hidden=hidden[:, -1:, :],
                past_key_values=[],
                generation_step=0,
            )

    def fail_compile(*_args, **_kwargs):
        raise AssertionError("unknown shapes must not compile on user path")

    monkeypatch.setattr(streaming, "_compile_talker_prefill", fail_compile)
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    _out, profile = _run_talker_prefill(
        DummyTalker(),
        tie,
        tam,
        tth,
        tpe,
        prefill_backend="compile_backend_eager",
        prefill_mask_mode="skip",
        prefill_compile_lengths=[tie.shape[1] + 1],
        prefill_compile_on_miss=False,
        prefill_unknown_shape_policy="eager",
    )

    assert profile["prefill_backend_used"] == "eager"
    assert profile["prefill_shape_policy"] == "eager_unknown"
    assert profile["prefill_shape_allowlist_hit"] is False
    assert profile["prefill_shape_length"] == tie.shape[1]


def test_run_talker_prefill_unknown_length_can_fail_before_compile(monkeypatch):
    def fail_compile(*_args, **_kwargs):
        raise AssertionError("unknown shapes must fail before compile")

    monkeypatch.setattr(streaming, "_compile_talker_prefill", fail_compile)
    tie, tam, tth, tpe = _dummy_prefill_inputs()

    with pytest.raises(UnsupportedPrefillConfiguration, match="not in"):
        _run_talker_prefill(
            _dummy_streaming_model(),
            tie,
            tam,
            tth,
            tpe,
            prefill_backend="compile_backend_eager",
            prefill_mask_mode="skip",
            prefill_compile_lengths=[tie.shape[1] + 1],
            prefill_compile_on_miss=False,
            prefill_unknown_shape_policy="error",
        )


def _verified_prefill_metadata():
    return {
        "prefill_attention_mask_all_valid": True,
        "prefill_mask_decision_source": "constructed_all_ones",
        "prefill_batch_size": 1,
        "prefill_has_sliding_window": False,
        "prefill_attn_implementation": "sdpa",
    }


def _dummy_streaming_model():
    class DummyConfig:
        codec_eos_token_id = 1
        vocab_size = 8

    class DummyPredictor:
        def get_input_embeddings(self):
            return [torch.nn.Embedding(8, 4) for _ in range(15)]

    class DummyTalker:
        def __init__(self):
            self.config = DummyConfig()
            self.code_predictor = DummyPredictor()
            self.codec_head = torch.nn.Linear(4, 8)
            self.rope_deltas = torch.zeros(1, 1)
            self._embed = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self._embed

    return DummyTalker()


def test_fast_generate_streaming_auto_unknown_metadata_rejects_compiled_prefill():
    talker = _dummy_streaming_model()
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    generator = streaming.fast_generate_streaming(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=talker.config,
        predictor_graph=types.SimpleNamespace(),
        talker_graph=types.SimpleNamespace(),
        input_metadata=None,
        prefill_backend="compile_backend_eager",
        prefill_mask_mode="auto",
    )

    with pytest.raises(UnsupportedPrefillConfiguration):
        next(generator)


def test_fast_generate_streaming_auto_verified_metadata_resolves_skip(monkeypatch):
    class ReachedPrefill(RuntimeError):
        pass

    def capture_prefill(*args, **kwargs):
        assert kwargs["prefill_backend"] == "compile_backend_eager"
        assert kwargs["prefill_mask_mode"] == "skip"
        raise ReachedPrefill

    monkeypatch.setattr(streaming, "_run_talker_prefill", capture_prefill)

    talker = _dummy_streaming_model()
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    generator = streaming.fast_generate_streaming(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=talker.config,
        predictor_graph=types.SimpleNamespace(),
        talker_graph=types.SimpleNamespace(),
        input_metadata=_verified_prefill_metadata(),
        prefill_backend="compile_backend_eager",
        prefill_mask_mode="auto",
    )

    with pytest.raises(ReachedPrefill):
        next(generator)


def test_fast_generate_streaming_forwards_prefill_compile_compat_mode(monkeypatch):
    class ReachedPrefill(RuntimeError):
        pass

    def capture_prefill(*args, **kwargs):
        assert kwargs["prefill_backend"] == "compile_reduce_overhead"
        assert kwargs["prefill_mask_mode"] == "skip"
        assert kwargs["prefill_compile_compat_mode"] == "strict_bf16_sdpa_v1"
        assert kwargs["input_metadata"] == _verified_prefill_metadata()
        raise ReachedPrefill

    monkeypatch.setattr(streaming, "_run_talker_prefill", capture_prefill)

    talker = _dummy_streaming_model()
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    generator = streaming.fast_generate_streaming(
        talker=talker,
        talker_input_embeds=tie.to(torch.bfloat16),
        attention_mask=tam,
        trailing_text_hiddens=tth.to(torch.bfloat16),
        tts_pad_embed=tpe.to(torch.bfloat16),
        config=talker.config,
        predictor_graph=types.SimpleNamespace(),
        talker_graph=types.SimpleNamespace(),
        input_metadata=_verified_prefill_metadata(),
        prefill_backend="compile_reduce_overhead",
        prefill_mask_mode="auto",
        prefill_compile_compat_mode="strict_bf16_sdpa_v1",
    )

    with pytest.raises(ReachedPrefill):
        next(generator)


@pytest.mark.parametrize(
    "metadata_update, match",
    [
        ({"prefill_attn_implementation": "eager"}, "SDPA attention"),
        ({"prefill_has_sliding_window": True}, "sliding-window"),
    ],
)
def test_run_talker_prefill_rejects_strict_compat_unsafe_metadata(
    metadata_update,
    match,
):
    metadata = _verified_prefill_metadata()
    metadata.update(metadata_update)
    tie, tam, tth, tpe = _dummy_prefill_inputs()

    with pytest.raises(ValueError, match=match):
        _run_talker_prefill(
            _dummy_streaming_model(),
            tie.to(torch.bfloat16),
            tam,
            tth.to(torch.bfloat16),
            tpe.to(torch.bfloat16),
            prefill_backend="compile_reduce_overhead",
            prefill_mask_mode="skip",
            prefill_compile_compat_mode="strict_bf16_sdpa_v1",
            input_metadata=metadata,
        )


@pytest.mark.parametrize(
    "prefill_backend, prefill_mask_mode, dtype, batch, match",
    [
        ("compile_backend_eager", "skip", torch.bfloat16, 1, "prefill_backend"),
        ("compile_reduce_overhead", "explicit", torch.bfloat16, 1, "mask skip"),
        ("compile_reduce_overhead", "skip", torch.float32, 1, "bfloat16"),
        ("compile_reduce_overhead", "skip", torch.bfloat16, 2, "batch size 1"),
    ],
)
def test_run_talker_prefill_rejects_strict_compat_unsafe_shape_or_backend(
    prefill_backend,
    prefill_mask_mode,
    dtype,
    batch,
    match,
):
    tie = torch.zeros(batch, 2, 4, dtype=dtype)
    tam = torch.ones(batch, 2, dtype=torch.long)
    tth = torch.zeros(batch, 1, 4, dtype=dtype)
    tpe = torch.zeros(batch, 1, 4, dtype=dtype)

    with pytest.raises((UnsupportedPrefillConfiguration, ValueError), match=match):
        _run_talker_prefill(
            _dummy_streaming_model(),
            tie,
            tam,
            tth,
            tpe,
            prefill_backend=prefill_backend,
            prefill_mask_mode=prefill_mask_mode,
            prefill_compile_compat_mode="strict_bf16_sdpa_v1",
            input_metadata=_verified_prefill_metadata(),
        )


def test_prefill_compile_cache_key_includes_compat_mode():
    tie, tam, tth, tpe = _dummy_prefill_inputs()
    talker = object()
    base = streaming._prefill_compile_cache_key(
        talker,
        tie,
        None,
        tth,
        tpe,
        "compile_reduce_overhead",
        "skip",
        "none",
    )
    strict = streaming._prefill_compile_cache_key(
        talker,
        tie,
        None,
        tth,
        tpe,
        "compile_reduce_overhead",
        "skip",
        "strict_bf16_sdpa_v1",
    )
    assert base != strict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast_generate syncs.")
def test_min_new_tokens_suppresses_early_eos():
    class DummyConfig:
        codec_eos_token_id = 1
        num_code_groups = 16
        vocab_size = 5

    class DummyCodePredictor:
        def __init__(self, vocab, hidden, num_codebooks, device):
            self._embeds = torch.nn.ModuleList(
                [torch.nn.Embedding(vocab, hidden).to(device) for _ in range(num_codebooks)]
            )

        def get_input_embeddings(self):
            return list(self._embeds)

    class FixedCodecHead(torch.nn.Module):
        def __init__(self, vocab, eos_id):
            super().__init__()
            self.vocab = vocab
            self.eos_id = eos_id

        def forward(self, x):
            logits = torch.full((x.shape[0], self.vocab), -10.0, device=x.device)
            logits[:, self.eos_id] = 10.0
            logits[:, 0] = 5.0
            return logits

    class DummyTalker:
        def __init__(self, hidden=4, device="cuda"):
            self.config = DummyConfig()
            self.code_predictor = DummyCodePredictor(
                self.config.vocab_size, hidden, self.config.num_code_groups - 1, device
            )
            self._embed = torch.nn.Embedding(self.config.vocab_size, hidden).to(device)
            self.codec_head = FixedCodecHead(self.config.vocab_size, self.config.codec_eos_token_id).to(device)
            self.rope_deltas = torch.zeros(1, 1, device=device)

        def get_input_embeddings(self):
            return self._embed

        def forward(self, inputs_embeds, attention_mask=None, **kwargs):
            device = inputs_embeds.device
            logits = torch.full((1, 1, self.config.vocab_size), -10.0, device=device)
            logits[..., self.config.codec_eos_token_id] = 10.0
            logits[..., 0] = 5.0
            past_hidden = torch.zeros(1, 1, inputs_embeds.shape[-1], device=device)
            past_kv = [(torch.zeros(1, 1, 1, 1, device=device), torch.zeros(1, 1, 1, 1, device=device))]
            return types.SimpleNamespace(
                past_key_values=past_kv,
                past_hidden=past_hidden,
                generation_step=0,
                logits=logits,
            )

    class DummyPredictorGraph:
        def run(self, pred_input):
            return torch.zeros(15, dtype=torch.long, device=pred_input.device)

    class DummyTalkerGraph:
        max_seq_len = 8

        def prefill_kv(self, past_key_values):
            return 1

        def set_generation_state(self, attention_mask, rope_deltas):
            return None

        def run(self, input_embeds, position):
            return input_embeds

    talker = DummyTalker()
    tie = torch.zeros(1, 3, 4, device="cuda")
    tam = torch.ones(1, 3, dtype=torch.long, device="cuda")
    tth = torch.zeros(1, 1, 4, device="cuda")
    tpe = torch.zeros(1, 1, 4, device="cuda")

    codec_ids, _ = fast_generate(
        talker=talker,
        talker_input_embeds=tie,
        attention_mask=tam,
        trailing_text_hiddens=tth,
        tts_pad_embed=tpe,
        config=talker.config,
        predictor_graph=DummyPredictorGraph(),
        talker_graph=DummyTalkerGraph(),
        max_new_tokens=3,
        min_new_tokens=2,
        do_sample=False,
    )

    assert codec_ids is not None
    assert codec_ids.shape[0] >= 2
    eos_id = talker.config.codec_eos_token_id
    assert (codec_ids[:2, 0] == eos_id).sum().item() == 0
