import sys
import types

import pytest
import torch
import torch.nn.functional as F

from faster_qwen3_tts import prefill_compat
from faster_qwen3_tts.model import FasterQwen3TTS


class Qwen3TTSRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.variance_epsilon = 1.0e-6

    def forward(self, hidden_states):
        return hidden_states


class Qwen3TTSTalkerTextMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4)
        self.up_proj = torch.nn.Linear(4, 4)
        self.down_proj = torch.nn.Linear(4, 4)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states):
        return hidden_states


class Qwen3TTSTalkerAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.head_dim = 4
        self.scaling = 0.5
        self.num_key_value_groups = 1
        self.layer_idx = 0
        self.is_causal = True

    def forward(self, hidden_states):
        return hidden_states


class CompleteTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Qwen3TTSRMSNorm()
        self.mlp = Qwen3TTSTalkerTextMLP()
        self.attn = Qwen3TTSTalkerAttention()


class PartialTalker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Qwen3TTSRMSNorm()
        self.mlp = Qwen3TTSTalkerTextMLP()


def test_strict_add_matches_torch_add():
    left = torch.randn(2, 3, dtype=torch.float32)
    right = torch.randn(2, 3, dtype=torch.float32)
    torch.testing.assert_close(
        prefill_compat._strict_add(left, right),
        left + right,
    )


def test_strict_mul_matches_torch_mul_for_non_contiguous_inputs():
    left = torch.randn(3, 2, 4, dtype=torch.float32).transpose(0, 1)
    right = torch.randn(3, 2, 4, dtype=torch.float32).transpose(0, 1)
    assert not left.is_contiguous()
    torch.testing.assert_close(
        prefill_compat._strict_mul(left, right),
        left * right,
    )


def test_strict_rmsnorm_matches_reference():
    tensor = torch.randn(2, 5, 4, dtype=torch.float32)
    weight = torch.randn(4, dtype=torch.float32)
    expected = weight * (
        tensor
        * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + 1.0e-6)
    )
    torch.testing.assert_close(
        prefill_compat._strict_rmsnorm(tensor, weight, 1.0e-6),
        expected,
    )


@pytest.mark.parametrize("enable_gqa", [False, True])
@pytest.mark.parametrize("is_causal", [False, True])
def test_strict_sdpa_matches_reference(enable_gqa, is_causal):
    query = torch.randn(1, 4, 5, 8, dtype=torch.float32)
    key_heads = 2 if enable_gqa else 4
    key = torch.randn(1, key_heads, 5, 8, dtype=torch.float32)
    value = torch.randn(1, key_heads, 5, 8, dtype=torch.float32)
    groups = 2 if enable_gqa else 1
    expected_key = key if enable_gqa else prefill_compat.qwen_modeling.repeat_kv(key, groups)
    expected_value = (
        value
        if enable_gqa
        else prefill_compat.qwen_modeling.repeat_kv(value, groups)
    )
    expected = F.scaled_dot_product_attention(
        query,
        expected_key,
        expected_value,
        attn_mask=None,
        dropout_p=0.0,
        scale=0.125,
        is_causal=is_causal,
        **({"enable_gqa": True} if enable_gqa else {}),
    ).transpose(1, 2).contiguous()
    torch.testing.assert_close(
        prefill_compat._strict_sdpa(
            query,
            key,
            value,
            0.125,
            groups,
            is_causal,
            enable_gqa,
        ),
        expected,
    )


@pytest.mark.parametrize(
    "op,args",
    [
        (
            prefill_compat._strict_add,
            (torch.randn(2, 3), torch.randn(2, 3)),
        ),
        (
            prefill_compat._strict_mul,
            (torch.randn(2, 3), torch.randn(2, 3)),
        ),
        (
            prefill_compat._strict_rmsnorm,
            (torch.randn(2, 3, 4), torch.randn(4), 1.0e-6),
        ),
        (
            prefill_compat._strict_sdpa,
            (
                torch.randn(1, 4, 5, 8),
                torch.randn(1, 2, 5, 8),
                torch.randn(1, 2, 5, 8),
                0.125,
                2,
                True,
                True,
            ),
        ),
    ],
)
def test_custom_ops_pass_opcheck(op, args):
    result = torch.library.opcheck(op, args, raise_exception=True)
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_strict_sdpa_cuda_bf16_gqa_layout():
    query = torch.randn(1, 4, 7, 16, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 2, 7, 16, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(1, 2, 7, 16, device="cuda", dtype=torch.bfloat16)
    output = prefill_compat._strict_sdpa(query, key, value, 0.25, 2, True, True)
    assert output.shape == (1, 7, 4, 16)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()


def test_configure_prefill_compile_compat_none_declares_immutable_mode():
    talker = CompleteTalker()
    metadata = prefill_compat.configure_prefill_compile_compat(talker, "none")

    assert metadata["prefill_compile_compat_mode"] == "none"
    assert metadata["prefill_compile_compat_applied"] is False
    assert (
        talker._faster_qwen3_tts_prefill_compile_compat_declared_mode
        == "none"
    )

    with pytest.raises(RuntimeError, match="does not match"):
        prefill_compat.ensure_prefill_compile_compat(
            talker,
            "strict_bf16_sdpa_v1",
        )


def test_configure_prefill_compile_compat_strict_rejects_later_none():
    talker = CompleteTalker()
    original_attention_forward = talker.attn.forward
    metadata = prefill_compat.configure_prefill_compile_compat(
        talker,
        "strict_bf16_sdpa_v1",
    )

    assert metadata["prefill_compile_compat_applied"] is False
    assert metadata["prefill_compile_compat_patched_modules"] == {}
    assert metadata["prefill_compile_compat_validated_modules"] == {
        "attention": 1,
        "mlp": 1,
        "rmsnorm": 1,
    }
    assert talker.attn.forward == original_attention_forward

    with pytest.raises(RuntimeError, match="does not match"):
        prefill_compat.ensure_prefill_compile_compat(talker, "none")


def test_prefill_compile_compat_context_restores_original_forwards():
    talker = CompleteTalker()
    original_attention_forward = talker.attn.forward
    prefill_compat.configure_prefill_compile_compat(
        talker,
        "strict_bf16_sdpa_v1",
    )

    with prefill_compat.prefill_compile_compat_context(
        talker,
        "strict_bf16_sdpa_v1",
    ) as metadata:
        assert metadata["prefill_compile_compat_applied"] is True
        assert metadata["prefill_compile_compat_patched_modules"] == {
            "attention": 1,
            "mlp": 1,
            "rmsnorm": 1,
        }
        assert talker.attn.forward != original_attention_forward

    assert talker.attn.forward == original_attention_forward
    metadata = prefill_compat.prefill_compile_compat_metadata(talker)
    assert metadata["prefill_compile_compat_applied"] is False
    assert metadata["prefill_compile_compat_patched_modules"] == {}
    assert metadata["prefill_compile_compat_validated_modules"] == {
        "attention": 1,
        "mlp": 1,
        "rmsnorm": 1,
    }


def test_legacy_unconfigured_talker_cannot_report_none_after_strict_patch():
    talker = CompleteTalker()

    prefill_compat.ensure_prefill_compile_compat(
        talker,
        "strict_bf16_sdpa_v1",
    )

    with pytest.raises(RuntimeError, match="different"):
        prefill_compat.ensure_prefill_compile_compat(talker, "none")


def test_prefill_compile_compat_patch_is_atomic_on_incomplete_talker():
    talker = PartialTalker()
    original_norm_forward = talker.norm.forward
    original_mlp_forward = talker.mlp.forward

    with pytest.raises(RuntimeError, match="Incomplete"):
        prefill_compat.apply_prefill_compile_compat(
            talker,
            "strict_bf16_sdpa_v1",
        )

    assert talker.norm.forward == original_norm_forward
    assert talker.mlp.forward == original_mlp_forward
    assert not hasattr(talker, "_faster_qwen3_tts_prefill_compile_compat_mode")


def test_from_pretrained_propagates_strict_prefill_compat_metadata(monkeypatch):
    import faster_qwen3_tts.predictor_graph as predictor_graph_module
    import faster_qwen3_tts.talker_graph as talker_graph_module

    talker = CompleteTalker()
    talker.model = torch.nn.Identity()
    talker.code_predictor = types.SimpleNamespace(
        model=types.SimpleNamespace(config=types.SimpleNamespace())
    )
    base_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            talker=talker,
            config=types.SimpleNamespace(
                talker_config=types.SimpleNamespace(hidden_size=4)
            ),
        ),
    )

    class FakeQwen3TTSModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return base_model

    class FakePredictorGraph:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeTalkerGraph:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        types.SimpleNamespace(Qwen3TTSModel=FakeQwen3TTSModel),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(predictor_graph_module, "PredictorGraph", FakePredictorGraph)
    monkeypatch.setattr(talker_graph_module, "TalkerGraph", FakeTalkerGraph)

    model = FasterQwen3TTS.from_pretrained(
        "dummy-model",
        device="cuda",
        dtype="bfloat16",
        prefill_compile_compat_mode="strict_bf16_sdpa_v1",
    )

    assert model.prefill_compile_compat_mode == "strict_bf16_sdpa_v1"
    metadata = model.prefill_compile_compat_metadata
    assert metadata["prefill_compile_compat_wrapper_mode"] == "strict_bf16_sdpa_v1"
    assert metadata["prefill_compile_compat_declared_mode"] == "strict_bf16_sdpa_v1"
    assert metadata["prefill_compile_compat_mode"] == "strict_bf16_sdpa_v1"
    assert metadata["prefill_compile_compat_applied"] is False
    assert metadata["prefill_compile_compat_patched_modules"] == {}
    assert metadata["prefill_compile_compat_validated_modules"] == {
        "attention": 1,
        "mlp": 1,
        "rmsnorm": 1,
    }
