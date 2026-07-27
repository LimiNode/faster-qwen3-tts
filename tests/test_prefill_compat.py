import pytest
import torch
import torch.nn.functional as F

from faster_qwen3_tts import prefill_compat


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
