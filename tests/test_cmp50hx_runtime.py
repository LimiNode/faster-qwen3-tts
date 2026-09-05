"""Regression tests for the opt-in CMP 50HX runtime paths."""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import numpy as np
import torch

from faster_qwen3_tts import cmp50hx_diagnostic, model

_CMP_ENV_DEFAULTS = {
    "QTB_FASTER_EAGER_DIAGNOSTIC": "0",
    "QTB_FASTER_MLP_FP32_ISLAND": "0",
    "QTB_FASTER_RESIDUAL_CARRIER_FP32": "0",
    "QTB_FASTER_GRAPH_RESIDUAL_CARRIER_FP32": "0",
    "QTB_FASTER_MLP_NARROW_GATE_UP_FP16": "0",
    "QTB_FASTER_MLP_TRITON_SILU_MUL": "0",
    "QTB_FASTER_GRAPH_FINITE_CHECKER": "0",
    "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE": "0",
    "QTB_FASTER_CODEC_RIGHT_PADDED_CUDA_GRAPH": "0",
    "QTB_FASTER_ASYNC_CODEC_DECODE": "0",
    "QTB_FASTER_BASE_REFERENCE_CONTEXT_BOOTSTRAP": "0",
}


class _FakeDecoder:
    total_upsample = 2

    def __init__(self) -> None:
        self.forward_inputs: list[torch.Tensor] = []
        self.capture_windows: list[int] = []
        self.parameter = types.SimpleNamespace(device=torch.device("cuda:0"))

    def forward_optimized(self, codes: torch.Tensor) -> torch.Tensor:
        self.forward_inputs.append(codes.clone())
        return torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)

    def capture_cuda_graph(self, *, window_size: int) -> None:
        self.capture_windows.append(window_size)

    def parameters(self):
        return iter((self.parameter,))


class _FakeTokenizer:
    def __init__(self, decoder: _FakeDecoder) -> None:
        self.model = types.SimpleNamespace(decoder=decoder)

    @staticmethod
    def get_output_sample_rate() -> int:
        return 24_000


def _base_model(tokenizer: _FakeTokenizer) -> object:
    return types.SimpleNamespace(
        model=types.SimpleNamespace(speech_tokenizer=tokenizer)
    )


class Cmp50hxRuntimeTests(unittest.TestCase):
    def test_default_environment_is_noop(self) -> None:
        with mock.patch.dict(os.environ, _CMP_ENV_DEFAULTS):
            self.assertFalse(model._use_codec_right_padded_decode())
            self.assertFalse(model._use_codec_right_padded_cuda_graph())
            self.assertFalse(model._use_base_reference_context_bootstrap())
            self.assertFalse(model._use_async_codec_decode())
            cmp50hx_diagnostic._installed = False
            cmp50hx_diagnostic.install()
            self.assertFalse(cmp50hx_diagnostic._installed)

    def test_frozen_numerical_profile_enables_only_selected_boundaries(self) -> None:
        environment = {
            **_CMP_ENV_DEFAULTS,
            "QTB_FASTER_MLP_FP32_ISLAND": "1",
            "QTB_FASTER_MLP_NARROW_GATE_UP_FP16": "1",
            "QTB_FASTER_GRAPH_RESIDUAL_CARRIER_FP32": "1",
        }
        with mock.patch.dict(os.environ, environment):
            self.assertTrue(cmp50hx_diagnostic._mlp_fp32_island_enabled())
            self.assertTrue(
                cmp50hx_diagnostic._mlp_narrow_gate_up_fp16_enabled()
            )
            self.assertTrue(
                cmp50hx_diagnostic._graph_residual_carrier_fp32_enabled()
            )
            self.assertFalse(cmp50hx_diagnostic._enabled())
            self.assertFalse(cmp50hx_diagnostic._residual_carrier_fp32_enabled())
            self.assertFalse(cmp50hx_diagnostic._mlp_triton_silu_mul_enabled())
            self.assertFalse(cmp50hx_diagnostic._graph_finite_checker_enabled())

    def test_right_padding_preserves_shape_and_causal_prefix(self) -> None:
        decoder = _FakeDecoder()
        tokenizer = _FakeTokenizer(decoder)
        audio_codes = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long)
        environment = {
            **_CMP_ENV_DEFAULTS,
            "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE": "1",
            "QTB_FASTER_CODEC_RIGHT_PADDED_MAX_DECODE_INPUT_FRAMES": "2",
        }

        with mock.patch.dict(os.environ, environment):
            audio, sample_rate = model._decode_right_padded_window(
                tokenizer, audio_codes, 4
            )

        self.assertEqual(sample_rate, 24_000)
        np.testing.assert_array_equal(audio[0], np.array([0.0, 1.0], np.float32))
        self.assertEqual(tuple(decoder.forward_inputs[0].shape), (1, 3, 4))
        torch.testing.assert_close(
            decoder.forward_inputs[0][..., :2],
            audio_codes.transpose(0, 1).unsqueeze(0),
        )
        self.assertTrue(torch.count_nonzero(decoder.forward_inputs[0][..., 2:]) == 0)

    def test_right_padding_requires_and_enforces_input_bound(self) -> None:
        decoder = _FakeDecoder()
        tokenizer = _FakeTokenizer(decoder)
        audio_codes = torch.zeros((2, 3), dtype=torch.long)
        enabled = {
            **_CMP_ENV_DEFAULTS,
            "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE": "1",
        }

        with mock.patch.dict(
            os.environ, enabled, clear=True
        ), self.assertRaisesRegex(RuntimeError, "requires"):
            model._decode_right_padded_window(tokenizer, audio_codes, 4)

        with mock.patch.dict(
            os.environ,
            {
                **enabled,
                "QTB_FASTER_CODEC_RIGHT_PADDED_MAX_DECODE_INPUT_FRAMES": "1",
            },
        ), self.assertRaisesRegex(RuntimeError, "exceeds the verified"):
            model._decode_right_padded_window(tokenizer, audio_codes, 4)

    def test_manual_graph_requires_right_padded_decode(self) -> None:
        decoder = _FakeDecoder()
        tokenizer = _FakeTokenizer(decoder)
        with mock.patch.dict(
            os.environ,
            {
                **_CMP_ENV_DEFAULTS,
                "QTB_FASTER_CODEC_RIGHT_PADDED_CUDA_GRAPH": "1",
            },
        ), self.assertRaisesRegex(RuntimeError, "requires.*DECODE=1"):
            model._capture_right_padded_decoder_cuda_graph(_base_model(tokenizer))

    def test_manual_graph_captures_exact_configured_window(self) -> None:
        decoder = _FakeDecoder()
        tokenizer = _FakeTokenizer(decoder)
        with mock.patch.dict(
            os.environ,
            {
                **_CMP_ENV_DEFAULTS,
                "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE": "1",
                "QTB_FASTER_CODEC_RIGHT_PADDED_CUDA_GRAPH": "1",
                "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE_WINDOW_FRAMES": "48",
            },
        ):
            model._capture_right_padded_decoder_cuda_graph(_base_model(tokenizer))

        self.assertEqual(decoder.capture_windows, [48])

    def test_async_manual_graph_captures_and_reuses_dedicated_stream(self) -> None:
        decoder = _FakeDecoder()
        tokenizer = _FakeTokenizer(decoder)
        codec_stream = mock.Mock()
        current_stream = mock.Mock()
        stream_context = mock.MagicMock()
        environment = {
            **_CMP_ENV_DEFAULTS,
            "QTB_FASTER_CODEC_RIGHT_PADDED_DECODE": "1",
            "QTB_FASTER_CODEC_RIGHT_PADDED_CUDA_GRAPH": "1",
            "QTB_FASTER_ASYNC_CODEC_DECODE": "1",
        }

        with mock.patch.dict(os.environ, environment), mock.patch.object(
            torch.cuda, "Stream", return_value=codec_stream
        ), mock.patch.object(
            torch.cuda, "stream", return_value=stream_context
        ), mock.patch.object(
            torch.cuda, "current_stream", return_value=current_stream
        ):
            model._capture_right_padded_decoder_cuda_graph(_base_model(tokenizer))

        self.assertIs(decoder._qtb_async_codec_stream, codec_stream)
        self.assertIs(model._decoder_async_codec_stream(tokenizer), codec_stream)
        current_stream.wait_stream.assert_called_once_with(codec_stream)
        self.assertEqual(decoder.capture_windows, [80])

    def test_base_reference_context_window_uses_exact_history_tail(self) -> None:
        reference = torch.arange(90, dtype=torch.long).reshape(30, 3)
        first_generated = torch.arange(24, dtype=torch.long).reshape(8, 3) + 100
        second_generated = torch.arange(48, dtype=torch.long).reshape(16, 3) + 100

        first_window = model._base_reference_context_window(
            reference,
            first_generated,
            n_new=8,
            context_frames=25,
        )
        second_window = model._base_reference_context_window(
            reference,
            second_generated,
            n_new=8,
            context_frames=25,
        )

        torch.testing.assert_close(
            first_window,
            torch.cat((reference[-25:], first_generated), dim=0),
        )
        torch.testing.assert_close(
            second_window,
            torch.cat((reference[-17:], second_generated), dim=0),
        )


if __name__ == "__main__":
    unittest.main()
