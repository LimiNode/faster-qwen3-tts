import unittest

import torch

from faster_qwen3_tts.streaming import (
    UnsupportedPrefillConfiguration,
    _validate_padded_prefill_configuration,
    pad_prefill_inputs_left,
)
from faster_qwen3_tts.talker_graph import TalkerGraph


class PaddedPrefillResearchTests(unittest.TestCase):
    def test_left_padding_preserves_real_suffix_and_metadata(self) -> None:
        embeds = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.long)

        padded, padded_mask, metadata = pad_prefill_inputs_left(
            embeds,
            mask,
            target_length=5,
        )

        self.assertEqual((1, 5, 4), tuple(padded.shape))
        self.assertEqual([[0, 0, 1, 1, 1]], padded_mask.tolist())
        self.assertTrue(metadata["prefill_padding_enabled"])
        self.assertEqual(2, metadata["prefill_padding_left_tokens"])
        self.assertEqual(3, metadata["prefill_real_length"])
        self.assertTrue(torch.equal(embeds, padded[:, 2:, :]))

    def test_padding_rejects_batching_and_short_targets(self) -> None:
        embeds = torch.ones(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            pad_prefill_inputs_left(embeds, mask, target_length=5)

        with self.assertRaisesRegex(ValueError, "shorter than the real prefill"):
            pad_prefill_inputs_left(
                torch.ones(1, 3, 4),
                torch.ones(1, 3, dtype=torch.long),
                target_length=2,
            )

    def test_padded_route_is_eager_and_explicit_only(self) -> None:
        metadata = {
            "prefill_padding_enabled": True,
            "prefill_real_length": 20,
            "prefill_padding_left_tokens": 12,
        }
        _validate_padded_prefill_configuration(
            prefill_backend="eager",
            prefill_mask_mode="explicit",
            input_metadata=metadata,
        )
        with self.assertRaisesRegex(UnsupportedPrefillConfiguration, "eager"):
            _validate_padded_prefill_configuration(
                prefill_backend="compile_reduce_overhead",
                prefill_mask_mode="explicit",
                input_metadata=metadata,
            )
        with self.assertRaisesRegex(UnsupportedPrefillConfiguration, "explicit"):
            _validate_padded_prefill_configuration(
                prefill_backend="eager",
                prefill_mask_mode="skip",
                input_metadata=metadata,
            )

    def test_prefill_kv_copies_only_the_real_suffix(self) -> None:
        graph = TalkerGraph.__new__(TalkerGraph)
        graph.num_layers = 1
        graph.max_seq_len = 8
        graph.device = torch.device("cpu")
        graph.static_cache = _CapturingStaticCache()

        keys = torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4)
        values = keys + 100
        copied_length = graph.prefill_kv([(keys, values)], source_start=2)

        self.assertEqual(3, copied_length)
        update = graph.static_cache.updates[0]
        self.assertEqual([0, 1, 2], update[2]["cache_position"].tolist())
        self.assertTrue(torch.equal(keys[:, :, 2:, :], update[0]))
        self.assertTrue(torch.equal(values[:, :, 2:, :], update[1]))


class _CapturingStaticCache:
    def __init__(self) -> None:
        self.updates: list[tuple[torch.Tensor, torch.Tensor, dict]] = []

    def reset(self) -> None:
        self.updates.clear()

    def update(self, key, value, _layer, kwargs):
        self.updates.append((key.clone(), value.clone(), kwargs))


if __name__ == "__main__":
    unittest.main()
