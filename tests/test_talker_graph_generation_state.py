import types
import unittest

import torch

from faster_qwen3_tts.talker_graph import TalkerGraph


class TalkerGraphGenerationStateTests(unittest.TestCase):
    def test_none_then_verified_all_valid_reuses_mask_table(self) -> None:
        graph = _graph_with_mask_table(mask_key=None)
        mask = torch.ones(1, 32, dtype=torch.long)

        graph.set_generation_state(mask, None, attention_mask_all_valid=True)

        self.assertEqual([], graph._build_calls)
        self.assertIsNone(graph._mask_key)
        self.assertTrue(
            graph.last_generation_state_profile["generation_state_mask_cache_hit"]
        )
        self.assertTrue(
            graph.last_generation_state_profile[
                "generation_state_attention_mask_all_valid"
            ]
        )

    def test_repeated_all_valid_reuses_mask_table(self) -> None:
        graph = _graph_with_mask_table(mask_key=None)
        mask = torch.ones(1, 32, dtype=torch.long)

        graph.set_generation_state(mask, None)
        graph.set_generation_state(mask, None)

        self.assertEqual([], graph._build_calls)
        self.assertIsNone(graph._mask_key)

    def test_padded_mask_builds_once_and_reuses_same_pattern(self) -> None:
        graph = _graph_with_mask_table(mask_key=None)
        mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)

        graph.set_generation_state(mask, None)
        graph.set_generation_state(mask, None)

        self.assertEqual(1, len(graph._build_calls))
        self.assertEqual((2,), graph._mask_key)
        built_mask = graph._build_calls[0]
        self.assertEqual([0, 0, 1, 1, 1, 1], built_mask[0].tolist())

    def test_changed_padded_pattern_rebuilds(self) -> None:
        graph = _graph_with_mask_table(mask_key=None)

        graph.set_generation_state(torch.tensor([[0, 0, 1, 1]]), None)
        graph.set_generation_state(torch.tensor([[0, 1, 1, 1]]), None)

        self.assertEqual(2, len(graph._build_calls))
        self.assertEqual((1,), graph._mask_key)

    def test_rope_deltas_update_on_every_request(self) -> None:
        graph = _graph_with_mask_table(mask_key=None)

        graph.set_generation_state(None, torch.tensor([3.0]))
        self.assertEqual([[3.0]], graph.rope_deltas.tolist())
        graph.set_generation_state(None, torch.tensor([5.0]))
        self.assertEqual([[5.0]], graph.rope_deltas.tolist())


def _graph_with_mask_table(mask_key: tuple[int, ...] | None) -> TalkerGraph:
    graph = TalkerGraph.__new__(TalkerGraph)
    graph.max_seq_len = 6
    graph.attn_mask_table = ["captured"]
    graph._mask_key = mask_key
    graph.rope_deltas = torch.zeros(1, 1)
    graph.last_generation_state_profile = {}
    graph._build_calls = []

    def build_attention_masks(
        self: TalkerGraph,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        self._build_calls.append(
            None if attention_mask is None else attention_mask.clone()
        )
        self.attn_mask_table = ["rebuilt"]
        return {
            "generation_state_mask_table_build_ms": 1.25,
            "generation_state_masks_built": self.max_seq_len,
        }

    graph._build_attention_masks = types.MethodType(build_attention_masks, graph)
    return graph


if __name__ == "__main__":
    unittest.main()
