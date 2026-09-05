import torch

from faster_qwen3_tts.model import _talker_input_position_hashes


def test_talker_input_position_hashes_are_stable_and_position_specific() -> None:
    embeds = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    first = _talker_input_position_hashes(embeds)
    second = _talker_input_position_hashes(embeds.clone())

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3


def test_talker_input_position_hashes_change_when_one_position_changes() -> None:
    embeds = torch.zeros(1, 3, 4)
    changed = embeds.clone()
    changed[:, 2, 0] = 1.0

    first = _talker_input_position_hashes(embeds)
    second = _talker_input_position_hashes(changed)

    assert first[:2] == second[:2]
    assert first[2] != second[2]
