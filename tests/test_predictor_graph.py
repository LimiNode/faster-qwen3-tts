import torch

from faster_qwen3_tts.predictor_graph import PredictorGraph


class _FakeGraph:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _make_graph(*, static_output: bool) -> PredictorGraph:
    graph = PredictorGraph.__new__(PredictorGraph)
    graph.input_buf = torch.zeros(1, 2, 4)
    graph.output_tokens = torch.arange(15, dtype=torch.long)
    graph.graph = _FakeGraph(graph.output_tokens)
    graph.returns_static_output = static_output
    return graph


def test_predictor_graph_default_returns_snapshot() -> None:
    graph = _make_graph(static_output=False)

    output = graph.run(torch.ones(1, 2, 4))
    graph.output_tokens[0] = 99

    assert output[0].item() == 0


def test_predictor_graph_static_mode_returns_reusable_buffer() -> None:
    graph = _make_graph(static_output=True)

    output = graph.run(torch.ones(1, 2, 4))
    graph.output_tokens[0] = 99

    assert output is graph.output_tokens
    assert output[0].item() == 99
