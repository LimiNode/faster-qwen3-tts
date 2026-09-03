"""Opt-in torch profiler capture for the first autoregressive chunk."""

import os
from pathlib import Path

import torch


class FirstChunkProfiler:
    """Capture CUDA operations until the first streaming chunk is complete."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.profile = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        )

    def start(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile.start()

    def stop(self) -> dict[str, object]:
        try:
            torch.cuda.synchronize()
            self.profile.stop()
            self.profile.export_chrome_trace(str(self.trace_path))
            table_path = Path(f"{self.trace_path}.txt")
            table_path.write_text(
                self.profile.key_averages().table(
                    sort_by="self_cuda_time_total",
                    row_limit=80,
                ),
                encoding="utf-8",
            )
            return {
                "first_chunk_torch_profile_complete": True,
                "first_chunk_torch_profile_trace_path": str(self.trace_path),
                "first_chunk_torch_profile_table_path": str(table_path),
            }
        except Exception as exc:  # the diagnostic must not break synthesis
            return {
                "first_chunk_torch_profile_complete": False,
                "first_chunk_torch_profile_error": f"{type(exc).__name__}: {exc}",
            }


def create_first_chunk_profiler(device: torch.device) -> FirstChunkProfiler | None:
    """Create the profiler selected by the diagnostic environment, if any."""

    raw_path = os.environ.get("QTB_FASTER_TORCH_PROFILE_FIRST_CHUNK", "").strip()
    if not raw_path or device.type != "cuda":
        return None
    return FirstChunkProfiler(Path(raw_path))
