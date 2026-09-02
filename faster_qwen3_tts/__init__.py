"""faster-qwen3-tts: Real-time Qwen3-TTS inference using CUDA graphs."""

from .cmp50hx_diagnostic import install as _install_cmp50hx_diagnostic

_install_cmp50hx_diagnostic()

from .ggml_backend import GGMLQwen3TTS
from .model import FasterQwen3TTS

__version__ = "0.3.2"
__all__ = ["FasterQwen3TTS", "GGMLQwen3TTS"]
