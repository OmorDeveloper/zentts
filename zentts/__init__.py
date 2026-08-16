"""ZenTTS - English text-to-speech for the command line and for Python."""

from .config import (
    MODEL_FILENAME,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    VOICE_PREFIXES,
    VOICES_FILENAME,
    EspeakConfig,
)
from .engine import ZenTTS, trim_silence
from .models import ensure_models, model_dir, resolve_model_files

__all__ = [
    "ZenTTS",
    "EspeakConfig",
    "SAMPLE_RATE",
    "SUPPORTED_LANGUAGES",
    "VOICE_PREFIXES",
    "MODEL_FILENAME",
    "VOICES_FILENAME",
    "ensure_models",
    "model_dir",
    "resolve_model_files",
    "trim_silence",
    "main",
]


def main():
    """Entry point for the `zentts` command."""
    from .cli import main as cli_main

    return cli_main()
