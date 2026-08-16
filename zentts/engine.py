"""The ZenTTS speech engine: ONNX inference over phoneme tokens."""

import asyncio
import importlib.util
import os
import re
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray

from .config import (
    MAX_PHONEME_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    VOICE_PREFIXES,
    EspeakConfig,
)
from .log import log
from .tokenizer import Tokenizer


def trim_silence(
    audio: NDArray[np.float32],
    top_db: float = 60.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> NDArray[np.float32]:
    """Drop leading/trailing silence so concatenated chunks flow naturally.

    Frames the signal, compares each frame's RMS against the loudest frame and
    keeps everything within `top_db` of it.
    """
    if audio.size == 0:
        return audio

    padded = np.pad(audio.astype(np.float32), frame_length // 2)
    frame_count = 1 + (len(padded) - frame_length) // hop_length
    if frame_count < 1:
        return audio

    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)
    frames = frames[:: hop_length][:frame_count]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    loudest = rms.max()
    if loudest <= 0:
        return audio

    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / loudest)
    voiced = np.flatnonzero(db > -top_db)
    if voiced.size == 0:
        return audio

    start = min(len(audio), int(voiced[0]) * hop_length)
    end = min(len(audio), int(voiced[-1] + 1) * hop_length)
    return audio[start:end]


class ZenTTS:
    """Loads a ZenTTS ONNX model plus its voice pack and synthesises speech."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
    ):
        for label, path in (("Model", model_path), ("Voices", voices_path)):
            if not Path(path).exists():
                raise FileNotFoundError(f"{label} file not found at {path}")

        # See https://github.com/microsoft/onnxruntime/issues/22101 for provider notes.
        providers = ["CPUExecutionProvider"]
        if importlib.util.find_spec("onnxruntime-gpu"):
            providers = rt.get_available_providers()
        if os.getenv("ONNX_PROVIDER"):
            providers = [os.environ["ONNX_PROVIDER"]]

        log.debug(f"Providers: {providers}")
        self.model_path = model_path
        self.voices_path = voices_path
        self.sess = rt.InferenceSession(model_path, providers=providers)
        self.voices: np.ndarray = np.load(voices_path)
        self.tokenizer = Tokenizer(espeak_config)

    @classmethod
    def from_session(
        cls,
        session: rt.InferenceSession,
        voices_path: str,
        espeak_config: EspeakConfig | None = None,
    ) -> "ZenTTS":
        """Build an engine around an already-created ONNX session."""
        instance = cls.__new__(cls)
        instance.sess = session
        instance.model_path = getattr(session, "_model_path", "<session>")
        instance.voices_path = voices_path
        instance.voices = np.load(voices_path)
        instance.tokenizer = Tokenizer(espeak_config)
        return instance

    def get_voices(self) -> list[str]:
        """The English voices in the loaded voice pack."""
        return sorted(v for v in self.voices.keys() if v.startswith(VOICE_PREFIXES))

    def get_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def get_voice_style(self, name: str) -> NDArray[np.float32]:
        if name not in self.get_voices():
            raise ValueError(f"Voice {name} is not an available English voice")
        return self.voices[name]

    def _create_audio(
        self, phonemes: str, voice: NDArray[np.float32], speed: float
    ) -> tuple[NDArray[np.float32], int]:
        log.debug(f"Phonemes: {phonemes}")
        if len(phonemes) > MAX_PHONEME_LENGTH:
            log.warning(f"Phonemes too long, truncating to {MAX_PHONEME_LENGTH}")
        phonemes = phonemes[:MAX_PHONEME_LENGTH]

        start_t = time.time()
        tokens = self.tokenizer.tokenize(phonemes)
        assert len(tokens) <= MAX_PHONEME_LENGTH, (
            f"Context length is {MAX_PHONEME_LENGTH}, leaving room for the pad "
            "token 0 at the start and end"
        )

        style = voice[len(tokens)]
        tokens = [[0, *tokens, 0]]

        audio = self.sess.run(
            None,
            dict(
                tokens=tokens, style=style, speed=np.ones(1, dtype=np.float32) * speed
            ),
        )[0]

        audio_duration = len(audio) / SAMPLE_RATE
        create_duration = time.time() - start_t
        rtf = create_duration / audio_duration if audio_duration else 0.0
        log.debug(
            f"Created {audio_duration:.2f}s of audio from {len(phonemes)} phonemes "
            f"in {create_duration:.2f}s (RTF: {rtf:.2f})"
        )
        return audio, SAMPLE_RATE

    def _split_phonemes(self, phonemes: str) -> list[str]:
        """Split phonemes into model-sized batches, preferring punctuation breaks."""
        parts = re.split(r"([.,!?;])", phonemes)
        batches: list[str] = []
        current = ""

        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(current) + len(part) + 1 > MAX_PHONEME_LENGTH:
                batches.append(current.strip())
                current = part
            elif part in ".,!?;":
                current += part
            else:
                if current:
                    current += " "
                current += part

        if current:
            batches.append(current.strip())

        return batches

    def _resolve_voice(
        self, voice: str | NDArray[np.float32]
    ) -> NDArray[np.float32]:
        if isinstance(voice, str):
            return self.get_voice_style(voice)
        return voice

    @staticmethod
    def _validate(lang: str, speed: float) -> None:
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language must be one of {', '.join(SUPPORTED_LANGUAGES)}. Got {lang}"
            )
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed should be between 0.5 and 2.0")

    def create(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        phonemes: str | None = None,
        trim: bool = True,
    ) -> tuple[NDArray[np.float32], int]:
        """Synthesise `text` with the given voice and speed."""
        self._validate(lang, speed)
        style = self._resolve_voice(voice)

        start_t = time.time()
        if not phonemes:
            phonemes = self.tokenizer.phonemize(text, lang)
        batches = self._split_phonemes(phonemes)

        log.debug(f"Creating audio for {len(batches)} batches, {len(phonemes)} phonemes")
        audio = []
        for batch in batches:
            part, _ = self._create_audio(batch, style, speed)
            if trim:
                part = trim_silence(part)
            audio.append(part)

        merged = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
        log.debug(f"Created audio in {time.time() - start_t:.2f}s")
        return merged, SAMPLE_RATE

    async def create_stream(
        self,
        text: str,
        voice: str | NDArray[np.float32],
        speed: float = 1.0,
        lang: str = "en-us",
        phonemes: str | None = None,
        trim: bool = True,
    ) -> AsyncGenerator[tuple[NDArray[np.float32], int], None]:
        """Yield audio chunks as they finish, so playback can start early."""
        self._validate(lang, speed)
        style = self._resolve_voice(voice)

        if not phonemes:
            phonemes = self.tokenizer.phonemize(text, lang)
        batches = self._split_phonemes(phonemes)
        queue: asyncio.Queue[tuple[NDArray[np.float32], int] | None] = asyncio.Queue()

        async def process_batches():
            for i, batch in enumerate(batches):
                loop = asyncio.get_event_loop()
                # Inference blocks, so keep it off the event loop.
                part, sample_rate = await loop.run_in_executor(
                    None, self._create_audio, batch, style, speed
                )
                if trim:
                    part = trim_silence(part)
                log.debug(f"Processed chunk {i} of stream")
                await queue.put((part, sample_rate))
            await queue.put(None)

        task = asyncio.create_task(process_batches())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not task.done():
                task.cancel()
