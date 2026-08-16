#!/usr/bin/env python3
"""ZenTTS - English text-to-speech for the command line and for Python.

Everything lives in this one module: configuration, the espeak-ng tokenizer,
the ONNX engine, the model downloader and the CLI.

Author:  Omor <omor.developer@gmail.com>
         https://www.linkedin.com/in/omardeveloper/
Project: https://github.com/OmorDeveloper/zentts

Copyright (c) 2026 Omor. All rights reserved.
Proprietary software - see LICENSE. Third-party notices in NOTICE.md.
"""

__version__ = "1.3.0"
__author__ = "Omor"
__license__ = "Proprietary"
__url__ = "https://github.com/OmorDeveloper/zentts"

# Standard library imports
import asyncio
import collections
import ctypes
import ctypes.util
import difflib
import hashlib
import importlib.metadata
import importlib.util
import io
import itertools
import json
import logging
import os
import platform
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Third-party imports
import espeakng_loader
import numpy as np
import onnxruntime as rt
import phonemizer
import pymupdf
import pymupdf4llm
import soundfile as sf
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from numpy.typing import NDArray
from phonemizer.backend.espeak.wrapper import EspeakWrapper

warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")


##############################################################################
# Configuration
##############################################################################

SUPPORTED_LANGUAGES = ["en-us", "en-gb"]

# Voice-name prefixes for the English voice packs. A ZenTTS voice id reads
# zen_<region>_<gender><number>, e.g. zen_us_f10 or zen_uk_m03.
VOICE_PREFIXES = ("zen_us_f", "zen_us_m", "zen_uk_f", "zen_uk_m")

# Human-readable label for each voice-id prefix.
VOICE_GROUPS = {
    "zen_us_f": "US English, female",
    "zen_us_m": "US English, male",
    "zen_uk_f": "UK English, female",
    "zen_uk_m": "UK English, male",
}

# Language each voice group was trained for, used to suggest --lang.
VOICE_GROUP_LANGUAGE = {
    "zen_us_f": "en-us",
    "zen_us_m": "en-us",
    "zen_uk_f": "en-gb",
    "zen_uk_m": "en-gb",
}

# The ONNX graph is compiled for a fixed context length.
MAX_PHONEME_LENGTH = 510

# Output sample rate of the model, in Hz.
SAMPLE_RATE = 24000

# Model release the code is built against.
MODEL_VERSION = "v1.0"
MODEL_FILENAME = f"zentts-{MODEL_VERSION}.onnx"
VOICES_FILENAME = f"zentts-voices-{MODEL_VERSION}.bin"

# Where the model files are published. This tag tracks the model, not the
# package version: every ZenTTS release that uses model v1.0 downloads from
# here, so the files stay put and older installs keep working.
RELEASE_TAG = "v1.0.0"
RELEASE_BASE_URL = (
    f"https://github.com/OmorDeveloper/zentts/releases/download/{RELEASE_TAG}"
)
MODEL_URL = f"{RELEASE_BASE_URL}/{MODEL_FILENAME}"
VOICES_URL = f"{RELEASE_BASE_URL}/{VOICES_FILENAME}"


@dataclass
class EspeakConfig:
    """Override the bundled espeak-ng library/data locations."""

    lib_path: str | None = None
    data_path: str | None = None


def _build_vocab() -> dict[str, int]:
    """Symbol table shared by the model and the tokenizer."""
    pad = "$"
    punctuation = ';:,.!?¡¿—…"«»“” '
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    letters_ipa = (
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ"
        "ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    )
    symbols = [pad] + list(punctuation) + list(letters) + list(letters_ipa)
    return {symbol: index for index, symbol in enumerate(symbols)}


VOCAB = _build_vocab()


##############################################################################
# Logging
##############################################################################

def _create_logger() -> logging.Logger:
    logger = logging.getLogger("zentts")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s")
        )
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level, logging.WARNING))
    return logger


log = _create_logger()


##############################################################################
# Tokenizer
##############################################################################

class Tokenizer:
    """Turns English text into the phoneme token ids the model expects."""

    def __init__(self, espeak_config: EspeakConfig | None = None):
        if not espeak_config:
            espeak_config = EspeakConfig()
        if not espeak_config.data_path:
            espeak_config.data_path = espeakng_loader.get_data_path()
        if not espeak_config.lib_path:
            espeak_config.lib_path = espeakng_loader.get_library_path()

        # An explicit library path always wins over the bundled one.
        if os.getenv("PHONEMIZER_ESPEAK_LIBRARY"):
            espeak_config.lib_path = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")

        try:
            ctypes.cdll.LoadLibrary(espeak_config.lib_path)
        except Exception as e:
            log.error(f"Failed to load the bundled espeak-ng library: {e}")
            log.warning("Falling back to a system-wide espeak-ng install")

            error_info = (
                "Failed to load espeak-ng. Please install espeak-ng system wide.\n"
                "\tSee https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md\n"
                "\tYou can also point ZenTTS at a library with the "
                "PHONEMIZER_ESPEAK_LIBRARY environment variable.\n"
                f"Environment:\n\t{platform.platform()} ({platform.release()}) | {sys.version}"
            )
            espeak_config.lib_path = ctypes.util.find_library(
                "espeak-ng"
            ) or ctypes.util.find_library("espeak")
            if not espeak_config.lib_path:
                raise RuntimeError(error_info)
            try:
                ctypes.cdll.LoadLibrary(espeak_config.lib_path)
            except Exception as e:
                raise RuntimeError(f"{e}: {error_info}")

        EspeakWrapper.set_data_path(espeak_config.data_path)
        EspeakWrapper.set_library(espeak_config.lib_path)

    @staticmethod
    def split_num(num):
        """Read years and clock times the way a person would."""
        num = num.group()
        if "." in num:
            return num
        elif ":" in num:
            h, m = [int(n) for n in num.split(":")]
            if m == 0:
                return f"{h} o'clock"
            elif m < 10:
                return f"{h} oh {m}"
            return f"{h} {m}"
        year = int(num[:4])
        if year < 1100 or year % 1000 < 10:
            return num
        left, right = num[:2], int(num[2:4])
        s = "s" if num.endswith("s") else ""
        if 100 <= year % 1000 <= 999:
            if right == 0:
                return f"{left} hundred{s}"
            elif right < 10:
                return f"{left} oh {right}{s}"
        return f"{left} {right}{s}"

    @staticmethod
    def flip_money(m):
        """Turn "$4.50" into "4 dollars and 50 cents"."""
        m = m.group()
        bill = "dollar" if m[0] == "$" else "pound"
        if m[-1].isalpha():
            return f"{m[1:]} {bill}s"
        elif "." not in m:
            s = "" if m[1:] == "1" else "s"
            return f"{m[1:]} {bill}{s}"
        b, c = m[1:].split(".")
        s = "" if b == "1" else "s"
        c = int(c.ljust(2, "0"))
        coins = (
            f"cent{'' if c == 1 else 's'}"
            if m[0] == "$"
            else ("penny" if c == 1 else "pence")
        )
        return f"{b} {bill}{s} and {c} {coins}"

    @staticmethod
    def point_num(num) -> str:
        a, b = num.group().split(".")
        return " point ".join([a, " ".join(b)])

    @staticmethod
    def normalize_text(text) -> str:
        """Clean up punctuation, numbers and abbreviations before phonemizing."""
        # strip whitespace and drop empty lines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        # curly quotes to straight quotes
        text = text.replace(chr(8216), "'").replace(chr(8217), "'")
        text = text.replace("«", chr(8220)).replace("»", chr(8221))
        text = text.replace(chr(8220), '"').replace(chr(8221), '"')
        # parentheses read as quoted asides
        text = text.replace("(", "«").replace(")", "»")
        for a, b in zip("、。！，：；？", ",.!,:;?"):
            text = text.replace(a, b + " ")
        text = re.sub(r"[^\S \n]", " ", text)
        text = re.sub(r"  +", " ", text)
        text = re.sub(r"(?<=\n) +(?=\n)", "", text)
        text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
        text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
        text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
        text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
        text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
        text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
        text = re.sub(
            r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)",
            Tokenizer.split_num,
            text,
        )
        text = re.sub(r"(?<=\d),(?=\d)", "", text)
        text = re.sub(
            r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
            Tokenizer.flip_money,
            text,
        )
        text = re.sub(r"\d*\.\d+", Tokenizer.point_num, text)
        text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
        text = re.sub(r"(?<=\d)S", " S", text)
        text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
        text = re.sub(r"(?<=X')S\b", "s", text)
        text = re.sub(
            r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text
        )
        text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
        return text.strip()

    def tokenize(self, phonemes: str) -> list[int]:
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"text is too long, must be less than {MAX_PHONEME_LENGTH} phonemes"
            )
        return [i for i in map(VOCAB.get, phonemes) if i is not None]

    def phonemize(self, text: str, lang: str = "en-us", norm: bool = True) -> str:
        """Phonemize English text. `lang` is either 'en-us' or 'en-gb'."""
        if norm:
            text = Tokenizer.normalize_text(text)

        phonemes = phonemizer.phonemize(
            text, lang, preserve_punctuation=True, with_stress=True
        )

        # espeak emits a few symbols the model was not trained on.
        phonemes = (
            phonemes.replace("ʲ", "j")
            .replace("r", "ɹ")
            .replace("x", "k")
            .replace("ɬ", "l")
        )
        phonemes = re.sub(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)", " ", phonemes)
        phonemes = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', "z", phonemes)
        if lang == "en-us":
            phonemes = re.sub(r"(?<=nˈaɪn)ti(?!ː)", "di", phonemes)
        phonemes = "".join(filter(lambda p: p in VOCAB, phonemes))
        return phonemes.strip()


##############################################################################
# Engine
##############################################################################

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


##############################################################################
# Model files
##############################################################################

CHECKSUMS = {
    MODEL_FILENAME: "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    VOICES_FILENAME: "37fd54100503ac57f5ad27fdc128bba191d1da6203e7a44949fb27198ad0388e",
}

DOWNLOAD_URLS = {
    MODEL_FILENAME: MODEL_URL,
    VOICES_FILENAME: VOICES_URL,
}


def model_dir() -> Path:
    """Directory the downloaded model files live in.

    Override it with ZENTTS_HOME; otherwise use the per-user cache directory.
    """
    env_home = os.getenv("ZENTTS_HOME")
    if env_home:
        return Path(env_home).expanduser()
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "zentts"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "zentts"
    base = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "zentts"


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, quiet: bool = False) -> Path:
    """Download `url` to `destination`, showing progress and verifying the hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if not quiet:
        print(f"Downloading {destination.name}")
        print(f"  from {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "zentts"})
    started = time.time()
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(partial, "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    downloaded += len(block)
                    if quiet:
                        continue
                    if total:
                        done = int(30 * downloaded / total)
                        bar = "■" * done + "□" * (30 - done)
                        percent = 100 * downloaded / total
                        sys.stdout.write(
                            f"\r  [{bar}] {percent:5.1f}% "
                            f"({_human(downloaded)} / {_human(total)})"
                        )
                    else:
                        sys.stdout.write(f"\r  {_human(downloaded)} downloaded")
                    sys.stdout.flush()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    if not quiet:
        sys.stdout.write(f"\n  finished in {time.time() - started:.1f}s\n")

    expected = CHECKSUMS.get(destination.name)
    if expected:
        if not quiet:
            print("  verifying checksum...")
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}\n"
                f"  expected {expected}\n  got      {actual}"
            )

    partial.replace(destination)
    return destination


def _find_existing(filename: str) -> Path | None:
    """Look for a model file in the working directory, then the cache."""
    for candidate in (Path.cwd() / filename, model_dir() / filename):
        if candidate.exists():
            return candidate
    return None


def resolve_file(
    filename: str, explicit_path: str | None = None, allow_download: bool = True
) -> Path:
    """Return a usable path for a model file, downloading it if needed."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    existing = _find_existing(filename)
    if existing:
        return existing

    url = DOWNLOAD_URLS.get(filename)
    if not allow_download or not url:
        raise FileNotFoundError(
            f"{filename} not found in {Path.cwd()} or {model_dir()}.\n"
            f"Download it from {url or 'the project release page'} "
            "or pass --model / --voices."
        )

    return download_file(url, model_dir() / filename)


def resolve_model_files(
    model_path: str | None = None,
    voices_path: str | None = None,
    allow_download: bool = True,
) -> tuple[str, str]:
    """Resolve both model files, returning their paths as strings."""
    if os.getenv("ZENTTS_NO_DOWNLOAD"):
        allow_download = False

    model = resolve_file(MODEL_FILENAME, model_path, allow_download)
    voices = resolve_file(VOICES_FILENAME, voices_path, allow_download)
    return str(model), str(voices)


def ensure_models(quiet: bool = False) -> tuple[Path, Path]:
    """Fetch both model files, replacing any local copy that is out of date.

    An existing file is checked against its published sha256, so an older or
    partly written copy is downloaded again instead of being trusted.
    """
    paths = []
    for filename in (MODEL_FILENAME, VOICES_FILENAME):
        existing = _find_existing(filename)
        expected = CHECKSUMS.get(filename)

        if existing and expected:
            if not quiet:
                print(f"Checking {existing}...")
            if _sha256(existing) == expected:
                if not quiet:
                    print(f"  {filename} is up to date")
                paths.append(existing)
                continue
            print(f"  {filename} does not match the published file, replacing it")
        elif existing:
            paths.append(existing)
            continue

        target = existing if existing else model_dir() / filename
        paths.append(download_file(DOWNLOAD_URLS[filename], target, quiet))

    return paths[0], paths[1]


##############################################################################
# Command-line interface
##############################################################################


# Global flags to stop the spinner and audio
stop_spinner = False
stop_audio = False

DEFAULT_VOICE = "zen_us_f10"


def _filter_english_voices(voices):
    """Keep only the US/UK English voices from a voice list."""
    return sorted(v for v in voices if v.startswith(VOICE_PREFIXES))


def _filter_english_languages(languages):
    """Keep only English language codes from a language list."""
    return sorted(l for l in languages if l in SUPPORTED_LANGUAGES)


def load_engine(model_path=None, voices_path=None):
    """Resolve the model files (downloading on first run) and load the engine."""
    require_license()

    try:
        model, voices = resolve_model_files(model_path, voices_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        engine = ZenTTS(model, voices)
    except Exception as e:
        print(f"Error loading the ZenTTS model: {e}")
        sys.exit(1)

    if not engine.get_voices():
        print(
            f"Error: {voices} contains no ZenTTS voices, so it is probably an "
            "older voice pack.\nRun `zentts --download` to replace it."
        )
        sys.exit(1)

    return engine


def spinning_wheel(message="Processing...", progress=None):
    """Display a spinning wheel with a message."""
    spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while not stop_spinner:
        spin = next(spinner)
        if progress is not None:
            sys.stdout.write(f"\r{message} {progress} {spin}")
        else:
            sys.stdout.write(f"\r{message} {spin}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * (len(message) + 50) + "\r")
    sys.stdout.flush()


def voice_label(voice):
    """Describe a voice id, e.g. zen_uk_m03 -> 'UK English, male'."""
    for prefix, label in VOICE_GROUPS.items():
        if voice.startswith(prefix):
            return label
    return "English"


def list_available_voices(engine, numbered=True):
    """Print the voices grouped by region and gender, and return them in order."""
    voices = _filter_english_voices(engine.get_voices())
    if not voices:
        print(
            "Error: the voice file contains no ZenTTS voices. It is probably an "
            "older voice pack.\nRun `zentts --download` to fetch the current one."
        )
        sys.exit(1)

    index = 0
    for prefix, label in VOICE_GROUPS.items():
        group = [v for v in voices if v.startswith(prefix)]
        if not group:
            continue
        print(f"\n{label} (use --lang {VOICE_GROUP_LANGUAGE[prefix]}):")
        for voice in group:
            index += 1
            print(f"  {index:>2}. {voice}" if numbered else f"  {voice}")
    print()
    return voices


def extract_text_from_epub(epub_file):
    book = epub.read_epub(epub_file)
    full_text = ""
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            full_text += soup.get_text()
    return full_text


def chunk_text(text, initial_chunk_size=1000):
    """Split text into chunks at sentence boundaries with dynamic sizing."""
    sentences = text.replace("\n", " ").split(".")
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_size = initial_chunk_size

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence = sentence.strip() + "."
        sentence_size = len(sentence)

        if sentence_size > chunk_size:
            # Emit what is already queued first, or the long sentence would be
            # spoken before the sentences that come before it.
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            words = sentence.split()
            current_piece = []
            current_piece_size = 0

            def flush(piece):
                text = " ".join(piece).strip()
                chunks.append(text if text.endswith(".") else text + ".")

            for word in words:
                word_size = len(word) + 1
                if current_piece_size + word_size > chunk_size:
                    if current_piece:
                        flush(current_piece)
                    current_piece = [word]
                    current_piece_size = word_size
                else:
                    current_piece.append(word)
                    current_piece_size += word_size

            if current_piece:
                flush(current_piece)
            continue

        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def validate_language(lang, engine):
    """Validate that the language is supported (English only)."""
    supported_languages = set(_filter_english_languages(engine.get_languages()))
    if lang not in supported_languages:
        supported_langs = ", ".join(sorted(supported_languages))
        print(f"Error: Unsupported language: {lang}")
        print(f"Supported languages are: {supported_langs}")
        sys.exit(1)
    return lang


def print_usage():
    print("""
ZenTTS - English text-to-speech from the command line

Usage: zentts <input_file> [<more_input_files>...] [<output_audio_file>] [options]

Commands:
    start              Run the OpenAI-compatible speech API server
                       (see `zentts start --help`)
    -h, --help         Show this help message
    -v, --version      Show the version number
    --help-languages   List all supported languages
    --help-voices      List all available voices
    --download         Download the ZenTTS model files and exit
    --license          Show whether this install is licensed to run
    --merge-chunks     Merge existing chunks in split-output directory into chapter files

Options:
    --stream            Stream audio instead of saving to file
    --speed <float>     Set speech speed, 0.5 to 2.0 (default: 1.0)
    --lang <str>        Set language: en-us or en-gb (default: en-us)
    --voice <str>       Set voice or blend voices (default: interactive selection)
    --split-output <dir> Save each chunk as separate file in directory
    --format <str>      Audio format: wav or mp3 (default: wav)
    --debug             Show detailed debug information
    --model <path>      Path to the ZenTTS .onnx model file
    --voices <path>     Path to the ZenTTS voices .bin file

Input formats:
    .txt               Text file input
    .epub              EPUB book input (will process chapters)
    .pdf               PDF document input (extracts chapters from TOC or content)

Multiple inputs:
    Pass as many input files as you like and they are joined, in the order you
    give them, into one output file. Formats can be mixed. If the last argument
    ends in .wav or .mp3 it is the output file, otherwise the name is derived
    from the first input.

Model files:
    On first run ZenTTS downloads the model files automatically and caches them in
    {cache}. Set ZENTTS_HOME to change that location, or ZENTTS_NO_DOWNLOAD=1 to
    disable downloading.

Examples:
    zentts input.txt output.wav --speed 1.2 --lang en-us --voice zen_us_f10
    zentts part1.txt part2.txt part3.txt book.wav
    zentts intro.txt chapters.epub notes.pdf audiobook.mp3 --format mp3
    zentts input.epub --split-output ./chunks/ --format mp3
    zentts input.pdf output.wav --speed 1.2 --lang en-gb --voice zen_uk_f02
    zentts input.pdf --split-output ./chunks/ --format mp3
    zentts input.txt --stream --speed 0.8
    zentts input.txt output.wav --voice "zen_us_f10:60,zen_us_m01:40"
    zentts input.txt --stream --voice "zen_us_m01,zen_us_f10" # 50-50 blend
    zentts --merge-chunks --split-output ./chunks/ --format wav
    zentts start --port 8000
    zentts --help-voices
    zentts --help-languages
    zentts input.epub --split-output ./chunks/ --debug
    zentts input.txt output.wav --model /path/to/model.onnx --voices /path/to/voices.bin
""".replace("{cache}", str(model_dir())))


def print_server_usage():
    print("""
ZenTTS - OpenAI-compatible speech API

Usage: zentts start [options]

Options:
    --host <str>          Address to bind (default: 127.0.0.1, use 0.0.0.0 to
                          accept connections from other machines)
    --port <int>          Port to listen on (default: 8000)
    --api-key <str>       Require `Authorization: Bearer <key>`; also read from
                          the ZENTTS_API_KEY environment variable
    --default-voice <str> Voice used when a request does not name one
                          (default: """ + DEFAULT_VOICE + """)
    --lang <str>          Default language: en-us or en-gb
    --model <path>        Path to the ZenTTS .onnx model file
    --voices <path>       Path to the ZenTTS voices .bin file
    --no-cors             Do not send CORS headers
    --quiet               Do not log requests
    -h, --help            Show this message

Endpoints:
    GET  /                  list of endpoints
    GET  /health            liveness check
    GET  /v1/models         models this server answers to
    GET  /v1/voices         ZenTTS voices and their OpenAI aliases
    POST /v1/audio/speech   generate speech (OpenAI compatible)

POST /v1/audio/speech accepts:
    input             the text to speak (required)
    voice             ZenTTS id, OpenAI name, or a blend "a:60,b:40"
    response_format   mp3, wav, flac, ogg, opus or pcm (default: mp3)
    speed             0.5 to 2.0, values outside that are clamped
    language          en-us or en-gb
    model             accepted and ignored, so any client works

Examples:
    zentts start
    zentts start --host 0.0.0.0 --port 8080
    zentts start --api-key secret --default-voice zen_uk_f02

    curl http://127.0.0.1:8000/v1/audio/speech \
      -H "Content-Type: application/json" \
      -d '{"input": "Hello.", "voice": "zen_us_f10"}' --output hello.mp3

Point any OpenAI client at it:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
    client.audio.speech.create(model="tts-1", voice="nova", input="Hello.")
""")


def print_supported_languages(model_path=None, voices_path=None):
    """Print all supported (English) languages."""
    engine = load_engine(model_path, voices_path)
    print("\nSupported languages:")
    for lang in _filter_english_languages(engine.get_languages()):
        print(f"    {lang}")
    print()


def print_supported_voices(model_path=None, voices_path=None):
    """Print all supported (English) voices, grouped by region and gender."""
    engine = load_engine(model_path, voices_path)
    print("\nZenTTS voices:")
    list_available_voices(engine, numbered=False)
    print('Blend any two with --voice "zen_us_f10:60,zen_us_m01:40"\n')


def validate_voice(voice, engine):
    """Validate a voice name (English only) and handle voice blending.

    Format for blended voices: "voice1:weight,voice2:weight"
    Example: "zen_us_f10:60,zen_us_m01:40" for a 60-40 blend
    """
    supported_voices = set(_filter_english_voices(engine.get_voices()))

    try:
        if "," in voice:
            voices = []
            weights = []

            for pair in voice.split(","):
                if ":" in pair:
                    v, w = pair.strip().split(":")
                    voices.append(v.strip())
                    weights.append(float(w.strip()))
                else:
                    voices.append(pair.strip())
                    weights.append(50.0)

            if len(voices) != 2:
                raise ValueError("voice blending needs two comma separated voices")

            for v in voices:
                if v not in supported_voices:
                    supported_voices_list = ", ".join(sorted(supported_voices))
                    raise ValueError(
                        f"Unsupported voice: {v}\nSupported voices are: {supported_voices_list}"
                    )

            total = sum(weights)
            if total != 100:
                weights = [w * (100 / total) for w in weights]

            style1 = engine.get_voice_style(voices[0])
            style2 = engine.get_voice_style(voices[1])
            return np.add(style1 * (weights[0] / 100), style2 * (weights[1] / 100))

        if voice not in supported_voices:
            supported_voices_list = ", ".join(sorted(supported_voices))
            raise ValueError(
                f"Unsupported voice: {voice}\nSupported voices are: {supported_voices_list}"
            )
        return voice
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def extract_chapters_from_epub(epub_file, debug=False):
    """Extract chapters from an EPUB file using ebooklib's metadata and TOC."""
    if not os.path.exists(epub_file):
        raise FileNotFoundError(f"EPUB file not found: {epub_file}")

    book = epub.read_epub(epub_file)
    chapters = []

    if debug:
        print("\nBook Metadata:")
        for key, value in book.metadata.items():
            print(f"  {key}: {value}")

        print("\nTable of Contents:")

        def print_toc(items, depth=0):
            for item in items:
                indent = "  " * depth
                if isinstance(item, tuple):
                    section_title, section_items = item
                    print(f"{indent}• Section: {section_title}")
                    print_toc(section_items, depth + 1)
                elif isinstance(item, epub.Link):
                    print(f"{indent}• {item.title} -> {item.href}")

        print_toc(book.toc)

    def get_chapter_content(soup, start_id, next_id=None):
        """Extract content between two fragment IDs."""
        content = []
        start_elem = soup.find(id=start_id)

        if not start_elem:
            return ""

        if start_elem.name in ["h1", "h2", "h3", "h4"]:
            current = start_elem.find_next_sibling()
        else:
            current = start_elem

        while current:
            if next_id and current.get("id") == next_id:
                break
            if (
                current.name in ["h1", "h2", "h3"]
                and "chapter" in current.get_text().lower()
            ):
                break
            content.append(current.get_text())
            current = current.find_next_sibling()

        return "\n".join(content).strip()

    def process_toc_items(items, depth=0):
        processed = []
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                section_title, section_items = item
                if debug:
                    print(f"{'  ' * depth}Processing section: {section_title}")
                processed.extend(process_toc_items(section_items, depth + 1))
            elif isinstance(item, epub.Link):
                if debug:
                    print(f"{'  ' * depth}Processing link: {item.title} -> {item.href}")

                if item.title.lower() in [
                    "copy",
                    "copyright",
                    "title page",
                    "cover",
                ] or item.title.lower().startswith("by"):
                    continue

                href_parts = item.href.split("#")
                file_name = href_parts[0]
                fragment_id = href_parts[1] if len(href_parts) > 1 else None

                doc = next(
                    (
                        doc
                        for doc in book.get_items_of_type(ITEM_DOCUMENT)
                        if doc.file_name.endswith(file_name)
                    ),
                    None,
                )

                if doc:
                    content = doc.get_content().decode("utf-8")
                    soup = BeautifulSoup(content, "html.parser")

                    if not fragment_id:
                        text_content = soup.get_text().strip()
                    else:
                        next_item = items[i + 1] if i + 1 < len(items) else None
                        next_fragment = None
                        if isinstance(next_item, epub.Link):
                            next_href_parts = next_item.href.split("#")
                            if (
                                next_href_parts[0] == file_name
                                and len(next_href_parts) > 1
                            ):
                                next_fragment = next_href_parts[1]

                        text_content = get_chapter_content(
                            soup, fragment_id, next_fragment
                        )

                    if text_content:
                        chapters.append(
                            {
                                "title": item.title,
                                "content": text_content,
                                "order": len(processed) + 1,
                            }
                        )
                        processed.append(item)
                        if debug:
                            print(f"{'  ' * depth}Added chapter: {item.title}")
                            print(
                                f"{'  ' * depth}Content length: {len(text_content)} chars"
                            )
                            print(
                                f"{'  ' * depth}Word count: {len(text_content.split())}"
                            )
        return processed

    process_toc_items(book.toc)

    if not chapters:
        if debug:
            print("\nNo chapters found in TOC, processing all documents...")

        docs = sorted(book.get_items_of_type(ITEM_DOCUMENT), key=lambda x: x.file_name)

        for doc in docs:
            if debug:
                print(f"Processing document: {doc.file_name}")

            content = doc.get_content().decode("utf-8")
            soup = BeautifulSoup(content, "html.parser")

            chapter_divs = soup.find_all(
                ["h1", "h2", "h3"], class_=lambda x: x and "chapter" in x.lower()
            )
            if not chapter_divs:
                chapter_divs = soup.find_all(
                    lambda tag: tag.name in ["h1", "h2", "h3"]
                    and (
                        "chapter" in tag.get_text().lower()
                        or "book" in tag.get_text().lower()
                    )
                )

            if chapter_divs:
                for i, div in enumerate(chapter_divs):
                    title = div.get_text().strip()

                    content = ""
                    for tag in div.find_next_siblings():
                        if tag.name in ["h1", "h2", "h3"] and (
                            "chapter" in tag.get_text().lower()
                            or "book" in tag.get_text().lower()
                        ):
                            break
                        content += tag.get_text() + "\n"

                    if content.strip():
                        chapters.append(
                            {
                                "title": title,
                                "content": content.strip(),
                                "order": len(chapters) + 1,
                            }
                        )
                        if debug:
                            print(f"Added chapter: {title}")
            else:
                text_content = soup.get_text().strip()
                if text_content:
                    title_tag = soup.find(["h1", "h2", "title"])
                    title = (
                        title_tag.get_text().strip()
                        if title_tag
                        else f"Chapter {len(chapters) + 1}"
                    )

                    if title.lower() not in [
                        "copy",
                        "copyright",
                        "title page",
                        "cover",
                    ]:
                        chapters.append(
                            {
                                "title": title,
                                "content": text_content,
                                "order": len(chapters) + 1,
                            }
                        )
                        if debug:
                            print(f"Added chapter: {title}")

    if chapters:
        print("\nSuccessfully extracted {} chapters:".format(len(chapters)))
        for chapter in chapters:
            print(f"  {chapter['order']}. {chapter['title']}")

        total_words = sum(len(chapter["content"].split()) for chapter in chapters)
        print("\nBook Summary:")
        print(f"Total Chapters: {len(chapters)}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Duration: {total_words / 150:.1f} minutes")

        if debug:
            print("\nDetailed Chapter List:")
            for chapter in chapters:
                word_count = len(chapter["content"].split())
                print(f"  • {chapter['title']}")
                print(f"    Words: {word_count:,}")
                print(f"    Duration: {word_count / 150:.1f} minutes")
    else:
        print("\nWarning: No chapters were extracted!")
        if debug:
            print("\nAvailable documents:")
            for doc in book.get_items_of_type(ITEM_DOCUMENT):
                print(f"  • {doc.file_name}")

    return chapters


class PdfParser:
    """Parser for extracting chapters from PDF files.

    Attempts to extract chapters first from the table of contents,
    then falls back to markdown-based extraction if that fails.
    """

    def __init__(
        self, pdf_path: str, debug: bool = False, min_chapter_length: int = 50
    ):
        self.pdf_path = pdf_path
        self.chapters = []
        self.debug = debug
        self.min_chapter_length = min_chapter_length

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    def get_chapters(self):
        if self.debug:
            print("\nDEBUG: Starting chapter extraction...")
            print(f"DEBUG: PDF file: {self.pdf_path}")
            print(f"DEBUG: Min chapter length: {self.min_chapter_length}")

        if self.get_chapters_from_toc():
            if self.debug:
                print(
                    f"\nDEBUG: Successfully extracted {len(self.chapters)} chapters from TOC"
                )
            return self.chapters

        if self.debug:
            print("\nDEBUG: TOC extraction failed, trying markdown conversion...")

        self.chapters = self.get_chapters_from_markdown()

        if self.debug:
            print("\nDEBUG: Markdown extraction complete")
            print(f"DEBUG: Found {len(self.chapters)} chapters")

        return self.chapters

    def get_chapters_from_toc(self):
        doc = None
        try:
            doc = pymupdf.open(self.pdf_path)
            toc = doc.get_toc()

            if not toc:
                if self.debug:
                    print("\nDEBUG: No table of contents found")
                return False

            print("\nTable of Contents:")
            for level, title, page in toc:
                title = self._clean_title(title)
                indent = "  " * (level - 1)
                print(f"{indent}{'•' if level > 1 else '>'} {title} (page {page})")

            if self.debug:
                print(f"\nDEBUG: Found {len(toc)} TOC entries")

            print("\nPress Enter to start processing, or Ctrl+C to cancel...")
            input()

            seen_pages = set()
            chapter_markers = []

            for level, title, page in toc:
                if level == 1:
                    title = self._clean_title(title)
                    if title and page not in seen_pages:
                        chapter_markers.append((title, page))
                        seen_pages.add(page)

            if not chapter_markers:
                if self.debug:
                    print("\nDEBUG: No level 1 chapters found in TOC")
                return False

            if self.debug:
                print(f"\nDEBUG: Found {len(chapter_markers)} chapters:")
                for title, page in chapter_markers:
                    print(f"DEBUG: • {title} (page {page})")

            for i, (title, start_page) in enumerate(chapter_markers):
                if self.debug:
                    print(f"\nDEBUG: Processing chapter {i + 1}/{len(chapter_markers)}")
                    print(f"DEBUG: Title: {title}")
                    print(f"DEBUG: Start page: {start_page}")

                end_page = (
                    chapter_markers[i + 1][1] - 1
                    if i < len(chapter_markers) - 1
                    else doc.page_count
                )

                chapter_text = self._extract_chapter_text(doc, start_page - 1, end_page)

                if len(chapter_text.strip()) > self.min_chapter_length:
                    self.chapters.append(
                        {"title": title, "content": chapter_text, "order": i + 1}
                    )
                    if self.debug:
                        print(
                            f"DEBUG: Added chapter with {len(chapter_text.split())} words"
                        )

            return bool(self.chapters)

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in TOC extraction: {str(e)}")
            return False

        finally:
            if doc:
                doc.close()

    def get_chapters_from_markdown(self):
        chapters = []
        try:

            def progress(current, total):
                if self.debug:
                    print(f"\rConverting page {current}/{total}...", end="", flush=True)

            md_text = pymupdf4llm.to_markdown(
                self.pdf_path, show_progress=True, progress_callback=progress
            )

            md_text = self._clean_markdown(md_text)

            current_chapter = None
            current_text = []
            chapter_count = 0

            for line in md_text.split("\n"):
                if line.startswith("#"):
                    if current_chapter and current_text:
                        chapter_text = "".join(current_text)
                        if len(chapter_text.strip()) > self.min_chapter_length:
                            chapters.append(
                                {
                                    "title": current_chapter,
                                    "content": chapter_text,
                                    "order": chapter_count,
                                }
                            )

                    chapter_count += 1
                    current_chapter = (
                        f"Chapter {chapter_count}_{line.lstrip('#').strip()}"
                    )
                    current_text = []
                else:
                    if current_chapter is not None:
                        current_text.append(line + "\n")

            if current_chapter and current_text:
                chapter_text = "".join(current_text)
                if len(chapter_text.strip()) > self.min_chapter_length:
                    chapters.append(
                        {
                            "title": current_chapter,
                            "content": chapter_text,
                            "order": chapter_count,
                        }
                    )

            return chapters

        except Exception as e:
            if self.debug:
                print(f"\nDEBUG: Error in markdown extraction: {str(e)}")
            return chapters

    def _clean_title(self, title: str) -> str:
        """Clean up chapter title text."""
        return title.strip().replace("\u200b", " ")

    def _clean_markdown(self, text: str) -> str:
        """Clean up converted markdown text."""
        text = text.replace("-", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_chapter_text(self, doc, start_page: int, end_page: int) -> str:
        """Extract text from PDF pages."""
        chapter_text = []
        for page_num in range(start_page, end_page):
            try:
                page = doc[page_num]
                text = page.get_text()
                chapter_text.append(text)
            except Exception as e:
                if self.debug:
                    print(f"\nDEBUG: Error extracting page {page_num}: {str(e)}")
                continue

        return "\n".join(chapter_text)


def process_chunk_sequential(
    chunk: str,
    engine: ZenTTS,
    voice: str,
    speed: float,
    lang: str,
    retry_count=0,
    debug=False,
):
    """Process a single chunk of text, splitting it further if it is too long."""
    try:
        if debug:
            sys.stdout.write("\033[K")
            sys.stdout.write(f"\nDEBUG: Processing chunk of length {len(chunk)}")
            if retry_count > 0:
                sys.stdout.write(
                    f"\nDEBUG: Retry #{retry_count} - Reduced chunk size to {len(chunk)}"
                )
            sys.stdout.write("\n")
            sys.stdout.flush()

        samples, sample_rate = engine.create(chunk, voice=voice, speed=speed, lang=lang)
        return samples, sample_rate
    except Exception as e:
        error_msg = str(e)
        if "index 510 is out of bounds" in error_msg:
            current_size = len(chunk)
            new_size = int(current_size * 0.6)

            if debug:
                sys.stdout.write("\033[K")
                sys.stdout.write(
                    f"\nDEBUG: Phoneme length error detected on chunk size {current_size}"
                )
                sys.stdout.write(f"\nDEBUG: Attempting retry with size {new_size}")
                sys.stdout.write("\n")
            else:
                sys.stdout.write("\033[K")
                sys.stdout.write(
                    "\rNote: Automatically handling a long text segment..."
                )
                sys.stdout.write("\n")
            sys.stdout.flush()

            words = chunk.split()
            current_piece = []
            current_size = 0
            pieces = []

            for word in words:
                word_size = len(word) + 1
                if current_size + word_size > new_size:
                    if current_piece:
                        pieces.append(" ".join(current_piece).strip())
                    current_piece = [word]
                    current_size = word_size
                else:
                    current_piece.append(word)
                    current_size += word_size

            if current_piece:
                pieces.append(" ".join(current_piece).strip())

            if debug:
                sys.stdout.write("\033[K")
                sys.stdout.write(f"\nDEBUG: Split chunk into {len(pieces)} pieces")
                for i, piece in enumerate(pieces, 1):
                    sys.stdout.write(f"\nDEBUG: Piece {i} length: {len(piece)}")
                sys.stdout.write("\n")
                sys.stdout.flush()

            all_samples = []
            last_sample_rate = None

            for i, piece in enumerate(pieces, 1):
                if debug:
                    sys.stdout.write("\033[K")
                    sys.stdout.write(f"\nDEBUG: Processing piece {i}/{len(pieces)}")
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                samples, sr = process_chunk_sequential(
                    piece, engine, voice, speed, lang, retry_count + 1, debug
                )
                if samples is not None:
                    all_samples.extend(samples)
                    last_sample_rate = sr

            if all_samples:
                if debug:
                    sys.stdout.write("\033[K")
                    sys.stdout.write(
                        f"\nDEBUG: Successfully processed all {len(pieces)} pieces"
                    )
                    sys.stdout.write("\n")
                sys.stdout.flush()
                return all_samples, last_sample_rate

            if debug:
                sys.stdout.write("\033[K")
                sys.stdout.write("\nDEBUG: Failed to process any pieces after splitting")
                sys.stdout.write("\n")
            sys.stdout.flush()

        if not debug:
            sys.stdout.write("\033[K")
            sys.stdout.write(
                "\rError: Unable to process text segment. Try using smaller chunks or enable debug mode for details."
            )
        else:
            sys.stdout.write("\033[K")
            sys.stdout.write(f"\nError processing chunk: {e}")
            sys.stdout.write(f"\nDEBUG: Full error message: {error_msg}")
            sys.stdout.write(f"\nDEBUG: Chunk length: {len(chunk)}")
        sys.stdout.write("\n")
        sys.stdout.flush()

        return None, None


def load_chapters(input_file, debug=False, stdin_indicators=None):
    """Read one input file and return its chapters.

    A `.txt` file is a single chapter titled after the file; EPUB and PDF are
    split into their own chapters.
    """
    if stdin_indicators is None:
        stdin_indicators = ["/dev/stdin", "-", "CONIN$"]

    if input_file.endswith(".epub"):
        chapters = extract_chapters_from_epub(input_file, debug)
        if not chapters:
            print(f"No chapters found in {input_file}.")
        return chapters

    if input_file.endswith(".pdf"):
        return PdfParser(input_file, debug=debug).get_chapters()

    if input_file in stdin_indicators:
        text = sys.stdin.read()
        title = "Chapter 1"
    else:
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read()
        title = os.path.splitext(os.path.basename(input_file))[0] or "Chapter 1"

    if not text.strip():
        print(f"Warning: {input_file} is empty, skipping")
        return []

    return [{"title": title, "content": text, "order": 1}]


def convert_text_to_audio(
    input_files,
    output_file=None,
    voice=None,
    speed=1.0,
    lang="en-us",
    stream=False,
    split_output=None,
    format="wav",
    debug=False,
    stdin_indicators=None,
    model_path=None,
    voices_path=None,
):
    global stop_spinner

    if stdin_indicators is None:
        stdin_indicators = ["/dev/stdin", "-", "CONIN$"]

    engine = load_engine(model_path, voices_path)

    lang = validate_language(lang, engine)

    if voice:
        voice = validate_voice(voice, engine)
    else:
        if any(f in stdin_indicators for f in input_files):
            print(
                f"Using stdin - automatically selecting default voice ({DEFAULT_VOICE})"
            )
            voice = DEFAULT_VOICE
        else:
            voices = list_available_voices(engine)
            print("\nHow to choose a voice:")
            print("You can use either a single voice or blend two voices together.")
            print("\nFor a single voice:")
            print("  • Just enter one number (example: '7')")
            print("\nFor blending two voices:")
            print("  • Enter two numbers separated by comma")
            print("  • Optionally add weights after each number using ':weight'")
            print("\nExamples:")
            print("  • '7'      - Use voice #7 only")
            print("  • '7,11'   - Mix voices #7 and #11 equally (50% each)")
            print("  • '7:60,11:40' - Mix 60% of voice #7 with 40% of voice #11")
            try:
                voice_input = input("Choose voice(s) by number: ")
                if "," in voice_input:
                    pairs = []
                    for pair in voice_input.split(","):
                        if ":" in pair:
                            num, weight = pair.strip().split(":")
                            voice_idx = int(num.strip()) - 1
                            if not (0 <= voice_idx < len(voices)):
                                raise ValueError(f"Invalid voice number: {int(num)}")
                            pairs.append(f"{voices[voice_idx]}:{weight}")
                        else:
                            voice_idx = int(pair.strip()) - 1
                            if not (0 <= voice_idx < len(voices)):
                                raise ValueError(f"Invalid voice number: {int(pair)}")
                            pairs.append(voices[voice_idx])
                    voice = ",".join(pairs)
                else:
                    voice_choice = int(voice_input) - 1
                    if not (0 <= voice_choice < len(voices)):
                        raise ValueError("Invalid choice")
                    voice = voices[voice_choice]
                voice = validate_voice(voice, engine)
            except (ValueError, IndexError):
                print("Invalid choice. Using default voice.")
                voice = DEFAULT_VOICE

    chapters = []
    needs_confirmation = False

    for input_file in input_files:
        if len(input_files) > 1:
            print(f"\nReading: {input_file}")

        chapters.extend(load_chapters(input_file, debug, stdin_indicators))

        if input_file.endswith(".epub"):
            needs_confirmation = True

    if not chapters:
        print("No text found to convert.")
        sys.exit(1)

    # Number the chapters across every input so a combined run stays in order.
    for order, chapter in enumerate(chapters, 1):
        chapter["order"] = order

    if len(input_files) > 1:
        total_words = sum(len(c["content"].split()) for c in chapters)
        print(
            f"\nCombining {len(input_files)} files into "
            f"{len(chapters)} chapter(s), {total_words:,} words "
            f"(about {total_words / 150:.1f} minutes)"
        )

    if needs_confirmation:
        print("\nPress Enter to start processing, or Ctrl+C to cancel...")
        input()

    if split_output:
        os.makedirs(split_output, exist_ok=True)

        print("\nCreating chapter directories and info files...")
        for chapter_num, chapter in enumerate(chapters, 1):
            chapter_dir = os.path.join(split_output, f"chapter_{chapter_num:03d}")
            os.makedirs(chapter_dir, exist_ok=True)

            info_file = os.path.join(chapter_dir, "info.txt")
            if not os.path.exists(info_file):
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {chapter['title']}\n")
                    f.write(f"Order: {chapter['order']}\n")
                    f.write(f"Words: {len(chapter['content'].split())}\n")
                    f.write(
                        f"Estimated Duration: {len(chapter['content'].split()) / 150:.1f} minutes\n"
                    )

        print("Created chapter directories and info files")

    if stream:
        import asyncio

        for chapter in chapters:
            print(f"\nStreaming: {chapter['title']}")
            asyncio.run(
                stream_audio(engine, chapter["content"], voice, speed, lang, debug)
            )
    else:
        if split_output:
            os.makedirs(split_output, exist_ok=True)

            for chapter_num, chapter in enumerate(chapters, 1):
                chapter_dir = os.path.join(split_output, f"chapter_{chapter_num:03d}")

                if os.path.exists(chapter_dir):
                    info_file = os.path.join(chapter_dir, "info.txt")
                    if os.path.exists(info_file):
                        chunks = chunk_text(chapter["content"], initial_chunk_size=1000)
                        total_chunks = len(chunks)
                        existing_chunks = len(
                            [
                                f
                                for f in os.listdir(chapter_dir)
                                if f.startswith("chunk_") and f.endswith(f".{format}")
                            ]
                        )

                        if existing_chunks == total_chunks:
                            print(
                                f"\nSkipping {chapter['title']}: Already completed ({existing_chunks} chunks)"
                            )
                            continue
                        elif existing_chunks:
                            print(
                                f"\nResuming {chapter['title']}: Found {existing_chunks}/{total_chunks} chunks"
                            )

                print(f"\nProcessing: {chapter['title']}")
                os.makedirs(chapter_dir, exist_ok=True)

                info_file = os.path.join(chapter_dir, "info.txt")
                if not os.path.exists(info_file):
                    with open(info_file, "w", encoding="utf-8") as f:
                        f.write(f"Title: {chapter['title']}\n")

                chunks = chunk_text(chapter["content"], initial_chunk_size=1000)
                total_chunks = len(chunks)
                processed_chunks = len(
                    [
                        f
                        for f in os.listdir(chapter_dir)
                        if f.startswith("chunk_") and f.endswith(f".{format}")
                    ]
                )

                for chunk_num, chunk in enumerate(chunks, 1):
                    if stop_audio:
                        break

                    chunk_file = os.path.join(
                        chapter_dir, f"chunk_{chunk_num:03d}.{format}"
                    )
                    if os.path.exists(chunk_file):
                        continue

                    filled = "■" * processed_chunks
                    remaining = "□" * (total_chunks - processed_chunks)
                    progress_bar = (
                        f"[{filled}{remaining}] ({processed_chunks}/{total_chunks})"
                    )

                    stop_spinner = False
                    spinner_thread = threading.Thread(
                        target=spinning_wheel,
                        args=(f"Processing {chapter['title']}", progress_bar),
                    )
                    spinner_thread.start()

                    try:
                        samples, sample_rate = process_chunk_sequential(
                            chunk,
                            engine,
                            voice,
                            speed,
                            lang,
                            retry_count=0,
                            debug=debug,
                        )
                        if samples is not None:
                            sf.write(chunk_file, samples, sample_rate)
                            processed_chunks += 1
                    except Exception as e:
                        print(f"\nError processing chunk {chunk_num}: {e}")

                    stop_spinner = True
                    spinner_thread.join()

                    if stop_audio:
                        break

                print(
                    f"\nCompleted {chapter['title']}: {processed_chunks}/{total_chunks} chunks processed"
                )

                if stop_audio:
                    break

            print(f"\nCreated audio files for {len(chapters)} chapters in {split_output}/")
        else:
            all_samples = []
            sample_rate = None

            for chapter_num, chapter in enumerate(chapters, 1):
                print(f"\nProcessing: {chapter['title']}")
                chunks = chunk_text(chapter["content"], initial_chunk_size=1000)
                processed_chunks = 0
                total_chunks = len(chunks)

                for chunk_num, chunk in enumerate(chunks, 1):
                    if stop_audio:
                        break

                    stop_spinner = False
                    spinner_thread = threading.Thread(
                        target=spinning_wheel,
                        args=(f"Processing chunk {chunk_num}/{total_chunks}",),
                    )
                    spinner_thread.start()

                    try:
                        samples, sr = process_chunk_sequential(
                            chunk,
                            engine,
                            voice,
                            speed,
                            lang,
                            retry_count=0,
                            debug=debug,
                        )
                        if samples is not None:
                            if sample_rate is None:
                                sample_rate = sr
                            all_samples.extend(samples)
                            processed_chunks += 1
                    except Exception as e:
                        print(f"\nError processing chunk {chunk_num}: {e}")

                    stop_spinner = True
                    spinner_thread.join()

                print(
                    f"\nCompleted {chapter['title']}: {processed_chunks}/{total_chunks} chunks processed"
                )

            if all_samples:
                print("\nSaving complete audio file...")
                if not output_file:
                    output_file = f"{os.path.splitext(input_files[0])[0]}.{format}"
                sf.write(output_file, all_samples, sample_rate)
                print(f"Created {output_file}")


def _load_sounddevice():
    """Import sounddevice on demand, since only streaming needs PortAudio."""
    try:
        import sounddevice as sd
    except OSError as e:
        print(
            f"Error: audio playback is unavailable ({e}).\n"
            "Streaming needs the PortAudio library. On Debian or Ubuntu install\n"
            "it with `sudo apt install libportaudio2`, or drop --stream and write\n"
            "the audio to a file instead."
        )
        sys.exit(1)
    return sd


async def stream_audio(engine, text, voice, speed, lang, debug=False):
    global stop_spinner, stop_audio
    stop_spinner = False
    stop_audio = False

    sd = _load_sounddevice()

    print("Starting audio stream...")
    chunks = chunk_text(text, initial_chunk_size=1000)

    for i, chunk in enumerate(chunks, 1):
        if stop_audio:
            break
        spinner_thread = threading.Thread(
            target=spinning_wheel, args=(f"Streaming chunk {i}/{len(chunks)}",)
        )
        spinner_thread.start()

        async for samples, sample_rate in engine.create_stream(
            chunk, voice=voice, speed=speed, lang=lang
        ):
            if stop_audio:
                break
            if debug:
                print(f"\nDEBUG: Playing chunk of {len(samples)} samples")
            sd.play(samples, sample_rate)
            sd.wait()

        stop_spinner = True
        spinner_thread.join()
        stop_spinner = False

    print("\nStreaming completed.")


def handle_ctrl_c(signum, frame):
    global stop_spinner, stop_audio
    print("\nCtrl+C detected, stopping...")
    stop_spinner = True
    stop_audio = True
    sys.exit(0)


signal.signal(signal.SIGINT, handle_ctrl_c)


def merge_chunks_to_chapters(split_output_dir, format="wav"):
    """Merge audio chunks into complete chapter files."""
    if not os.path.exists(split_output_dir):
        print(f"Error: Directory {split_output_dir} does not exist.")
        return

    chapter_dirs = sorted(
        [
            d
            for d in os.listdir(split_output_dir)
            if d.startswith("chapter_")
            and os.path.isdir(os.path.join(split_output_dir, d))
        ]
    )

    if not chapter_dirs:
        print(f"No chapter directories found in {split_output_dir}")
        return

    used_titles = set()

    for chapter_dir in chapter_dirs:
        chapter_path = os.path.join(split_output_dir, chapter_dir)
        chunk_files = sorted(
            [
                f
                for f in os.listdir(chapter_path)
                if f.startswith("chunk_") and f.endswith(f".{format}")
            ]
        )

        if not chunk_files:
            print(f"No chunks found in {chapter_dir}")
            continue

        chapter_title = chapter_dir
        info_file = os.path.join(chapter_path, "info.txt")
        if os.path.exists(info_file):
            with open(info_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Title:"):
                        chapter_title = line.replace("Title:", "").strip()
                        break

        safe_title = "".join(
            c for c in chapter_title if c.isalnum() or c in (" ", "-", "_")
        ).strip()

        if not safe_title or safe_title in used_titles:
            merged_file = os.path.join(split_output_dir, f"{chapter_dir}.{format}")
        else:
            merged_file = os.path.join(split_output_dir, f"{safe_title}.{format}")
            used_titles.add(safe_title)

        print(f"\nMerging chunks for {chapter_title}")

        all_samples = []
        sample_rate = None
        total_duration = 0

        total_chunks = len(chunk_files)
        processed_chunks = 0

        for chunk_file in chunk_files:
            chunk_path = os.path.join(chapter_path, chunk_file)

            print(f"\rProcessing chunk {processed_chunks + 1}/{total_chunks}", end="")

            try:
                data, sr = sf.read(chunk_path)

                if len(data) == 0:
                    print(f"\nWarning: Empty audio data in {chunk_file}")
                    continue

                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    print(f"\nWarning: Sample rate mismatch in {chunk_file}")
                    continue

                total_duration += len(data) / sr
                all_samples.extend(data)
                processed_chunks += 1

            except Exception as e:
                print(f"\nError processing {chunk_file}: {e}")

        print()

        if all_samples:
            print(f"Saving merged chapter to {merged_file}")
            print(f"Total duration: {total_duration:.2f} seconds")

            try:
                all_samples = np.array(all_samples)

                sf.write(merged_file, all_samples, sample_rate)
                print(f"Successfully merged {processed_chunks}/{total_chunks} chunks")

                if os.path.exists(merged_file):
                    output_data, output_sr = sf.read(merged_file)
                    output_duration = len(output_data) / output_sr
                    print(f"Verified output file: {output_duration:.2f} seconds")
                else:
                    print("Warning: Output file was not created")

            except Exception as e:
                print(f"Error saving merged file: {e}")
        else:
            print("No valid audio data to merge")


def get_valid_options():
    """Return the set of valid command line options."""
    return {
        "-h",
        "--help",
        "--help-languages",
        "--help-voices",
        "--download",
        "--merge-chunks",
        "--stream",
        "--speed",
        "--lang",
        "--voice",
        "--split-output",
        "--format",
        "--debug",
        "--model",
        "--voices",
        "-v",
        "--version",
        "--host",
        "--port",
        "--api-key",
        "--default-voice",
        "--no-cors",
        "--quiet",
        "--license",
    }


##############################################################################
# Activation
##############################################################################

# How long an install may run on its last check.
LICENSE_GRACE_DAYS = 7

# How often to check again once a check has succeeded.
LICENSE_CHECK_HOURS = 12


def license_state_path():
    return model_dir() / "license.json"


def _read_license_state():
    try:
        with open(license_state_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_license_state(state):
    try:
        license_state_path().parent.mkdir(parents=True, exist_ok=True)
        with open(license_state_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except OSError as e:
        log.debug(f"Could not store the licence state: {e}")


def _fetch_json(url, timeout=8):
    request = urllib.request.Request(url, headers={"User-Agent": f"zentts/{__version__}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _version_tuple(text):
    parts = []
    for piece in str(text).split("."):
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0, 0, 0])[:3]


def check_license(force=False, quiet=True):
    """Check whether this release is still supported.

    Returns (allowed, message). The result is cached, so a normal run does not
    call out every time.
    """
    _LICENSE_CONTROL_URL = (
        "https://raw.githubusercontent.com/OmorDeveloper/zentts/master/control.json"
    )
    _LICENSE_PACKAGE_URL = "https://pypi.org/pypi/zentts/json"

    state = _read_license_state()
    now = time.time()

    if state.get("disabled"):
        return False, state.get("message") or "This copy of ZenTTS has been disabled."

    checked_at = float(state.get("checked_at") or 0)
    if not force and now - checked_at < LICENSE_CHECK_HOURS * 3600:
        return True, "using the cached licence check"

    url = os.getenv("ZENTTS_LICENSE_URL", _LICENSE_CONTROL_URL)
    try:
        control = _fetch_json(url)
        if not isinstance(control, dict):
            raise ValueError("control file is not an object")

        if not control.get("enabled", True):
            message = control.get("message") or "This copy of ZenTTS has been disabled."
            _write_license_state(
                {"disabled": True, "message": message, "checked_at": now}
            )
            return False, message

        minimum = control.get("min_version")
        if minimum and _version_tuple(__version__) < _version_tuple(minimum):
            message = control.get("outdated_message") or (
                f"ZenTTS {__version__} is no longer supported. "
                f"Upgrade to {minimum} or later: pip install --upgrade zentts"
            )
            return False, message

        if __version__ in (control.get("blocked_versions") or []):
            message = control.get("message") or (
                f"ZenTTS {__version__} has been withdrawn. "
                "Run: pip install --upgrade zentts"
            )
            return False, message

        # The package itself must still be published.
        if control.get("require_package", True):
            try:
                _fetch_json(_LICENSE_PACKAGE_URL)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    message = "The zentts package is no longer published, so it will not run."
                    _write_license_state(
                        {"disabled": True, "message": message, "checked_at": now}
                    )
                    return False, message
                raise

        _write_license_state({"disabled": False, "checked_at": now})
        return True, "licence check passed"

    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
        # Offline. Run on the last good check until the grace period expires.
        if not checked_at:
            return False, (
                "ZenTTS could not verify its licence and has never checked in "
                f"successfully ({e}). Connect to the internet once to activate it."
            )
        days = (now - checked_at) / 86400
        if days > LICENSE_GRACE_DAYS:
            return False, (
                f"ZenTTS has been offline for {days:.0f} days, past the "
                f"{LICENSE_GRACE_DAYS} day limit. Connect to the internet to continue."
            )
        if not quiet:
            print(f"Note: licence check skipped, offline for {days:.1f} days")
        return True, f"offline, {LICENSE_GRACE_DAYS - days:.1f} days of grace left"


def require_license():
    """Stop unless this release is supported."""
    allowed, message = check_license()
    if not allowed:
        print("\nZenTTS is not available.\n")
        print(f"  {message}\n")
        sys.exit(2)


##############################################################################
# Sessions, history and logs
##############################################################################

# Requests kept in memory for the browser log view.
REQUEST_LOG = collections.deque(maxlen=500)

# Cap on stored audio, so history cannot fill a disk unnoticed.
MAX_SESSION_ITEMS = 500


def sessions_dir():
    return model_dir() / "sessions"


def _session_path(session_id):
    return sessions_dir() / f"{session_id}.json"


def _safe_session_id(session_id):
    """Reject anything that is not a plain id, so paths cannot escape."""
    session_id = str(session_id or "")
    if not session_id or not all(c.isalnum() or c in "-_" for c in session_id):
        raise ValueError(f"Invalid session id: {session_id}")
    return session_id


def new_session(name=None):
    """Create a session and write it to disk."""
    session_id = f"s{int(time.time() * 1000):x}{os.urandom(2).hex()}"
    session = {
        "id": session_id,
        "name": name or time.strftime("Session %Y-%m-%d %H:%M"),
        "created": time.time(),
        "updated": time.time(),
        "items": [],
    }
    save_session(session)
    return session


def save_session(session):
    directory = sessions_dir()
    directory.mkdir(parents=True, exist_ok=True)
    session["updated"] = time.time()
    path = _session_path(session["id"])
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(session, fh, indent=2)
    tmp.replace(path)
    return session


def load_session(session_id):
    path = _session_path(_safe_session_id(session_id))
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def list_sessions():
    """Every session, newest first, without the item bodies."""
    directory = sessions_dir()
    if not directory.exists():
        return []

    sessions = []
    for path in directory.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as fh:
                session = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        sessions.append(
            {
                "id": session.get("id", path.stem),
                "name": session.get("name", path.stem),
                "created": session.get("created", 0),
                "updated": session.get("updated", 0),
                "items": len(session.get("items", [])),
            }
        )
    return sorted(sessions, key=lambda s: s["updated"], reverse=True)


def delete_session(session_id):
    """Remove a session and every audio file it owns."""
    session = load_session(session_id)
    if not session:
        return False

    for item in session.get("items", []):
        audio = sessions_dir() / "audio" / item.get("file", "")
        if item.get("file"):
            try:
                audio.unlink(missing_ok=True)
            except OSError as e:
                log.debug(f"Could not delete {audio}: {e}")

    _session_path(session["id"]).unlink(missing_ok=True)
    return True


def rename_session(session_id, name):
    session = load_session(session_id)
    if not session:
        return None
    name = str(name).strip()
    if not name:
        raise ValueError("Name cannot be empty")
    session["name"] = name[:200]
    return save_session(session)


def add_session_item(session_id, audio, meta):
    """Store one generated clip in a session and return its record."""
    session = load_session(session_id)
    if not session:
        return None

    audio_dir = sessions_dir() / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    item_id = f"i{int(time.time() * 1000):x}{os.urandom(2).hex()}"
    filename = f"{session['id']}_{item_id}.{meta['format']}"
    with open(audio_dir / filename, "wb") as fh:
        fh.write(audio)

    item = {
        "id": item_id,
        "created": time.time(),
        "text": meta["text"][:5000],
        "voice": meta["voice"],
        "format": meta["format"],
        "speed": meta["speed"],
        "language": meta["language"],
        "seconds": meta.get("seconds", 0),
        "bytes": len(audio),
        "file": filename,
    }

    session.setdefault("items", []).append(item)

    # Keep history bounded, oldest first.
    while len(session["items"]) > MAX_SESSION_ITEMS:
        dropped = session["items"].pop(0)
        if dropped.get("file"):
            try:
                (audio_dir / dropped["file"]).unlink(missing_ok=True)
            except OSError:
                pass

    save_session(session)
    return item


def session_item_path(session_id, item_id):
    """Locate the audio file for one history item."""
    session = load_session(session_id)
    if not session:
        return None, None
    for item in session.get("items", []):
        if item["id"] == item_id:
            return sessions_dir() / "audio" / item["file"], item
    return None, None


##############################################################################
# HTTP server (OpenAI-compatible)
##############################################################################


# Default address for `zentts start`.
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# The model ids this server answers to. Anything else is accepted too, so a
# client hard-coded to "tts-1" keeps working.
SERVER_MODELS = ["zentts-1", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"]

# OpenAI voice names mapped onto their closest ZenTTS voice, so a client
# written against the OpenAI API needs no changes.
OPENAI_VOICE_ALIASES = {
    "alloy": "zen_us_f01",
    "ash": "zen_us_m01",
    "ballad": "zen_uk_m03",
    "coral": "zen_us_f06",
    "echo": "zen_us_m02",
    "fable": "zen_uk_m02",
    "nova": "zen_us_f08",
    "onyx": "zen_us_m07",
    "sage": "zen_us_f09",
    "shimmer": "zen_us_f11",
    "verse": "zen_us_m05",
}

# Formats whose encoded chunks can simply follow one another in a stream.
STREAMABLE_FORMATS = {"mp3", "pcm"}

# Largest request body accepted, so a bad client cannot exhaust memory.
MAX_REQUEST_BYTES = 10 * 1024 * 1024

# response_format -> (soundfile format, subtype, content type)
SERVER_FORMATS = {
    "mp3": ("MP3", None, "audio/mpeg"),
    "wav": ("WAV", None, "audio/wav"),
    "flac": ("FLAC", None, "audio/flac"),
    "ogg": ("OGG", "VORBIS", "audio/ogg"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/pcm"),  # raw 24 kHz 16-bit mono, like OpenAI
}


def resolve_api_voice(name, engine):
    """Turn a requested voice into something the engine accepts.

    Accepts a ZenTTS id, an OpenAI voice name, or a blend such as
    "zen_us_f10:60,zen_us_m01:40".
    """
    name = (name or DEFAULT_VOICE).strip()
    available = set(engine.get_voices())

    if "," in name:
        voices, weights = [], []
        for pair in name.split(","):
            if ":" in pair:
                voice, weight = pair.strip().split(":", 1)
                voices.append(OPENAI_VOICE_ALIASES.get(voice.strip(), voice.strip()))
                weights.append(float(weight))
            else:
                alias = pair.strip()
                voices.append(OPENAI_VOICE_ALIASES.get(alias, alias))
                weights.append(50.0)

        if len(voices) != 2:
            raise ValueError("Blending takes exactly two comma separated voices")
        for voice in voices:
            if voice not in available:
                raise ValueError(f"Unknown voice: {voice}")

        total = sum(weights) or 100.0
        weights = [w * (100 / total) for w in weights]
        first = engine.get_voice_style(voices[0])
        second = engine.get_voice_style(voices[1])
        return np.add(first * (weights[0] / 100), second * (weights[1] / 100))

    voice = OPENAI_VOICE_ALIASES.get(name, name)
    if voice not in available:
        raise ValueError(
            f"Unknown voice: {name}. Use a ZenTTS id such as {DEFAULT_VOICE}, or an "
            f"OpenAI name such as {', '.join(sorted(OPENAI_VOICE_ALIASES)[:4])}."
        )
    return voice


def encode_audio(samples, sample_rate, response_format):
    """Encode samples in the requested format, returning bytes and its type."""
    if response_format not in SERVER_FORMATS:
        raise ValueError(
            f"Unsupported response_format: {response_format}. "
            f"Supported: {', '.join(sorted(SERVER_FORMATS))}"
        )

    sound_format, subtype, content_type = SERVER_FORMATS[response_format]
    samples = np.asarray(samples, dtype=np.float32)

    if sound_format is None:  # raw PCM, 16-bit little-endian
        clipped = np.clip(samples, -1.0, 1.0)
        return (clipped * 32767).astype("<i2").tobytes(), content_type

    buffer = io.BytesIO()
    kwargs = {"subtype": subtype} if subtype else {}
    sf.write(buffer, samples, sample_rate, format=sound_format, **kwargs)
    return buffer.getvalue(), content_type


def synthesize(engine, text, voice, speed=1.0, lang="en-us"):
    """Synthesise text for the API, splitting long input into chunks."""
    pieces = []
    for chunk in chunk_text(text, initial_chunk_size=1000) or [text]:
        samples, _ = engine.create(chunk, voice=voice, speed=speed, lang=lang)
        if len(samples):
            pieces.append(samples)

    if not pieces:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE
    return np.concatenate(pieces), SAMPLE_RATE


##############################################################################
# Web interface
##############################################################################

WEB_UI = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZenTTS Studio</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#16181d;--muted:#666d7a;--line:#e2e5ea;
--accent:#3d5afe;--accent-ink:#fff;--danger:#c0392b;--radius:10px}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--panel:#1c1f25;--ink:#e8eaee;
--muted:#98a0ad;--line:#2a2f38;--accent:#6f86ff;--accent-ink:#0d0f13}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--ink)}
header{display:flex;align-items:baseline;gap:12px;padding:14px 20px;
border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
header h1{font-size:17px;margin:0;letter-spacing:.2px}
header .by{color:var(--muted);font-size:13px}
header .spacer{flex:1}
.layout{display:grid;grid-template-columns:250px 1fr;gap:18px;padding:18px;
max-width:1200px;margin:0 auto;align-items:start}
@media (max-width:820px){.layout{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
margin:0 0 10px}
textarea{width:100%;min-height:150px;resize:vertical;padding:10px;border-radius:8px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
select,input[type=text],input[type=number]{width:100%;padding:8px;border-radius:8px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);font:inherit}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
button.link{border:0;background:none;color:var(--muted);padding:2px 6px}
button.link:hover{color:var(--accent)}
button[disabled]{opacity:.55;cursor:not-allowed}
.actions{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
.status{color:var(--muted);font-size:13px}
ul{list-style:none;margin:0;padding:0}
li.session{display:flex;align-items:center;gap:6px;padding:7px 8px;border-radius:8px;cursor:pointer}
li.session:hover{background:var(--bg)}
li.session.on{background:var(--bg);box-shadow:inset 2px 0 0 var(--accent)}
li.session .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
li.session .ct{color:var(--muted);font-size:12px}
.item{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
.item .txt{font-size:14px;margin-bottom:6px}
.item .meta{color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap}
audio{width:100%;margin-top:8px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:10px;
overflow:auto;max-height:340px;font-size:12px;margin:0}
.tabs{display:flex;gap:6px;margin-bottom:12px}
.tabs button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.hide{display:none}
.err{color:var(--danger)}
</style>
</head>
<body>
<header>
  <h1>ZenTTS Studio</h1>
  <span class="by" id="ver"></span>
  <span class="spacer"></span>
  <span class="by"><a href="/api" style="color:inherit">API</a></span>
</header>

<div class="layout">
  <aside class="card">
    <h2>Sessions</h2>
    <ul id="sessions"></ul>
    <div class="actions"><button id="newSession">New session</button></div>
  </aside>

  <main>
    <div class="tabs">
      <button id="tabStudio" class="on">Studio</button>
      <button id="tabHistory">History</button>
      <button id="tabLogs">Logs</button>
    </div>

    <section id="studio" class="card">
      <h2>Text</h2>
      <textarea id="text" placeholder="Paste any amount of text. Long text is split, spoken in order and joined into one file."></textarea>
      <div class="row">
        <div><label for="voice">Voice</label><select id="voice"></select></div>
        <div><label for="lang">Language</label><select id="lang">
          <option value="en-us">en-us</option><option value="en-gb">en-gb</option></select></div>
        <div><label for="format">Format</label><select id="format"></select></div>
        <div><label for="speed">Speed</label>
          <input id="speed" type="number" min="0.5" max="2" step="0.1" value="1"></div>
      </div>
      <div class="actions">
        <button id="go" class="primary">Generate</button>
        <button id="stream">Stream</button>
        <a id="dl" class="hide"><button>Download</button></a>
        <span class="status" id="status"></span>
      </div>
      <audio id="player" controls class="hide"></audio>
    </section>

    <section id="history" class="card hide">
      <h2>History <span class="status" id="histName"></span></h2>
      <div id="items"></div>
    </section>

    <section id="logs" class="card hide">
      <h2>Requests</h2>
      <div class="actions" style="margin:0 0 10px">
        <button id="refreshLogs">Refresh</button>
      </div>
      <pre id="logbox"></pre>
    </section>
  </main>
</div>

<script>
const $ = id => document.getElementById(id);
let sessionId = localStorage.getItem('zentts_session') || null;
let lastBlob = null;

const api = async (path, options) => {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).error.message; } catch (e) {}
    throw new Error(detail);
  }
  return response.json();
};

async function boot() {
  const index = await api('/api');
  $('ver').textContent = 'v' + index.version + ' - by ' + index.author;
  const voices = await api('/v1/voices');
  const groups = {};
  voices.data.forEach(v => { (groups[v.description] ||= []).push(v); });
  $('voice').innerHTML = Object.entries(groups).map(([label, list]) =>
    `<optgroup label="${label}">` + list.map(v =>
      `<option value="${v.id}"${v.id === voices.default ? ' selected' : ''}>${v.id}</option>`
    ).join('') + '</optgroup>').join('');
  $('format').innerHTML = index.speech_request.response_format
    .map(f => `<option${f === 'mp3' ? ' selected' : ''}>${f}</option>`).join('');
  await loadSessions();
}

async function loadSessions() {
  const data = await api('/v1/sessions');
  if (!data.data.length) { await api('/v1/sessions', {method: 'POST'}); return loadSessions(); }
  if (!data.data.some(s => s.id === sessionId)) sessionId = data.data[0].id;
  localStorage.setItem('zentts_session', sessionId);
  $('sessions').innerHTML = data.data.map(s => `
    <li class="session ${s.id === sessionId ? 'on' : ''}" data-id="${s.id}">
      <span class="nm">${escapeHtml(s.name)}</span>
      <span class="ct">${s.items}</span>
      <button class="link" data-act="rename" title="Rename">rename</button>
      <button class="link" data-act="delete" title="Delete">del</button>
    </li>`).join('');
  $('sessions').querySelectorAll('li').forEach(li => {
    li.onclick = async event => {
      const id = li.dataset.id, act = event.target.dataset.act;
      if (act === 'rename') {
        const name = prompt('New name', li.querySelector('.nm').textContent);
        if (name) await api('/v1/sessions/' + id, {method: 'PATCH',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name})});
      } else if (act === 'delete') {
        if (!confirm('Delete this session and its audio?')) return;
        await api('/v1/sessions/' + id, {method: 'DELETE'});
        if (id === sessionId) sessionId = null;
      } else {
        sessionId = id;
      }
      localStorage.setItem('zentts_session', sessionId || '');
      await loadSessions();
      if (!$('history').classList.contains('hide')) loadHistory();
    };
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, c =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
}

function body() {
  return {
    input: $('text').value,
    voice: $('voice').value,
    language: $('lang').value,
    response_format: $('format').value,
    speed: parseFloat($('speed').value) || 1,
    session_id: sessionId
  };
}

async function generate(stream) {
  const payload = body();
  if (!payload.input.trim()) { $('status').textContent = 'Enter some text first.'; return; }
  if (stream) { payload.stream = true; payload.response_format = 'mp3'; }

  $('go').disabled = $('stream').disabled = true;
  $('status').textContent = stream ? 'Streaming...' : 'Generating...';
  const started = Date.now();

  try {
    const response = await fetch('/v1/audio/speech', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error((await response.json()).error.message);

    lastBlob = await response.blob();
    const url = URL.createObjectURL(lastBlob);
    $('player').src = url;
    $('player').classList.remove('hide');
    $('player').play().catch(() => {});
    const link = $('dl');
    link.href = url;
    link.download = 'zentts.' + payload.response_format;
    link.classList.remove('hide');
    $('status').textContent =
      `${(lastBlob.size / 1024).toFixed(0)} kB in ${((Date.now() - started) / 1000).toFixed(1)}s`;
    await loadSessions();
    if (!$('history').classList.contains('hide')) loadHistory();
  } catch (e) {
    $('status').innerHTML = '<span class="err">' + escapeHtml(e.message) + '</span>';
  } finally {
    $('go').disabled = $('stream').disabled = false;
  }
}

async function loadHistory() {
  if (!sessionId) return;
  const session = await api('/v1/sessions/' + sessionId);
  $('histName').textContent = session.name;
  $('items').innerHTML = session.items.length ? session.items.slice().reverse().map(i => `
    <div class="item">
      <div class="txt">${escapeHtml(i.text.slice(0, 240))}${i.text.length > 240 ? '...' : ''}</div>
      <div class="meta">
        <span>${i.voice}</span><span>${i.language}</span><span>x${i.speed}</span>
        <span>${i.format}</span><span>${(i.bytes / 1024).toFixed(0)} kB</span>
        <span>${new Date(i.created * 1000).toLocaleString()}</span>
      </div>
      <audio controls preload="none" src="/v1/sessions/${sessionId}/items/${i.id}/audio"></audio>
      <div class="actions">
        <a href="/v1/sessions/${sessionId}/items/${i.id}/audio?download=1"><button>Download</button></a>
      </div>
    </div>`).join('') : '<p class="status">Nothing generated in this session yet.</p>';
}

async function loadLogs() {
  const data = await api('/v1/logs');
  $('logbox').textContent = data.data.map(l =>
    `${new Date(l.time * 1000).toLocaleTimeString()}  ${String(l.status).padEnd(4)} ` +
    `${l.method.padEnd(6)} ${l.path}  ${l.ms}ms${l.detail ? '  ' + l.detail : ''}`
  ).join('\\n') || 'No requests yet.';
}

function tab(name) {
  ['studio', 'history', 'logs'].forEach(section => {
    $(section).classList.toggle('hide', section !== name);
    $('tab' + section[0].toUpperCase() + section.slice(1)).classList.toggle('on', section === name);
  });
  if (name === 'history') loadHistory();
  if (name === 'logs') loadLogs();
}

$('go').onclick = () => generate(false);
$('stream').onclick = () => generate(true);
$('newSession').onclick = async () => {
  const session = await api('/v1/sessions', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({})});
  sessionId = session.id;
  localStorage.setItem('zentts_session', sessionId);
  await loadSessions();
};
$('tabStudio').onclick = () => tab('studio');
$('tabHistory').onclick = () => tab('history');
$('tabLogs').onclick = () => tab('logs');
$('refreshLogs').onclick = loadLogs;

boot().catch(e => { $('status').innerHTML = '<span class="err">' + e.message + '</span>'; });
</script>
</body>
</html>
"""


##############################################################################
# Request handler
##############################################################################


class ZenTTSHandler(BaseHTTPRequestHandler):
    """Serves the web interface and the OpenAI-compatible endpoints."""

    server_version = f"ZenTTS/{__version__}"
    protocol_version = "HTTP/1.1"

    # Replaced by run_server.
    engine = None
    api_key = None
    default_voice = DEFAULT_VOICE
    default_lang = "en-us"
    allow_cors = True
    quiet = False

    # -- plumbing ---------------------------------------------------------

    def log_message(self, format, *args):
        if not self.quiet:
            message = format % args
            sys.stderr.write(
                f"{self.log_date_time_string()} {self.address_string()} {message}\n"
            )

    def _record(self, status, detail=""):
        REQUEST_LOG.append(
            {
                "time": time.time(),
                "method": self.command,
                "path": self.path,
                "status": status,
                "ms": int((time.time() - getattr(self, "_started", time.time())) * 1000),
                "detail": detail,
            }
        )

    def _send(self, status, body=b"", content_type="application/json", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self.allow_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers", "Authorization, Content-Type"
            )
            self.send_header(
                "Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS"
            )
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self._record(status)

    def _send_json(self, status, payload, extra=None):
        self._send(status, json.dumps(payload, indent=2), "application/json", extra)

    def _drain_body(self):
        """Discard an unread request body, so keep-alive stays in sync.

        Draining a body that was already read would block until the client
        gives up, so this is a no-op in that case.
        """
        if getattr(self, "_body_read", False):
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        remaining = min(length, MAX_REQUEST_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def _send_error(self, status, message, kind="invalid_request_error"):
        # Shaped like an OpenAI error, so existing clients can read it.
        self._drain_body()
        self._send(
            status,
            json.dumps({"error": {"message": message, "type": kind}}, indent=2),
            "application/json",
        )
        REQUEST_LOG[-1]["detail"] = message[:120]

    def _authorized(self):
        if not self.api_key:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[7:].strip() == self.api_key:
            return True
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        return query.get("api_key", [None])[0] == self.api_key

    def _read_json(self):
        """Read a JSON body, or raise ValueError with a reason."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ValueError("Invalid Content-Length")

        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            raise ValueError(f"Request body is larger than {MAX_REQUEST_BYTES} bytes")

        if length <= 0:
            self._body_read = True
            return {}

        raw = self.rfile.read(length)
        self._body_read = True
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Body must be JSON: {e}")
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object")
        return payload

    # -- routing ----------------------------------------------------------

    def do_OPTIONS(self):
        self._started = time.time()
        self._send(204)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._started = time.time()
        self._body_read = True
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        parts = [p for p in path.split("/") if p]

        if path == "/":
            return self._send(200, WEB_UI, "text/html; charset=utf-8")

        if path in ("/api", "/v1"):
            return self._send_json(200, self._index())

        if path in ("/health", "/healthz"):
            return self._send_json(200, {"status": "ok", "version": __version__})

        if path in ("/v1/models", "/models"):
            return self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": name, "object": "model", "owned_by": "zentts"}
                        for name in SERVER_MODELS
                    ],
                },
            )

        if path in ("/v1/voices", "/voices"):
            return self._send_json(200, self._voices())

        if path == "/v1/license":
            allowed, message = check_license()
            return self._send_json(
                200, {"enabled": allowed, "message": message, "version": __version__}
            )

        if path == "/v1/logs":
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            return self._send_json(200, {"object": "list", "data": list(REQUEST_LOG)})

        if path == "/v1/sessions":
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            return self._send_json(200, {"object": "list", "data": list_sessions()})

        # /v1/sessions/<id>
        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            session = load_session(parts[2])
            if not session:
                return self._send_error(404, f"No such session: {parts[2]}", "not_found_error")
            return self._send_json(200, session)

        # /v1/sessions/<id>/items/<item>/audio
        if (
            len(parts) == 6
            and parts[:2] == ["v1", "sessions"]
            and parts[3] == "items"
            and parts[5] == "audio"
        ):
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            return self._send_item_audio(parts[2], parts[4], parsed.query)

        self._send_error(404, f"Unknown endpoint: {path}", "not_found_error")

    def do_POST(self):
        self._started = time.time()
        self._body_read = False
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

        if path in ("/v1/audio/speech", "/audio/speech"):
            return self._speech()

        if path == "/v1/sessions":
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            try:
                payload = self._read_json()
            except ValueError as e:
                return self._send_error(400, str(e))
            return self._send_json(201, new_session(payload.get("name")))

        self._send_error(404, f"Unknown endpoint: {path}", "not_found_error")

    def do_PATCH(self):
        self._started = time.time()
        self._body_read = False
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]

        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            try:
                payload = self._read_json()
                session = rename_session(parts[2], payload.get("name", ""))
            except ValueError as e:
                return self._send_error(400, str(e))
            if not session:
                return self._send_error(404, f"No such session: {parts[2]}", "not_found_error")
            return self._send_json(200, session)

        self._send_error(404, "Unknown endpoint", "not_found_error")

    def do_DELETE(self):
        self._started = time.time()
        self._body_read = True
        parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]

        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            if not self._authorized():
                return self._send_error(401, "Invalid API key", "authentication_error")
            try:
                deleted = delete_session(parts[2])
            except ValueError as e:
                return self._send_error(400, str(e))
            if not deleted:
                return self._send_error(404, f"No such session: {parts[2]}", "not_found_error")
            return self._send_json(200, {"id": parts[2], "deleted": True})

        self._send_error(404, "Unknown endpoint", "not_found_error")

    # -- endpoints --------------------------------------------------------

    def _send_item_audio(self, session_id, item_id, query):
        try:
            path, item = session_item_path(session_id, item_id)
        except ValueError as e:
            return self._send_error(400, str(e))

        if not path or not path.exists():
            return self._send_error(404, "No such history item", "not_found_error")

        _, _, content_type = SERVER_FORMATS.get(item["format"], (None, None, "audio/mpeg"))
        extra = {}
        if urllib.parse.parse_qs(query).get("download"):
            extra["Content-Disposition"] = f'attachment; filename="{path.name}"'
        with open(path, "rb") as fh:
            self._send(200, fh.read(), content_type, extra)

    def _speech(self):
        if not self._authorized():
            return self._send_error(401, "Invalid API key", "authentication_error")

        try:
            payload = self._read_json()
        except ValueError as e:
            return self._send_error(400, str(e))

        text = payload.get("input") or payload.get("text") or ""
        if not str(text).strip():
            return self._send_error(400, "Field 'input' is required")

        response_format = str(payload.get("response_format", "mp3")).lower()
        if response_format not in SERVER_FORMATS:
            return self._send_error(
                400,
                f"Unsupported response_format: {response_format}. "
                f"Supported: {', '.join(sorted(SERVER_FORMATS))}",
            )

        lang = str(payload.get("language") or payload.get("lang") or self.default_lang)
        if lang not in SUPPORTED_LANGUAGES:
            return self._send_error(
                400,
                f"Unsupported language: {lang}. "
                f"ZenTTS speaks {', '.join(SUPPORTED_LANGUAGES)}.",
            )

        try:
            speed = float(payload.get("speed", 1.0))
        except (TypeError, ValueError):
            return self._send_error(400, "Field 'speed' must be a number")

        # OpenAI accepts 0.25 to 4.0; the engine handles 0.5 to 2.0.
        clamped = min(max(speed, 0.5), 2.0)

        try:
            voice = resolve_api_voice(payload.get("voice") or self.default_voice, self.engine)
        except ValueError as e:
            return self._send_error(400, str(e))

        if payload.get("stream") and response_format in STREAMABLE_FORMATS:
            return self._stream_speech(str(text), voice, clamped, lang, response_format)

        try:
            samples, rate = synthesize(self.engine, str(text), voice, clamped, lang)
            audio, content_type = encode_audio(samples, rate, response_format)
        except ValueError as e:
            return self._send_error(400, str(e))
        except Exception as e:
            log.exception("Synthesis failed")
            return self._send_error(500, f"Synthesis failed: {e}", "server_error")

        headers = {"X-ZenTTS-Sample-Rate": str(rate)}
        if clamped != speed:
            headers["X-ZenTTS-Speed-Clamped"] = f"{speed} -> {clamped}"

        item = self._store(payload, audio, text, voice, response_format, clamped, lang,
                           len(samples) / rate if rate else 0)
        if item:
            headers["X-ZenTTS-Item"] = item["id"]

        self._send(200, audio, content_type, headers)

    def _stream_speech(self, text, voice, speed, lang, response_format):
        """Send audio in chunks as it is generated, using chunked encoding."""
        _, _, content_type = SERVER_FORMATS[response_format]
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-ZenTTS-Sample-Rate", str(SAMPLE_RATE))
        if self.allow_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        collected = []
        try:
            for piece in chunk_text(text, initial_chunk_size=600) or [text]:
                samples, rate = self.engine.create(
                    piece, voice=voice, speed=speed, lang=lang
                )
                if not len(samples):
                    continue
                collected.append(samples)
                audio, _ = encode_audio(samples, rate, response_format)
                self.wfile.write(f"{len(audio):X}\r\n".encode())
                self.wfile.write(audio)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return self._record(499, "client went away mid-stream")
        except Exception as e:
            log.exception("Streaming failed")
            self.close_connection = True
            return self._record(500, f"stream failed: {e}")

        self._record(200, "streamed")

    def _store(self, payload, audio, text, voice, response_format, speed, lang, seconds):
        """Add the clip to a session, when the request names one."""
        session_id = payload.get("session_id")
        if not session_id:
            return None
        try:
            return add_session_item(
                session_id,
                audio,
                {
                    "text": str(text),
                    "voice": voice if isinstance(voice, str) else "blend",
                    "format": response_format,
                    "speed": speed,
                    "language": lang,
                    "seconds": round(seconds, 2),
                },
            )
        except (ValueError, OSError) as e:
            log.debug(f"Could not store history: {e}")
            return None

    def _voices(self):
        return {
            "object": "list",
            "default": self.default_voice,
            "data": [
                {
                    "id": voice,
                    "object": "voice",
                    "description": voice_label(voice),
                    "language": VOICE_GROUP_LANGUAGE[
                        next(p for p in VOICE_PREFIXES if voice.startswith(p))
                    ],
                }
                for voice in self.engine.get_voices()
            ],
            "openai_aliases": OPENAI_VOICE_ALIASES,
        }

    def _index(self):
        return {
            "name": "ZenTTS",
            "version": __version__,
            "author": __author__,
            "url": __url__,
            "openai_compatible": True,
            "endpoints": {
                "GET /": "web interface",
                "GET /api": "this list of endpoints",
                "GET /health": "liveness check",
                "GET /v1/models": "models this server answers to",
                "GET /v1/voices": "voices and their OpenAI aliases",
                "GET /v1/license": "licence status of this install",
                "GET /v1/logs": "recent requests",
                "GET /v1/sessions": "list sessions",
                "POST /v1/sessions": "create a session",
                "GET /v1/sessions/{id}": "session with its history",
                "PATCH /v1/sessions/{id}": "rename a session",
                "DELETE /v1/sessions/{id}": "delete a session and its audio",
                "GET /v1/sessions/{id}/items/{item}/audio": "download a clip",
                "POST /v1/audio/speech": "generate speech (OpenAI compatible)",
            },
            "speech_request": {
                "input": "the text to speak, any length (required)",
                "voice": f"ZenTTS id, OpenAI name, or a blend (default {self.default_voice})",
                "response_format": sorted(SERVER_FORMATS),
                "speed": "0.5 to 2.0, values outside are clamped",
                "language": list(SUPPORTED_LANGUAGES),
                "stream": f"true to stream {', '.join(sorted(STREAMABLE_FORMATS))} as it is made",
                "session_id": "store the result in this session's history",
                "model": "accepted and ignored, any id works",
            },
            "authentication": "Bearer token required" if self.api_key else "none",
        }


def run_server(
    host=SERVER_HOST,
    port=SERVER_PORT,
    model_path=None,
    voices_path=None,
    api_key=None,
    default_voice=DEFAULT_VOICE,
    lang="en-us",
    cors=True,
    quiet=False,
):
    """Start the web interface and the OpenAI-compatible API."""
    require_license()
    engine = load_engine(model_path, voices_path)

    if default_voice not in engine.get_voices():
        resolved = OPENAI_VOICE_ALIASES.get(default_voice)
        if resolved not in engine.get_voices():
            print(f"Error: unknown default voice: {default_voice}")
            sys.exit(1)
        default_voice = resolved

    handler = type(
        "ConfiguredZenTTSHandler",
        (ZenTTSHandler,),
        {
            "engine": engine,
            "api_key": api_key,
            "default_voice": default_voice,
            "default_lang": lang,
            "allow_cors": cors,
            "quiet": quiet,
        },
    )

    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as e:
        print(f"Error: cannot listen on {host}:{port} ({e})")
        sys.exit(1)

    reachable = "127.0.0.1" if host in ("0.0.0.0", "") else host
    base = f"http://{reachable}:{port}"

    print(f"\nZenTTS {__version__} - speech studio and OpenAI-compatible API")
    print(f"Open {base} in a browser\n")
    print("Endpoints:")
    print(f"  GET    {base}/                                    web interface")
    print(f"  GET    {base}/api                                 endpoint list")
    print(f"  GET    {base}/health                              liveness check")
    print(f"  GET    {base}/v1/models                           models")
    print(f"  GET    {base}/v1/voices                           voices")
    print(f"  GET    {base}/v1/license                          licence status")
    print(f"  GET    {base}/v1/logs                             recent requests")
    print(f"  GET    {base}/v1/sessions                         list sessions")
    print(f"  POST   {base}/v1/sessions                         create a session")
    print(f"  GET    {base}/v1/sessions/<id>                    session history")
    print(f"  PATCH  {base}/v1/sessions/<id>                    rename a session")
    print(f"  DELETE {base}/v1/sessions/<id>                    delete a session")
    print(f"  GET    {base}/v1/sessions/<id>/items/<item>/audio  download a clip")
    print(f"  POST   {base}/v1/audio/speech                     generate speech")
    print(f"\nVoices:   {len(engine.get_voices())} English, default {default_voice}")
    print(f"Formats:  {', '.join(sorted(SERVER_FORMATS))}")
    print(f"Stream:   {', '.join(sorted(STREAMABLE_FORMATS))}")
    print(f"History:  {sessions_dir()}")
    print(f"Auth:     {'Bearer token required' if api_key else 'none (local use)'}")
    print("\nPress Ctrl+C to stop.\n")
    sys.stdout.flush()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        httpd.server_close()


# Options that consume the argument after them.
OPTIONS_WITH_VALUES = {
    "--speed",
    "--lang",
    "--voice",
    "--split-output",
    "--format",
    "--model",
    "--voices",
    "--host",
    "--port",
    "--api-key",
    "--default-voice",
}


def split_positionals(argv, format="wav"):
    """Separate the input files from an optional output file.

    Everything that is not an option or an option's value is positional. If the
    last positional looks like an audio file it is the output, and the rest are
    inputs to be joined together:

        zentts part1.txt part2.txt book.wav   -> 2 inputs, 1 output
        zentts part1.txt part2.txt            -> 2 inputs, output derived
    """
    audio_suffixes = (".wav", ".mp3")

    positionals = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in OPTIONS_WITH_VALUES:
            skip = True
            continue
        if arg.startswith("-") and arg not in ("-",):
            continue
        positionals.append(arg)

    if not positionals:
        return [], None

    last = positionals[-1]
    is_output = last.lower().endswith(audio_suffixes) and (
        len(positionals) > 1 or last.lower().endswith("." + format)
    )
    if is_output and len(positionals) > 1:
        return positionals[:-1], last
    if is_output:
        # A lone audio file is not a valid input, so treat it as the output.
        return [], last
    return positionals, None


def _read_model_options(argv):
    """Pull --model / --voices out of the arguments, if present."""
    model_path = None
    voices_path = None
    for i, arg in enumerate(argv):
        if arg == "--model" and i + 1 < len(argv):
            model_path = argv[i + 1]
        elif arg == "--voices" and i + 1 < len(argv):
            voices_path = argv[i + 1]
    return model_path, voices_path


def _option_value(argv, name, default=None):
    """Read the value that follows an option, if it is there."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
    return default


def start_command(argv):
    """Handle `zentts start`, which runs the OpenAI-compatible API server."""
    if "--help" in argv or "-h" in argv:
        print_server_usage()
        sys.exit(0)

    port = _option_value(argv, "--port", str(SERVER_PORT))
    try:
        port = int(port)
    except ValueError:
        print(f"Error: --port must be a number, got {port}")
        sys.exit(1)

    model_path, voices_path = _read_model_options(argv)
    lang = _option_value(argv, "--lang", "en-us")
    if lang not in SUPPORTED_LANGUAGES:
        print(f"Error: unsupported language: {lang}")
        print(f"ZenTTS speaks {', '.join(SUPPORTED_LANGUAGES)}")
        sys.exit(1)

    run_server(
        host=_option_value(argv, "--host", SERVER_HOST),
        port=port,
        model_path=model_path,
        voices_path=voices_path,
        api_key=_option_value(argv, "--api-key") or os.getenv("ZENTTS_API_KEY"),
        default_voice=_option_value(argv, "--default-voice", DEFAULT_VOICE),
        lang=lang,
        cors="--no-cors" not in argv,
        quiet="--quiet" in argv,
    )


def main():
    """Main entry point for the zentts CLI tool."""
    stdin_indicators = ["/dev/stdin", "-", "CONIN$"]

    if len(sys.argv) > 1 and sys.argv[1] in ("start", "serve"):
        start_command(sys.argv[2:])
        return

    valid_options = get_valid_options()

    unknown_options = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--") and arg not in valid_options:
            unknown_options.append(arg)
        elif arg in OPTIONS_WITH_VALUES:
            i += 1
        i += 1

    if unknown_options:
        print("Error: Unknown option(s):", ", ".join(unknown_options))
        print("\nDid you mean one of these?")
        for unknown in unknown_options:
            similar = difflib.get_close_matches(unknown, valid_options, n=3, cutoff=0.4)
            if similar:
                print(f"  {unknown} -> {', '.join(similar)}")
        print("\n")
        print_usage()
        sys.exit(1)

    if "--version" in sys.argv or "-v" in sys.argv:
        try:
            version = importlib.metadata.version("zentts")
        except importlib.metadata.PackageNotFoundError:
            version = __version__
        print(f"ZenTTS {version}")
        print(f"By {__author__} - {__url__}")
        print(f"Copyright (c) 2026 {__author__}. All rights reserved.")
        sys.exit(0)
    elif "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        sys.exit(0)
    elif "--license" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] == "license"):
        allowed, message = check_license(force=True, quiet=False)
        print(f"\nZenTTS {__version__}")
        print(f"Status:  {'active' if allowed else 'disabled'}")
        print(f"Reason:  {message}\n")
        sys.exit(0 if allowed else 2)
    elif "--download" in sys.argv:
        try:
            model, voices = ensure_models()
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"\nModel files ready:\n  {model}\n  {voices}")
        sys.exit(0)
    elif "--help-languages" in sys.argv:
        print_supported_languages(*_read_model_options(sys.argv))
        sys.exit(0)
    elif "--help-voices" in sys.argv:
        print_supported_voices(*_read_model_options(sys.argv))
        sys.exit(0)

    stream = "--stream" in sys.argv
    speed = 1.0
    lang = "en-us"
    voice = None
    split_output = None
    format = "wav"
    merge_chunks = "--merge-chunks" in sys.argv
    model_path, voices_path = _read_model_options(sys.argv)

    for i, arg in enumerate(sys.argv):
        if arg == "--speed" and i + 1 < len(sys.argv):
            try:
                speed = float(sys.argv[i + 1])
            except ValueError:
                print("Error: Speed must be a number")
                sys.exit(1)
            if not 0.5 <= speed <= 2.0:
                print("Error: Speed must be between 0.5 and 2.0")
                sys.exit(1)
        elif arg == "--lang" and i + 1 < len(sys.argv):
            lang = sys.argv[i + 1]
        elif arg == "--voice" and i + 1 < len(sys.argv):
            voice = sys.argv[i + 1]
        elif arg == "--split-output" and i + 1 < len(sys.argv):
            split_output = sys.argv[i + 1]
        elif arg == "--format" and i + 1 < len(sys.argv):
            format = sys.argv[i + 1].lower()
            if format not in ["wav", "mp3"]:
                print("Error: Format must be either 'wav' or 'mp3'")
                sys.exit(1)

    input_files, output_file = split_positionals(sys.argv[1:], format)

    if merge_chunks:
        if not split_output:
            print(
                "Error: --split-output directory must be specified when using --merge-chunks"
            )
            sys.exit(1)
        merge_chunks_to_chapters(split_output, format)
        sys.exit(0)

    if not input_files:
        print("Error: Input file required for text-to-speech conversion")
        print_usage()
        sys.exit(1)

    for path in input_files:
        if path not in stdin_indicators and not os.access(path, os.R_OK):
            print(
                f"Error: Cannot read from {path}. File may not exist or you may not "
                "have permission to read it."
            )
            sys.exit(1)

    if output_file and not output_file.endswith("." + format):
        print(f"Error: Output file must have .{format} extension.")
        sys.exit(1)

    debug = "--debug" in sys.argv

    convert_text_to_audio(
        input_files,
        output_file,
        voice=voice,
        stream=stream,
        speed=speed,
        lang=lang,
        split_output=split_output,
        format=format,
        debug=debug,
        stdin_indicators=stdin_indicators,
        model_path=model_path,
        voices_path=voices_path,
    )


if __name__ == "__main__":
    main()

