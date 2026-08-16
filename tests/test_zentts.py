"""Tests for ZenTTS.

The tests that need the ONNX model are marked `model` and skip themselves when
the model files are not on the machine, so the rest of the suite still runs on
a clean checkout.
"""

import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zentts


##############################################################################
# Argument parsing
##############################################################################


@pytest.mark.parametrize(
    "argv, format, expected",
    [
        (["in.txt"], "wav", (["in.txt"], None)),
        (["in.txt", "out.wav"], "wav", (["in.txt"], "out.wav")),
        (["a.txt", "b.txt", "book.wav"], "wav", (["a.txt", "b.txt"], "book.wav")),
        (["a.txt", "b.txt"], "wav", (["a.txt", "b.txt"], None)),
        (["a.txt", "b.epub", "c.pdf", "all.mp3"], "mp3", (["a.txt", "b.epub", "c.pdf"], "all.mp3")),
        (["in.txt", "--stream", "--voice", "zen_us_f10"], "wav", (["in.txt"], None)),
        (["in.txt", "out.mp3", "--format", "mp3"], "mp3", (["in.txt"], "out.mp3")),
        (["--split-output", "./chunks/", "a.txt"], "wav", (["a.txt"], None)),
        (["out.wav"], "wav", ([], "out.wav")),
        (["-", "out.wav"], "wav", (["-"], "out.wav")),
        ([], "wav", ([], None)),
    ],
)
def test_split_positionals(argv, format, expected):
    assert zentts.split_positionals(argv, format) == expected


def test_split_positionals_ignores_option_values():
    """A value that looks like a file must not be mistaken for an input."""
    argv = ["--model", "custom.onnx", "--voices", "custom.bin", "book.txt", "out.wav"]
    assert zentts.split_positionals(argv) == (["book.txt"], "out.wav")


def test_read_model_options():
    argv = ["zentts", "in.txt", "--model", "m.onnx", "--voices", "v.bin"]
    assert zentts._read_model_options(argv) == ("m.onnx", "v.bin")
    assert zentts._read_model_options(["zentts", "in.txt"]) == (None, None)


##############################################################################
# Text chunking
##############################################################################


def test_chunk_text_splits_on_sentences():
    """Sentences are packed together until the next one would overflow."""
    text = "One. Two. Three."
    assert zentts.chunk_text(text, initial_chunk_size=10) == ["One. Two.", "Three."]
    assert zentts.chunk_text(text, initial_chunk_size=5) == ["One.", "Two.", "Three."]


def test_chunk_text_keeps_short_text_together():
    text = "One. Two. Three."
    assert zentts.chunk_text(text, initial_chunk_size=1000) == ["One. Two. Three."]


def test_chunk_text_keeps_order_around_a_long_sentence():
    """A sentence too long to fit must not jump ahead of queued text."""
    long_sentence = " ".join(["word"] * 60) + "."
    text = f"Short one. {long_sentence} Short two."
    chunks = zentts.chunk_text(text, initial_chunk_size=60)
    assert chunks[0] == "Short one."
    assert chunks[-1] == "Short two."
    assert "".join(chunks).count("word") == 60


def test_chunk_text_does_not_double_the_full_stop():
    chunks = zentts.chunk_text("One. " + " ".join(["word"] * 30) + ".", 40)
    assert not any(chunk.endswith("..") for chunk in chunks)


def test_chunk_text_breaks_up_a_long_sentence():
    sentence = " ".join(["word"] * 200) + "."
    chunks = zentts.chunk_text(sentence, initial_chunk_size=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 105 for chunk in chunks)
    assert "word" in chunks[0]


def test_chunk_text_ignores_blank_input():
    assert zentts.chunk_text("   \n  ") == []


##############################################################################
# Silence trimming
##############################################################################


def test_trim_silence_removes_padding():
    tone = (0.5 * np.sin(np.linspace(0, 600, 40000))).astype(np.float32)
    padded = np.concatenate([np.zeros(9000, np.float32), tone, np.zeros(12000, np.float32)])
    trimmed = zentts.trim_silence(padded)
    assert len(trimmed) < len(padded)
    assert len(trimmed) >= len(tone) - 2048


def test_trim_silence_handles_empty_and_silent_input():
    assert zentts.trim_silence(np.zeros(0, np.float32)).size == 0
    silence = np.zeros(5000, np.float32)
    assert len(zentts.trim_silence(silence)) == len(silence)


def test_trim_silence_keeps_audio_without_padding():
    tone = (0.5 * np.sin(np.linspace(0, 600, 40000))).astype(np.float32)
    assert len(zentts.trim_silence(tone)) == pytest.approx(len(tone), abs=2048)


##############################################################################
# Text normalisation
##############################################################################


@pytest.mark.parametrize(
    "raw, expected_fragment",
    [
        ("$4.50", "4 dollars and 50 cents"),
        ("£3", "3 pounds"),
        ("Dr. Smith", "Doctor Smith"),
        ("Mr. Smith", "Mister Smith"),
        ("at 3:30", "3 30"),
        ("at 4:00", "4 o'clock"),
        ("in 1995", "19 95"),
        ("yeah", "ye'a"),
    ],
)
def test_normalize_text(raw, expected_fragment):
    assert expected_fragment in zentts.Tokenizer.normalize_text(raw)


def test_normalize_text_collapses_whitespace():
    assert zentts.Tokenizer.normalize_text("a  b\n\n\nc") == "a b\nc"


##############################################################################
# Voices
##############################################################################


def test_filter_english_voices_keeps_only_zentts_ids():
    voices = ["zen_us_f01", "af_sarah", "zen_uk_m04", "jf_alpha", "zen_us_m09"]
    assert zentts._filter_english_voices(voices) == [
        "zen_uk_m04",
        "zen_us_f01",
        "zen_us_m09",
    ]


def test_filter_english_languages():
    assert zentts._filter_english_languages(["en-gb", "fr-fr", "en-us", "ja"]) == [
        "en-gb",
        "en-us",
    ]


@pytest.mark.parametrize(
    "voice, label",
    [
        ("zen_us_f01", "US English, female"),
        ("zen_us_m09", "US English, male"),
        ("zen_uk_f04", "UK English, female"),
        ("zen_uk_m01", "UK English, male"),
    ],
)
def test_voice_label(voice, label):
    assert zentts.voice_label(voice) == label


def test_voice_groups_cover_every_prefix():
    assert set(zentts.VOICE_GROUPS) == set(zentts.VOICE_PREFIXES)
    assert set(zentts.VOICE_GROUP_LANGUAGE) == set(zentts.VOICE_PREFIXES)
    assert set(zentts.VOICE_GROUP_LANGUAGE.values()) <= set(zentts.SUPPORTED_LANGUAGES)


def test_default_voice_is_a_real_voice_id():
    assert zentts.DEFAULT_VOICE.startswith(zentts.VOICE_PREFIXES)


##############################################################################
# Reading input files
##############################################################################


def test_load_chapters_titles_a_text_file_after_its_name(tmp_path):
    path = tmp_path / "intro.txt"
    path.write_text("Hello there.", encoding="utf-8")

    chapters = zentts.load_chapters(str(path))

    assert len(chapters) == 1
    assert chapters[0]["title"] == "intro"
    assert chapters[0]["content"] == "Hello there."


def test_load_chapters_skips_an_empty_file(tmp_path, capsys):
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    assert zentts.load_chapters(str(path)) == []
    assert "empty" in capsys.readouterr().out


def test_load_chapters_reads_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("piped text"))
    chapters = zentts.load_chapters("-", stdin_indicators=["-"])
    assert chapters[0]["content"] == "piped text"


##############################################################################
# Model files
##############################################################################


def test_model_dir_follows_zentts_home(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENTTS_HOME", str(tmp_path / "models"))
    assert zentts.model_dir() == tmp_path / "models"


def test_model_dir_default_is_per_user(monkeypatch):
    monkeypatch.delenv("ZENTTS_HOME", raising=False)
    assert zentts.model_dir().name == "zentts"


def test_resolve_file_prefers_an_explicit_path(tmp_path):
    path = tmp_path / "custom.onnx"
    path.write_bytes(b"x")
    assert zentts.resolve_file(zentts.MODEL_FILENAME, str(path)) == path


def test_resolve_file_rejects_a_missing_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        zentts.resolve_file(zentts.MODEL_FILENAME, str(tmp_path / "nope.onnx"))


def test_resolve_file_without_download_explains_itself(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENTTS_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        zentts.resolve_file(zentts.MODEL_FILENAME, allow_download=False)

    message = str(excinfo.value)
    assert zentts.MODEL_FILENAME in message
    assert "--model" in message


def test_resolve_model_files_honours_the_no_download_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENTTS_HOME", str(tmp_path))
    monkeypatch.setenv("ZENTTS_NO_DOWNLOAD", "1")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        zentts.resolve_model_files()


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    payload = b"zentts" * 1000
    path.write_bytes(payload)
    assert zentts._sha256(path) == hashlib.sha256(payload).hexdigest()


def test_checksums_are_recorded_for_both_model_files():
    assert set(zentts.CHECKSUMS) == {zentts.MODEL_FILENAME, zentts.VOICES_FILENAME}
    assert all(len(digest) == 64 for digest in zentts.CHECKSUMS.values())


def test_download_urls_point_at_the_release():
    for url in zentts.DOWNLOAD_URLS.values():
        assert url.startswith(zentts.RELEASE_BASE_URL)


class _FakeResponse:
    """Minimal stand-in for the object urlopen returns."""

    def __init__(self, payload):
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size):
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_file_writes_and_verifies(monkeypatch, tmp_path):
    payload = b"pretend model bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(zentts.CHECKSUMS, "fake.bin", digest)
    monkeypatch.setattr(
        zentts.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload)
    )

    target = tmp_path / "fake.bin"
    assert zentts.download_file("https://example.invalid/fake.bin", target, quiet=True) == target
    assert target.read_bytes() == payload


def test_download_file_rejects_a_bad_checksum(monkeypatch, tmp_path):
    monkeypatch.setitem(zentts.CHECKSUMS, "fake.bin", "0" * 64)
    monkeypatch.setattr(
        zentts.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"wrong")
    )

    target = tmp_path / "fake.bin"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        zentts.download_file("https://example.invalid/fake.bin", target, quiet=True)

    assert not target.exists()
    assert not target.with_suffix(".bin.part").exists()


##############################################################################
# Merging chunks back into chapters
##############################################################################


def test_merge_chunks_joins_a_chapter(tmp_path, capsys):
    chapter = tmp_path / "chapter_001"
    chapter.mkdir()
    (chapter / "info.txt").write_text("Title: Opening\n", encoding="utf-8")

    tone = np.linspace(-0.2, 0.2, 2400, dtype=np.float32)
    for index in (1, 2):
        sf.write(chapter / f"chunk_{index:03d}.wav", tone, 24000)

    zentts.merge_chunks_to_chapters(str(tmp_path), "wav")

    merged = tmp_path / "Opening.wav"
    assert merged.exists()
    data, rate = sf.read(merged)
    assert rate == 24000
    assert len(data) == 2 * len(tone)


def test_merge_chunks_reports_a_missing_directory(tmp_path, capsys):
    zentts.merge_chunks_to_chapters(str(tmp_path / "absent"), "wav")
    assert "does not exist" in capsys.readouterr().out


##############################################################################
# PDF helpers
##############################################################################


def test_pdf_parser_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        zentts.PdfParser(str(tmp_path / "missing.pdf"))


def test_clean_title_drops_zero_width_spaces(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    parser = zentts.PdfParser(str(pdf))
    assert parser._clean_title(" Chapter​One ") == "Chapter One"


##############################################################################
# Playback
##############################################################################


def test_importing_zentts_does_not_need_portaudio():
    """Only --stream needs PortAudio, so the module must import without it."""
    assert "sounddevice" not in sys.modules or True
    source = (Path(__file__).resolve().parent.parent / "zentts.py").read_text(
        encoding="utf-8"
    )
    import_block = source[: source.index("warnings.filterwarnings")]
    assert "import sounddevice" not in import_block


def test_load_sounddevice_explains_a_missing_portaudio(monkeypatch, capsys):
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit):
        zentts._load_sounddevice()

    assert "libportaudio2" in capsys.readouterr().out


##############################################################################
# Engine (needs the model files)
##############################################################################


@pytest.fixture(scope="module")
def engine():
    try:
        model, voices = zentts.resolve_model_files(allow_download=False)
    except FileNotFoundError:
        pytest.skip("model files not available")
    return zentts.ZenTTS(model, voices)


@pytest.mark.model
def test_engine_exposes_the_english_voices(engine):
    voices = engine.get_voices()
    assert len(voices) == 28
    assert all(v.startswith(zentts.VOICE_PREFIXES) for v in voices)
    assert engine.get_languages() == ["en-us", "en-gb"]


@pytest.mark.model
def test_engine_creates_audio(engine):
    samples, rate = engine.create("Testing one two three.", voice=zentts.DEFAULT_VOICE)
    assert rate == zentts.SAMPLE_RATE
    assert len(samples) > rate * 0.5
    assert np.abs(samples).max() > 0.01


@pytest.mark.model
def test_engine_blends_two_voices(engine):
    first = engine.get_voice_style("zen_us_f10")
    second = engine.get_voice_style("zen_us_m01")
    blend = np.add(first * 0.6, second * 0.4)
    samples, _ = engine.create("A blended voice.", voice=blend)
    assert len(samples) > 0


@pytest.mark.model
def test_engine_rejects_bad_arguments(engine):
    with pytest.raises(ValueError, match="Voice"):
        engine.get_voice_style("af_sarah")
    with pytest.raises(ValueError, match="Language"):
        engine.create("x", voice=zentts.DEFAULT_VOICE, lang="fr-fr")
    with pytest.raises(ValueError, match="Speed"):
        engine.create("x", voice=zentts.DEFAULT_VOICE, speed=3.0)


@pytest.mark.model
def test_engine_streams_chunks(engine):
    import asyncio

    async def collect():
        return [
            chunk
            async for chunk, _ in engine.create_stream(
                "First sentence. Second sentence.", voice=zentts.DEFAULT_VOICE
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks and all(len(c) > 0 for c in chunks)
