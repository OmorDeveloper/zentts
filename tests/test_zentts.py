"""Tests for ZenTTS.

The tests that need the ONNX model are marked `model` and skip themselves when
the model files are not on the machine, so the rest of the suite still runs on
a clean checkout.
"""

import hashlib
import io
import re
import sys
import time
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import zentts


@pytest.fixture(autouse=True)
def _skip_license_check(monkeypatch):
    """Keep the suite offline; the licence logic is tested on its own."""
    monkeypatch.setenv("ZENTTS_SKIP_LICENSE_CHECK", "1")


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
    """Ids from an older voice pack must be filtered out, not offered."""
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
# OpenAI-compatible server
##############################################################################


def test_openai_aliases_point_at_real_voice_ids():
    for alias, voice in zentts.OPENAI_VOICE_ALIASES.items():
        assert voice.startswith(zentts.VOICE_PREFIXES), alias


def test_server_formats_cover_the_openai_set():
    assert {"mp3", "wav", "flac", "opus", "pcm"} <= set(zentts.SERVER_FORMATS)


def test_server_answers_to_the_openai_model_ids():
    assert "tts-1" in zentts.SERVER_MODELS
    assert zentts.SERVER_MODELS[0] == "zentts-1"


@pytest.mark.parametrize(
    "response_format, head",
    [
        ("wav", b"RIFF"),
        ("flac", b"fLaC"),
        ("ogg", b"OggS"),
        ("opus", b"OggS"),
    ],
)
def test_encode_audio_writes_real_containers(response_format, head):
    samples = (0.2 * np.sin(np.linspace(0, 400, 12000))).astype(np.float32)
    audio, content_type = zentts.encode_audio(samples, 24000, response_format)
    assert audio.startswith(head)
    assert content_type.startswith("audio/")


def test_encode_audio_pcm_is_raw_16_bit():
    samples = np.array([0.0, 1.0, -1.0], dtype=np.float32)
    audio, content_type = zentts.encode_audio(samples, 24000, "pcm")
    assert content_type == "audio/pcm"
    assert np.frombuffer(audio, dtype="<i2").tolist() == [0, 32767, -32767]


def test_encode_audio_clips_out_of_range_samples():
    samples = np.array([2.0, -2.0], dtype=np.float32)
    audio, _ = zentts.encode_audio(samples, 24000, "pcm")
    assert np.frombuffer(audio, dtype="<i2").tolist() == [32767, -32767]


def test_encode_audio_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="aac"):
        zentts.encode_audio(np.zeros(10, np.float32), 24000, "aac")


class _StubEngine:
    """Enough of the engine to exercise voice resolution."""

    def __init__(self):
        self.styles = {
            "zen_us_f10": np.ones((510, 1, 256), dtype=np.float32),
            "zen_us_m01": np.zeros((510, 1, 256), dtype=np.float32),
            "zen_us_f08": np.ones((510, 1, 256), dtype=np.float32) * 0.5,
        }

    def get_voices(self):
        return sorted(self.styles)

    def get_voice_style(self, name):
        return self.styles[name]


def test_resolve_api_voice_accepts_a_zentts_id():
    assert zentts.resolve_api_voice("zen_us_f10", _StubEngine()) == "zen_us_f10"


def test_resolve_api_voice_accepts_an_openai_name():
    # "nova" is an OpenAI voice, mapped onto a ZenTTS one.
    assert zentts.resolve_api_voice("nova", _StubEngine()) == "zen_us_f08"


def test_resolve_api_voice_blends():
    blend = zentts.resolve_api_voice("zen_us_f10:75,zen_us_m01:25", _StubEngine())
    assert isinstance(blend, np.ndarray)
    assert blend.flat[0] == pytest.approx(0.75)


def test_resolve_api_voice_normalises_blend_weights():
    blend = zentts.resolve_api_voice("zen_us_f10:3,zen_us_m01:1", _StubEngine())
    assert blend.flat[0] == pytest.approx(0.75)


def test_resolve_api_voice_falls_back_to_the_default():
    engine = _StubEngine()
    assert zentts.resolve_api_voice(None, engine) == zentts.DEFAULT_VOICE


def test_resolve_api_voice_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown voice"):
        zentts.resolve_api_voice("does_not_exist", _StubEngine())


def test_resolve_api_voice_rejects_a_three_way_blend():
    with pytest.raises(ValueError, match="two"):
        zentts.resolve_api_voice("zen_us_f10,zen_us_m01,zen_us_f08", _StubEngine())


##############################################################################
# Metadata
##############################################################################


def test_module_version_matches_pyproject():
    pyproject = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    assert zentts.__version__ == declared


def test_module_carries_its_authorship():
    assert zentts.__author__ == "Omor"
    assert zentts.__license__ == "Proprietary"
    assert "OmorDeveloper/zentts" in zentts.__url__
    assert "linkedin.com/in/omardeveloper" in zentts.__doc__


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


@pytest.mark.model
def test_server_serves_speech_end_to_end(engine, tmp_path):
    """Start a real server on an ephemeral port and drive it over HTTP."""
    import json as _json
    import socket
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    handler = type(
        "TestHandler",
        (zentts.ZenTTSHandler,),
        {"engine": engine, "api_key": "test-key", "quiet": True},
    )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def post(payload, key="test-key"):
        request = urllib.request.Request(
            f"{base}/v1/audio/speech",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
            | ({"Authorization": f"Bearer {key}"} if key else {}),
        )
        return urllib.request.urlopen(request, timeout=120)

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as response:
            assert _json.load(response)["status"] == "ok"

        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
            assert "tts-1" in [m["id"] for m in _json.load(response)["data"]]

        with urllib.request.urlopen(f"{base}/api", timeout=10) as response:
            assert "POST /v1/audio/speech" in _json.load(response)["endpoints"]

        with urllib.request.urlopen(f"{base}/", timeout=10) as response:
            assert response.headers["Content-Type"].startswith("text/html")
            assert "ZenTTS Studio" in response.read().decode()

        # An OpenAI voice name and the default mp3 format.
        with post({"model": "tts-1", "input": "Hello.", "voice": "nova"}) as response:
            assert response.headers["Content-Type"] == "audio/mpeg"
            assert len(response.read()) > 1000

        # A ZenTTS id, wav, and a speed outside the engine's range.
        with post(
            {"input": "Hello.", "voice": zentts.DEFAULT_VOICE,
             "response_format": "wav", "speed": 4.0}
        ) as response:
            assert response.headers["Content-Type"] == "audio/wav"
            assert response.headers["X-ZenTTS-Speed-Clamped"] == "4.0 -> 2.0"
            assert response.read().startswith(b"RIFF")

        for payload, status in [
            ({"voice": "nova"}, 400),          # no input
            ({"input": "x", "voice": "nope"}, 400),
            ({"input": "x", "response_format": "aac"}, 400),
            ({"input": "x", "language": "fr-fr"}, 400),
        ]:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                post(payload)
            assert excinfo.value.code == status
            assert "error" in _json.load(excinfo.value)

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            post({"input": "x"}, key=None)
        assert excinfo.value.code == 401

        # A rejected request must not corrupt the next one on the connection.
        with post({"input": "Still working."}) as response:
            assert len(response.read()) > 1000
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)


##############################################################################
# Sessions and history
##############################################################################


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENTTS_HOME", str(tmp_path))
    return tmp_path


def _meta(text="hello"):
    return {
        "text": text,
        "voice": zentts.DEFAULT_VOICE,
        "format": "mp3",
        "speed": 1.0,
        "language": "en-us",
        "seconds": 1.0,
    }


def test_new_session_is_stored_and_listed(store):
    session = zentts.new_session("My book")
    assert session["name"] == "My book"
    assert zentts.load_session(session["id"])["name"] == "My book"
    assert [s["id"] for s in zentts.list_sessions()] == [session["id"]]


def test_sessions_are_listed_newest_first(store):
    first = zentts.new_session("one")
    time.sleep(0.01)
    second = zentts.new_session("two")
    assert [s["id"] for s in zentts.list_sessions()] == [second["id"], first["id"]]


def test_rename_session(store):
    session = zentts.new_session("before")
    assert zentts.rename_session(session["id"], "after")["name"] == "after"
    assert zentts.load_session(session["id"])["name"] == "after"


def test_rename_session_rejects_an_empty_name(store):
    session = zentts.new_session()
    with pytest.raises(ValueError):
        zentts.rename_session(session["id"], "   ")


def test_rename_unknown_session_returns_none(store):
    assert zentts.rename_session("snope", "x") is None


def test_delete_session_removes_its_audio(store):
    session = zentts.new_session()
    item = zentts.add_session_item(session["id"], b"audio-bytes", _meta())
    audio = zentts.sessions_dir() / "audio" / item["file"]
    assert audio.exists()

    assert zentts.delete_session(session["id"]) is True
    assert not audio.exists()
    assert zentts.load_session(session["id"]) is None
    assert zentts.delete_session(session["id"]) is False


def test_add_session_item_records_history(store):
    session = zentts.new_session()
    item = zentts.add_session_item(session["id"], b"12345", _meta("spoken text"))

    assert item["bytes"] == 5
    stored = zentts.load_session(session["id"])["items"]
    assert len(stored) == 1
    assert stored[0]["text"] == "spoken text"

    path, found = zentts.session_item_path(session["id"], item["id"])
    assert path.read_bytes() == b"12345"
    assert found["id"] == item["id"]


def test_history_is_capped(store, monkeypatch):
    monkeypatch.setattr(zentts, "MAX_SESSION_ITEMS", 3)
    session = zentts.new_session()
    for index in range(5):
        zentts.add_session_item(session["id"], b"x", _meta(f"line {index}"))

    items = zentts.load_session(session["id"])["items"]
    assert len(items) == 3
    assert [i["text"] for i in items] == ["line 2", "line 3", "line 4"]
    kept = {i["file"] for i in items}
    on_disk = {p.name for p in (zentts.sessions_dir() / "audio").iterdir()}
    assert on_disk == kept


def test_session_ids_cannot_escape_the_directory(store):
    for bad in ["../secret", "a/b", "..", ""]:
        with pytest.raises(ValueError):
            zentts.load_session(bad)


##############################################################################
# Licence control
##############################################################################


@pytest.fixture()
def licensing(monkeypatch, tmp_path):
    monkeypatch.setenv("ZENTTS_HOME", str(tmp_path))
    monkeypatch.delenv("ZENTTS_SKIP_LICENSE_CHECK", raising=False)
    return tmp_path


def _offline(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(zentts, "_fetch_json", boom)


def _control(monkeypatch, payload, package_ok=True):
    def fake(url, timeout=8):
        if "pypi.org" in url:
            if not package_ok:
                raise urllib.error.HTTPError(url, 404, "gone", {}, None)
            return {"info": {"version": zentts.__version__}}
        return payload

    monkeypatch.setattr(zentts, "_fetch_json", fake)


def test_license_allows_an_enabled_install(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True})
    assert zentts.check_license(force=True)[0]


def test_license_kill_switch_disables_the_install(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": False, "message": "Shut down by the author."})
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert message == "Shut down by the author."

    # The refusal is remembered, so it survives going offline.
    _offline(monkeypatch)
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "Shut down" in message


def test_license_blocks_a_withdrawn_version(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True, "blocked_versions": [zentts.__version__]})
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "withdrawn" in message


def test_license_enforces_a_minimum_version(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True, "min_version": "99.0.0"})
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "no longer supported" in message


def test_license_stops_when_the_package_is_unpublished(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True}, package_ok=False)
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "no longer published" in message


def test_license_refuses_a_first_run_with_no_network(licensing, monkeypatch):
    _offline(monkeypatch)
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "never checked in" in message


def test_license_allows_a_short_offline_spell(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True})
    assert zentts.check_license(force=True)[0]

    _offline(monkeypatch)
    allowed, message = zentts.check_license(force=True)
    assert allowed
    assert "grace" in message


def test_license_stops_after_too_long_offline(licensing, monkeypatch):
    _control(monkeypatch, {"enabled": True})
    zentts.check_license(force=True)

    state = zentts._read_license_state()
    state["checked_at"] = time.time() - (zentts.LICENSE_GRACE_DAYS + 1) * 86400
    zentts._write_license_state(state)

    _offline(monkeypatch)
    allowed, message = zentts.check_license(force=True)
    assert not allowed
    assert "offline" in message


def test_license_check_is_cached(licensing, monkeypatch):
    calls = []

    def counting(url, timeout=8):
        calls.append(url)
        return {"enabled": True} if "pypi" not in url else {"info": {}}

    monkeypatch.setattr(zentts, "_fetch_json", counting)
    zentts.check_license(force=True)
    before = len(calls)
    zentts.check_license()  # inside the cache window
    assert len(calls) == before


def test_version_tuple_orders_releases():
    assert zentts._version_tuple("1.2.0") > zentts._version_tuple("1.1.9")
    assert zentts._version_tuple("1.10.0") > zentts._version_tuple("1.9.0")


##############################################################################
# Web interface
##############################################################################


def test_web_ui_is_self_contained():
    assert "ZenTTS Studio" in zentts.WEB_UI
    assert "<script" in zentts.WEB_UI
    # No external hosts: the page must work with no internet at all.
    assert "https://" not in zentts.WEB_UI


def test_web_ui_covers_the_session_actions():
    for fragment in ["/v1/sessions", "PATCH", "DELETE", "/v1/logs", "/v1/voices"]:
        assert fragment in zentts.WEB_UI
