# ZenTTS

English text-to-speech from your terminal. ZenTTS turns `.txt`, `.epub` and
`.pdf` files into `.wav` or `.mp3` audio, with per-chapter splitting, voice
blending and live streaming playback.

- English only — US (`en-us`) and UK (`en-gb`)
- 100% offline after the first run, no API keys, no accounts
- Self-contained engine: the ONNX runtime and the phonemizer are all ZenTTS needs

## Install

```bash
pip install zentts
```

Or from a clone of this repository:

```bash
git clone https://github.com/OmorDeveloper/zentts.git
cd zentts
pip install -e .
```

## Model files

ZenTTS needs two files:

| File | Size | What it is |
| --- | --- | --- |
| `zentts-v1.0.onnx` | ~310 MB | the speech model |
| `zentts-voices-v1.0.bin` | ~14 MB | the 28 English voice packs |

They download automatically the first time you run ZenTTS, with a progress bar
and a checksum check, and are cached for later runs. To fetch them ahead of
time:

```bash
zentts --download
```

The cache lives in `%LOCALAPPDATA%\zentts` on Windows, `~/Library/Caches/zentts`
on macOS and `~/.cache/zentts` on Linux.

| Environment variable | Effect |
| --- | --- |
| `ZENTTS_HOME` | store the model files somewhere else |
| `ZENTTS_NO_DOWNLOAD=1` | never download; use `--model` / `--voices` instead |
| `ONNX_PROVIDER` | force an ONNX Runtime execution provider |
| `LOG_LEVEL=DEBUG` | verbose engine logging |

You can always point at your own copies:

```bash
zentts input.txt out.wav --model ./zentts-v1.0.onnx --voices ./zentts-voices-v1.0.bin
```

## Usage

```bash
zentts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah
zentts book.epub --split-output ./chunks/ --format mp3
zentts paper.pdf output.wav --lang en-gb --voice bf_emma
zentts input.txt --stream --voice "af_sarah:60,am_adam:40"
zentts --merge-chunks --split-output ./chunks/ --format wav
zentts --help-voices
zentts --help-languages
zentts --help
```

### Options

| Option | Description |
| --- | --- |
| `--stream` | play audio as it is generated instead of saving it |
| `--speed <float>` | speech speed, 0.5 to 2.0 (default 1.0) |
| `--lang <str>` | `en-us` or `en-gb` (default `en-us`) |
| `--voice <str>` | a voice name, or two names to blend |
| `--split-output <dir>` | write one file per chunk, grouped by chapter |
| `--format <str>` | `wav` or `mp3` (default `wav`) |
| `--model` / `--voices` | paths to the model files |
| `--download` | fetch the model files and exit |
| `--merge-chunks` | merge a split-output directory back into chapter files |
| `--debug` | detailed progress and error output |

### Voices

Voice names carry their accent and gender in the prefix: `af_` American female,
`am_` American male, `bf_` British female, `bm_` British male. Run
`zentts --help-voices` for the full list in your voice file.

Blend two voices by weight:

```bash
zentts input.txt out.wav --voice "af_sarah:60,am_adam:40"
zentts input.txt out.wav --voice "am_adam,af_sarah"   # 50-50
```

### Long books

`--split-output` writes `chapter_001/chunk_001.wav`, … and skips work that is
already done, so an interrupted run resumes where it stopped. When it finishes,
join everything into one file per chapter:

```bash
zentts --merge-chunks --split-output ./chunks/ --format wav
```

## Use it from Python

```python
from zentts import ZenTTS, resolve_model_files
import soundfile as sf

model, voices = resolve_model_files()   # downloads on first use
engine = ZenTTS(model, voices)

samples, sample_rate = engine.create("Hello from ZenTTS.", voice="af_sarah")
sf.write("hello.wav", samples, sample_rate)
```

## Requirements

Python 3.11 or 3.12. Everything else is installed by pip. `--format mp3` needs
`libsndfile` with MP3 support, which ships with the `soundfile` wheels on
Windows, macOS and Linux.

## License

MIT — see [LICENSE](LICENSE). Third-party components and the model weights are
credited in [NOTICE.md](NOTICE.md).
