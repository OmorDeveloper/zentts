# ZenTTS

By [Omor](https://www.linkedin.com/in/omardeveloper/)

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
zentts input.txt output.wav --speed 1.2 --lang en-us --voice zen_us_f10
zentts part1.txt part2.txt part3.txt book.wav
zentts intro.txt chapters.epub notes.pdf audiobook.mp3 --format mp3
zentts book.epub --split-output ./chunks/ --format mp3
zentts paper.pdf output.wav --lang en-gb --voice zen_uk_f02
zentts input.txt --stream --voice "zen_us_f10:60,zen_us_m01:40"
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

There are 28 ZenTTS voices. A voice id reads `zen_<region>_<gender><number>`:

| Prefix | Voices | Use with |
| --- | --- | --- |
| `zen_us_f` | `zen_us_f01` … `zen_us_f11` | `--lang en-us` |
| `zen_us_m` | `zen_us_m01` … `zen_us_m09` | `--lang en-us` |
| `zen_uk_f` | `zen_uk_f01` … `zen_uk_f04` | `--lang en-gb` |
| `zen_uk_m` | `zen_uk_m01` … `zen_uk_m04` | `--lang en-gb` |

Run `zentts --help-voices` to list them grouped by region and gender.

Blend two voices by weight:

```bash
zentts input.txt out.wav --voice "zen_us_f10:60,zen_us_m01:40"
zentts input.txt out.wav --voice "zen_us_m01,zen_us_f10"   # 50-50
```

## Multiple input files

Pass as many inputs as you like and they are joined, in order, into one output
file. Formats can be mixed. If the last argument ends in `.wav` or `.mp3` it is
the output; otherwise the name is taken from the first input.

```bash
zentts ch1.txt ch2.txt ch3.txt book.wav
zentts intro.txt body.epub appendix.pdf audiobook.mp3 --format mp3
zentts ch1.txt ch2.txt --stream          # plays them back to back
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

samples, sample_rate = engine.create("Hello from ZenTTS.", voice="zen_us_f10")
sf.write("hello.wav", samples, sample_rate)
```

## Requirements

Python 3.11 or 3.12. Everything else is installed by pip. `--format mp3` needs
`libsndfile` with MP3 support, which ships with the `soundfile` wheels on
Windows, macOS and Linux.

Writing audio to a file works anywhere. `--stream` plays through your speakers,
so it additionally needs the PortAudio system library — already present on
Windows and macOS, and installed on Debian or Ubuntu with
`sudo apt install libportaudio2`.

## License

ZenTTS is proprietary software, © 2026 Omor, all rights reserved — see
[LICENSE](LICENSE). You may use it, including commercially, but not
redistribute or modify it without permission.

Third-party components and the model weights it builds on keep their own
licences, reproduced in [NOTICE.md](NOTICE.md).

## Author

Omor — [LinkedIn](https://www.linkedin.com/in/omardeveloper/) ·
[GitHub](https://github.com/OmorDeveloper)
