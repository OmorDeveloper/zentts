# ZENTTS TTS

A CLI text-to-speech tool using the ZENTTS model, supporting multiple languages, voices (with blending), and various input formats including EPUB books and PDF documents.

![ngpt-s-c](https://raw.githubusercontent.com/nazdridoy/ZENTTS-tts/main/previews/ZENTTS-tts-h.png)

## Features

- Multiple language and voice support
- Voice blending with customizable weights
- EPUB, PDF and TXT file input support
- Standard input (stdin) and `|` piping from other programs
- Streaming audio playback
- Split output into chapters
- Adjustable speech speed
- WAV and MP3 output formats
- Chapter merging capability
- Detailed debug output option
- GPU Support

## Demo

ZENTTS TTS is an open-source CLI tool that delivers high-quality text-to-speech right from your terminal. Think of it as your personal voice studio, capable of transforming any text into natural-sounding speech with minimal effort.

https://github.com/user-attachments/assets/8413e640-59e9-490e-861d-49187e967526

[Demo Audio (MP3)](https://github.com/nazdridoy/ZENTTS-tts/raw/main/previews/demo.mp3) | [Demo Audio (WAV)](https://github.com/nazdridoy/ZENTTS-tts/raw/main/previews/demo.wav)

## TODO

- [x] Add GPU support
- [x] Add PDF support
- [ ] Add GUI

## Prerequisites

- Python 3.11-3.12 (Python 3.13+ is not currently supported)

## Installation

### Method 1: Install from PyPI (Recommended)

The easiest way to install ZENTTS TTS is from PyPI:

```bash
# Using uv (recommended)
uv tool install ZENTTS-tts

# Using pip
pip install ZENTTS-tts
```

After installation, you can run:
```bash
ZENTTS-tts --help
```

### Method 2: Install from Git

Install directly from the repository:

```bash
# Using uv (recommended)
uv tool install git+https://github.com/nazdridoy/ZENTTS-tts

# Using pip
pip install git+https://github.com/nazdridoy/ZENTTS-tts
```

### Method 3: Clone and Install Locally

1. Clone the repository:
```bash
git clone https://github.com/nazdridoy/ZENTTS-tts.git
cd ZENTTS-tts
```

2. Install the package:

**With `uv` (recommended):**
```bash
uv venv
uv pip install -e .
```

**With `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. Run the tool:
```bash
# If using uv
uv run ZENTTS-tts --help

# If using pip with activated venv
ZENTTS-tts --help
```

### Method 4: Run Without Installation

If you prefer to run without installing:

1. Clone the repository:
```bash
git clone https://github.com/nazdridoy/ZENTTS-tts.git
cd ZENTTS-tts
```

2. Install dependencies only:

**With `uv`:**
```bash
uv venv
uv sync
```

**With `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run directly:
```bash
# With uv
uv run -m ZENTTS_tts --help

# With pip (venv activated)
python -m ZENTTS_tts --help
```

### Download Model Files

After installation, download the required model files to your working directory:

```bash
# Download voice data (bin format is preferred)
wget https://github.com/nazdridoy/ZENTTS-tts/releases/download/v1.0.0/voices-v1.0.bin

# Download the model
wget https://github.com/nazdridoy/ZENTTS-tts/releases/download/v1.0.0/ZENTTS-v1.0.onnx
```

> The script requires `voices-v1.0.bin` and `ZENTTS-v1.0.onnx` to be present in the same directory where you run the `ZENTTS-tts` command.

## Supported voices:

| **Category** | **Voices** | **Language Code** |
| --- | --- | --- |
| 🇺🇸 👩 | af\_alloy, af\_aoede, af\_bella, af\_heart, af\_jessica, af\_kore, af\_nicole, af\_nova, af\_river, af\_sarah, af\_sky | **en-us** |
| 🇺🇸 👨 | am\_adam, am\_echo, am\_eric, am\_fenrir, am\_liam, am\_michael, am\_onyx, am\_puck | **en-us** |
| 🇬🇧 | bf\_alice, bf\_emma, bf\_isabella, bf\_lily, bm\_daniel, bm\_fable, bm\_george, bm\_lewis | **en-gb** |
| 🇫🇷 | ff\_siwis | **fr-fr** |
| 🇮🇹 | if\_sara, im\_nicola | **it** |
| 🇯🇵 | jf\_alpha, jf\_gongitsune, jf\_nezumi, jf\_tebukuro, jm\_kumo | **ja** |
| 🇨🇳 | zf\_xiaobei, zf\_xiaoni, zf\_xiaoxiao, zf\_xiaoyi, zm\_yunjian, zm\_yunxi, zm\_yunxia, zm\_yunyang | **cmn** |

## Usage

### Basic Usage

```bash
ZENTTS-tts <input_text_file> [<output_audio_file>] [options]
```

> [!NOTE]
> - If you installed via Method 1 (PyPI) or Method 2 (git install), use `ZENTTS-tts` directly
> - If you installed via Method 3 (local install), use `uv run ZENTTS-tts` or activate your virtual environment first
> - If you're using Method 4 (no install), use `uv run -m ZENTTS_tts` or `python -m ZENTTS_tts` with activated venv

### Commands

- `-h, --help`: Show help message
- `--help-languages`: List supported languages
- `--help-voices`: List available voices
- `--merge-chunks`: Merge existing chunks into chapter files

### Options

- `--stream`: Stream audio instead of saving to file
- `--speed <float>`: Set speech speed (default: 1.0)
- `--lang <str>`: Set language (default: en-us)
- `--voice <str>`: Set voice or blend voices (default: interactive selection)
  - Single voice: Use voice name (e.g., "af_sarah")
  - Blended voices: Use "voice1:weight,voice2:weight" format
- `--split-output <dir>`: Save each chunk as separate file in directory
- `--format <str>`: Audio format: wav or mp3 (default: wav)
- `--debug`: Show detailed debug information during processing

### Input Formats

- `.txt`: Text file input
- `.epub`: EPUB book input (will process chapters)
- `.pdf`: PDF document input (extracts chapters from TOC or content)
- `-` or `/dev/stdin` (Linux/macOS) or `CONIN$` (Windows): Standard input (stdin)

### Examples

```bash
# Basic usage with output file
ZENTTS-tts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah

# Read from standard input (stdin)
echo "Hello World" | ZENTTS-tts - --stream
cat input.txt | ZENTTS-tts - output.wav

# Cross-platform stdin support:
# Linux/macOS: echo "text" | ZENTTS-tts - --stream
# Windows: echo "text" | ZENTTS-tts - --stream
# All platforms also support: ZENTTS-tts /dev/stdin --stream (Linux/macOS) or ZENTTS-tts CONIN$ --stream (Windows)

# Use voice blending (60-40 mix)
ZENTTS-tts input.txt output.wav --voice "af_sarah:60,am_adam:40"

# Use equal voice blend (50-50)
ZENTTS-tts input.txt --stream --voice "am_adam,af_sarah"

# Process EPUB and split into chunks
ZENTTS-tts input.epub --split-output ./chunks/ --format mp3

# Stream audio directly
ZENTTS-tts input.txt --stream --speed 0.8

# Merge existing chunks
ZENTTS-tts --merge-chunks --split-output ./chunks/ --format wav

# Process EPUB with detailed debug output
ZENTTS-tts input.epub --split-output ./chunks/ --debug

# Process PDF and split into chapters
ZENTTS-tts input.pdf --split-output ./chunks/ --format mp3

# List available voices
ZENTTS-tts --help-voices

# List supported languages
ZENTTS-tts --help-languages
```

> [!TIP]
> If you're using Method 3, replace `ZENTTS-tts` with `uv run ZENTTS-tts` in the examples above.
> If you're using Method 4, replace `ZENTTS-tts` with `uv run -m ZENTTS_tts` or `python -m ZENTTS_tts` in the examples above.

## Features in Detail

### EPUB Processing
- Automatically extracts chapters from EPUB files
- Preserves chapter titles and structure
- Creates organized output for each chapter
- Detailed debug output available for troubleshooting

### Audio Processing
- Chunks long text into manageable segments
- Supports streaming for immediate playback
- Voice blending with customizable mix ratios
- Progress indicators for long processes
- Handles interruptions gracefully

### Output Options
- Single file output
- Split output with chapter organization
- Chunk merging capability
- Multiple audio format support

### Debug Mode
- Shows detailed information about file processing
- Displays NCX parsing details for EPUB files
- Lists all found chapters and their metadata
- Helps troubleshoot processing issues

### Input Options
- Text file input (.txt)
- EPUB book input (.epub)
- Standard input (stdin)
- Supports piping from other programs

## Contributing

This is a personal project. But if you want to contribute, please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [ZENTTS-ONNX](https://github.com/thewh1teagle/ZENTTS-onnx)
