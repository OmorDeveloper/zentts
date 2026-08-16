# Contributing to ZenTTS

Thanks for your interest in ZenTTS. This document covers how to set up the
project and what a good contribution looks like.

## Development setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/zentts.git`
3. Enter the directory: `cd zentts`
4. Create a virtual environment: `python -m venv .venv`
5. Activate it:
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
6. Install the project: `pip install -e .` (or `uv sync`)
7. Fetch the model files: `zentts --download`

## Code structure

Everything lives in one module, `zentts.py`, in these sections:

| Section | Purpose |
| --- | --- |
| Configuration | languages, voice ids, vocabulary, file names, release URLs |
| Logging | logger, controlled by `LOG_LEVEL` |
| Tokenizer | text normalisation, phonemization, token encoding |
| Engine | the `ZenTTS` class: ONNX inference and streaming |
| Model files | finding, downloading and verifying the model files |
| Command-line interface | argument parsing, EPUB/PDF reading, chunking, output |

## Code style

- Follow PEP 8
- Use descriptive names and add docstrings to functions and classes
- 4-space indentation
- Group imports: standard library, third-party, then local
- Document user-facing changes in the README

## Scope

ZenTTS is deliberately English-only (`en-us` and `en-gb`). Please don't send
patches that add other languages — they widen the model surface the project
supports and are out of scope.

## Pull requests

Before opening a PR, make sure that:

- your code follows the style above
- you tested your changes
- your commit messages follow [COMMIT_GUIDELINES.md](COMMIT_GUIDELINES.md)
- you updated the README where it matters
- existing functionality still works

Use the pull request template — it prompts for everything reviewers need.

## Testing

There is no automated test suite yet, so test by hand what your change touches:

- plain `.txt` input to `.wav` and `.mp3`
- EPUB and PDF chapter extraction
- `--split-output`, resuming an interrupted run, and `--merge-chunks`
- `--stream` playback
- single voices and blended voices
- a fresh `--download` on a machine with no cached model

## Reporting issues

Use the issue templates. Include your OS, Python version, the exact command you
ran, and the full error output. Search existing issues first to avoid
duplicates. For questions, use GitHub Discussions rather than the issue tracker.

## License

ZenTTS is proprietary software, © 2026 Omor, all rights reserved. By submitting
a contribution you assign copyright in it to the project owner, so that it can
be distributed under the project's [LICENSE](LICENSE). If you are not willing
to do that, please open an issue describing the change instead of sending code.

Contact: https://www.linkedin.com/in/omardeveloper/
