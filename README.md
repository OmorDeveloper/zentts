# zentts

A command-line text-to-speech tool for English audio. Converts `.txt`,
`.epub`, and `.pdf` input into `.wav` or `.mp3` output, with per-chapter
splitting, voice blending, and streaming playback.

## Install

```bash
pip install -e .
```

## Required model files

Place these two files in the directory you run `zentts` from (or point to
them with `--model` / `--voices`):

- `ZENTTS-v1.0.onnx`
- `voices-v1.0.bin`

If they're missing, `zentts` will print the exact `wget` commands to fetch
them from your release page.

## Usage

```bash
zentts input.txt output.wav --speed 1.2 --lang en-us --voice af_sarah
zentts input.epub --split-output ./chunks/ --format mp3
zentts input.txt --stream --voice "af_sarah:60,am_adam:40"
zentts --help-voices
zentts --help-languages
zentts --help
```

## Voices and languages

English only — US (`af_`/`am_` prefixes) and UK (`bf_`/`bm_` prefixes)
voices, `en-us` and `en-gb` languages. Run `zentts --help-voices` to list
what's available from your voice file.

## Credits

Built on the Kokoro TTS engine via the `kokoro-onnx` package (Apache 2.0).
