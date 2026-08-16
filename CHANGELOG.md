# Changelog

## 1.3.0

- **Web studio.** `zentts start` now serves a browser interface at `/`: text
  box, every voice, speed, language, format, generate or stream, plus session
  and history management and a request log view.
- **Sessions and history, saved on your machine.** Create, rename and delete
  sessions; every clip is stored under `<ZENTTS_HOME>/sessions` with its text,
  voice and settings, and can be replayed or downloaded later.
  - `GET/POST /v1/sessions`, `GET/PATCH/DELETE /v1/sessions/{id}`
  - `GET /v1/sessions/{id}/items/{item}/audio` to download a clip
  - `session_id` on a speech request files the result in that session
- **Streaming.** `"stream": true` sends mp3 or pcm in chunks as it is made, so
  playback starts before the whole text is finished.
- **Request log endpoint**, `GET /v1/logs`, also shown in the studio.
- **Admin kill switch.** Every install checks a control file the author owns
  and stops when it is disabled, when its version is blocked, or when the
  package is unpublished. Cached for 12 hours, with 7 days of offline grace.
  `zentts --license` and `GET /v1/license` report the state. See ADMIN.md.
- Long text works end to end over the API: split, spoken in order, joined into
  one file.

## 1.2.0

- **`zentts start` runs an OpenAI-compatible speech API server.** Point any
  OpenAI client at `http://127.0.0.1:8000/v1` and it works unchanged; the
  official Python SDK is tested against it. Built on the standard library, so
  it adds no dependencies.
  - `POST /v1/audio/speech` with `input`, `voice`, `response_format`, `speed`
    and `language`
  - `GET /`, `/health`, `/v1/models`, `/v1/voices`
  - Formats: mp3, wav, flac, ogg, opus and raw pcm
  - OpenAI voice names (`nova`, `onyx`, `fable`, …) map onto ZenTTS voices,
    and voice blending works over the API too
  - Optional bearer-token auth via `--api-key` or `ZENTTS_API_KEY`, CORS on by
    default, request bodies capped at 10 MB
- A speed outside the engine's range is clamped rather than refused, and the
  response reports it in `X-ZenTTS-Speed-Clamped`.

## 1.1.2

- `--version` now prints the author and copyright line alongside the version.
- The module carries `__version__`, `__author__`, `__license__` and `__url__`,
  and its header states the copyright, so authorship travels with the file.

## 1.1.1

- Import `sounddevice` only when `--stream` is used, so `import zentts` no
  longer fails on machines without the PortAudio system library. Writing audio
  to a file now works on a bare Linux server; only playback needs PortAudio,
  and the error says how to install it.

## 1.1.0

Renamed voices, so this is a breaking change for anyone who used 1.0.0.

- **Voices renamed** to ZenTTS ids: `zen_us_f01`–`zen_us_f11`,
  `zen_us_m01`–`zen_us_m09`, `zen_uk_f01`–`zen_uk_f04`, `zen_uk_m01`–`zen_uk_m04`.
  The default voice is `zen_us_f10`. `--help-voices` groups them by region and
  gender. The voice pack file itself was rewritten, so old ids no longer exist.
- **Several input files at once**, joined into a single output:
  `zentts ch1.txt ch2.txt ch3.txt book.wav`. Formats can be mixed. If the last
  argument ends in `.wav` or `.mp3` it is the output, otherwise the name comes
  from the first input. `--stream` plays the inputs back to back.
- **`--download` now verifies** the files it finds against their published
  sha256 and replaces anything stale, which is how an older voice pack gets
  upgraded. Loading an outdated pack gives a clear message instead of an
  obscure failure.
- **Single module**: the package was merged into one `zentts.py`.
- **Relicensed** from MIT to a proprietary ZenTTS licence. Third-party
  attributions remain in `NOTICE.md`, as their licences require.
- **Fixed** a chunking bug where a sentence longer than the chunk size was
  emitted before the text queued ahead of it, reordering the audio. Split
  pieces no longer get a doubled full stop.
- **Added** a test suite (64 tests) and CI on Linux, Windows and macOS for
  Python 3.11 and 3.12.

## 1.0.0

First release. **Yanked** — it was built before the rename, so it carries the
old voice ids and a checksum that no longer matches the published voice pack.
Use 1.1.1 or later.

- English-only text-to-speech CLI for `.txt`, `.epub` and `.pdf` input
- Self-contained ZenTTS engine, with no third-party TTS package dependency
- Automatic model download with progress and sha256 verification
- Chapter splitting, resumable runs, chunk merging, voice blending, streaming
