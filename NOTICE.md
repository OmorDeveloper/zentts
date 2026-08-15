# Third-party notices

zentts is built on top of the following open-source components. They are not
part of zentts's own MIT license (see LICENSE) — each keeps its original
license.

## kokoro-onnx

zentts depends on the `kokoro-onnx` Python package (installed via pip, not
vendored/copied into this repo) for the underlying TTS runtime.

- Project: https://github.com/thewh1teagle/kokoro-onnx
- License: MIT

## Kokoro model weights

The `.onnx` model and voice-pack files zentts downloads at runtime originate
from the Kokoro TTS model project. Check that project's license before
redistributing the weight files yourself.
