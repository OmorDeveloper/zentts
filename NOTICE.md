# Third-party notices

ZenTTS is MIT licensed (see [LICENSE](LICENSE)). It builds on the open-source
work listed below, which keeps its own license. These notices are required by
those licenses and must stay with the project.

## Inference code

`zentts/engine.py`, `zentts/tokenizer.py` and `zentts/config.py` are derived
from the `kokoro-onnx` project by thewh1teagle, licensed MIT.

- Project: https://github.com/thewh1teagle/kokoro-onnx
- License: MIT

```
MIT License

Copyright (c) 2025 thewh1teagle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model weights

The `zentts-v1.0.onnx` model and the `zentts-voices-v1.0.bin` voice packs are
redistributions of the Kokoro TTS model weights by hexgrad, licensed
Apache-2.0. The files are renamed for this project; the weights are unmodified.

- Project: https://huggingface.co/hexgrad/Kokoro-82M
- License: Apache-2.0

## Runtime dependencies

Installed by pip, not vendored here: `onnxruntime` (MIT), `phonemizer-fork`
(GPL-3.0), `espeakng-loader` and the bundled espeak-ng data (GPL-3.0),
`numpy` (BSD-3-Clause), `soundfile` and `sounddevice` (BSD-3-Clause),
`PyMuPDF` (AGPL-3.0 or commercial), `EbookLib` (AGPL-3.0),
`beautifulsoup4` (MIT).

Note that espeak-ng/phonemizer are GPL-3.0 and PyMuPDF/EbookLib are AGPL-3.0.
That does not affect using ZenTTS, but it does constrain how a combined
distribution may be relicensed.
