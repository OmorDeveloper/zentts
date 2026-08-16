"""Static configuration for the ZenTTS engine."""

from dataclasses import dataclass

# ZenTTS is English-only: US and UK.
SUPPORTED_LANGUAGES = ["en-us", "en-gb"]

# Voice-name prefixes for the English voice packs
# (a = American, b = British, f = female, m = male).
VOICE_PREFIXES = ("af_", "am_", "bf_", "bm_")

# The ONNX graph is compiled for a fixed context length.
MAX_PHONEME_LENGTH = 510

# Output sample rate of the model, in Hz.
SAMPLE_RATE = 24000

# Model release the code is built against.
MODEL_VERSION = "v1.0"
MODEL_FILENAME = f"zentts-{MODEL_VERSION}.onnx"
VOICES_FILENAME = f"zentts-voices-{MODEL_VERSION}.bin"

# Where the model files are published.
RELEASE_TAG = "v1.0.0"
RELEASE_BASE_URL = (
    f"https://github.com/OmorDeveloper/zentts/releases/download/{RELEASE_TAG}"
)
MODEL_URL = f"{RELEASE_BASE_URL}/{MODEL_FILENAME}"
VOICES_URL = f"{RELEASE_BASE_URL}/{VOICES_FILENAME}"


@dataclass
class EspeakConfig:
    """Override the bundled espeak-ng library/data locations."""

    lib_path: str | None = None
    data_path: str | None = None


def _build_vocab() -> dict[str, int]:
    """Symbol table shared by the model and the tokenizer."""
    pad = "$"
    punctuation = ';:,.!?¡¿—…"«»“” '
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    letters_ipa = (
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ"
        "ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    )
    symbols = [pad] + list(punctuation) + list(letters) + list(letters_ipa)
    return {symbol: index for index, symbol in enumerate(symbols)}


VOCAB = _build_vocab()
