"""Locating and downloading the ZenTTS model files."""

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import (
    MODEL_FILENAME,
    MODEL_URL,
    VOICES_FILENAME,
    VOICES_URL,
)

# sha256 of the published release assets.
CHECKSUMS = {
    MODEL_FILENAME: "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    VOICES_FILENAME: "5944de81ec3459cd93acf3ce0ce50e1d69ff6bfd4847750cc468fe84fdd92597",
}

DOWNLOAD_URLS = {
    MODEL_FILENAME: MODEL_URL,
    VOICES_FILENAME: VOICES_URL,
}


def model_dir() -> Path:
    """Directory the downloaded model files live in.

    Override it with ZENTTS_HOME; otherwise use the per-user cache directory.
    """
    env_home = os.getenv("ZENTTS_HOME")
    if env_home:
        return Path(env_home).expanduser()
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base) / "zentts"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "zentts"
    base = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "zentts"


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, quiet: bool = False) -> Path:
    """Download `url` to `destination`, showing progress and verifying the hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if not quiet:
        print(f"Downloading {destination.name}")
        print(f"  from {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "zentts"})
    started = time.time()
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            with open(partial, "wb") as out:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    downloaded += len(block)
                    if quiet:
                        continue
                    if total:
                        done = int(30 * downloaded / total)
                        bar = "■" * done + "□" * (30 - done)
                        percent = 100 * downloaded / total
                        sys.stdout.write(
                            f"\r  [{bar}] {percent:5.1f}% "
                            f"({_human(downloaded)} / {_human(total)})"
                        )
                    else:
                        sys.stdout.write(f"\r  {_human(downloaded)} downloaded")
                    sys.stdout.flush()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {e}") from e

    if not quiet:
        sys.stdout.write(f"\n  finished in {time.time() - started:.1f}s\n")

    expected = CHECKSUMS.get(destination.name)
    if expected:
        if not quiet:
            print("  verifying checksum...")
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}\n"
                f"  expected {expected}\n  got      {actual}"
            )

    partial.replace(destination)
    return destination


def _find_existing(filename: str) -> Path | None:
    """Look for a model file in the working directory, then the cache."""
    for candidate in (Path.cwd() / filename, model_dir() / filename):
        if candidate.exists():
            return candidate
    return None


def resolve_file(
    filename: str, explicit_path: str | None = None, allow_download: bool = True
) -> Path:
    """Return a usable path for a model file, downloading it if needed."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    existing = _find_existing(filename)
    if existing:
        return existing

    url = DOWNLOAD_URLS.get(filename)
    if not allow_download or not url:
        raise FileNotFoundError(
            f"{filename} not found in {Path.cwd()} or {model_dir()}.\n"
            f"Download it from {url or 'the project release page'} "
            "or pass --model / --voices."
        )

    return download_file(url, model_dir() / filename)


def resolve_model_files(
    model_path: str | None = None,
    voices_path: str | None = None,
    allow_download: bool = True,
) -> tuple[str, str]:
    """Resolve both model files, returning their paths as strings."""
    if os.getenv("ZENTTS_NO_DOWNLOAD"):
        allow_download = False

    model = resolve_file(MODEL_FILENAME, model_path, allow_download)
    voices = resolve_file(VOICES_FILENAME, voices_path, allow_download)
    return str(model), str(voices)


def ensure_models(quiet: bool = False) -> tuple[Path, Path]:
    """Fetch both model files into the cache if they are not there yet."""
    paths = []
    for filename in (MODEL_FILENAME, VOICES_FILENAME):
        existing = _find_existing(filename)
        if existing:
            if not quiet:
                print(f"{filename} already present at {existing}")
            paths.append(existing)
        else:
            paths.append(
                download_file(DOWNLOAD_URLS[filename], model_dir() / filename, quiet)
            )
    return paths[0], paths[1]
