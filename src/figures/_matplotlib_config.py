"""Point matplotlib and fontconfig caches at a writable directory.

Needed on clusters and other headless or read-only environments where the
default cache location under the home directory cannot be written.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _is_writable_dir(path: Path) -> bool:
    """True if path exists (or can be created) and a file can be written in it."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
        return True
    except OSError:
        return False


def _pick_cache_root() -> Path:
    """First writable directory among TRAIT_CACHE_DIR, the temp-dir variables and ./.cache."""
    candidates = [
        os.environ.get("TRAIT_CACHE_DIR"),
        os.environ.get("TMPDIR"),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        tempfile.gettempdir(),
        Path.cwd() / ".cache",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if _is_writable_dir(path):
            return path
    raise RuntimeError("No writable cache directory found for matplotlib/fontconfig")


def _set_writable_env_dir(env_name: str, default_path: Path) -> None:
    """Set env_name to default_path unless it already points at a writable directory."""
    current = os.environ.get(env_name)
    if current and _is_writable_dir(Path(current).expanduser()):
        return

    if not _is_writable_dir(default_path):
        raise RuntimeError(f"Cannot create writable cache directory: {default_path}")
    os.environ[env_name] = str(default_path)


def configure_matplotlib_cache() -> None:
    """Set writable cache directories before importing matplotlib."""
    cache_root = _pick_cache_root()
    _set_writable_env_dir("MPLCONFIGDIR", cache_root / "matplotlib-cache")
    _set_writable_env_dir("XDG_CACHE_HOME", cache_root / "fontconfig-cache")
