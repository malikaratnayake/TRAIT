"""Portable matplotlib cache setup for batch and headless environments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
        return True
    except OSError:
        return False


def _pick_cache_root() -> Path:
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
