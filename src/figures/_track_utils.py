"""Shared track smoothing helpers for trajectory figures."""

from __future__ import annotations

import numpy as np


def _sg_coeffs(w: int, poly: int = 2) -> np.ndarray:
    half = w // 2
    pos = np.arange(-half, half + 1, dtype=float)
    A = np.column_stack([pos ** i for i in range(poly + 1)])
    return np.linalg.solve(A.T @ A, A.T)[0]


def sg_smooth(arr: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Savitzky–Golay smooth (numpy-only); interpolates NaNs, keeps edges raw."""
    arr = np.asarray(arr, float).copy()
    mask = np.isnan(arr)
    if mask.any() and (~mask).any():
        arr[mask] = np.interp(
            np.where(mask)[0], np.where(~mask)[0], arr[~mask]
        )
    n = len(arr)
    w = min(window, n)
    w = w - 1 if w % 2 == 0 else w
    if w < 3 or w <= poly:
        return arr
    c = _sg_coeffs(w, poly)
    out = np.convolve(arr, c[::-1], mode="same")
    h = w // 2
    out[:h] = arr[:h]
    out[-h:] = arr[-h:]
    return out
