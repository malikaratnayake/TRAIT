"""
src/utils/smoothing.py
======================
Gap filling and Savitzky–Golay smoothing of centroid coordinates (paper
Sec. 3.2). Shared by src/generate_features.py and the sequence baselines
(src/train_fcn.py, src/train_rocket.py, src/train_convtran.py) so that every
model sees the same preprocessed trajectory.

Public API
----------
  smooth_coords(x, y, window_length=11, polyorder=2) -> (sx, sy)   numpy arrays
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _pick_savgol_window(n: int, requested: int) -> int | None:
    """Return the largest valid odd window ≤ min(requested, n), or None."""
    if n < 3:
        return None
    window = min(requested, n)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return None
    return window


def smooth_coords(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Gap-fill and Savitzky–Golay smooth (x, y) coordinate arrays.

    NaN gaps are filled by linear interpolation first. For tracks shorter than
    window_length the largest valid odd window is used; tracks too short to
    filter are returned interpolated but unsmoothed.

    Parameters
    ----------
    x, y          : 1-D float arrays (NaNs allowed — will be interpolated first)
    window_length : Savitzky–Golay window size (odd integer)
    polyorder     : Savitzky–Golay polynomial order

    Returns
    -------
    sx, sy : smoothed float arrays (same length as input)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n == 0:
        return x.copy(), y.copy()

    # Interpolate any NaN gaps before smoothing
    idx = np.arange(n)
    for arr in (x, y):
        nans = np.isnan(arr)
        if nans.any() and (~nans).any():
            arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])

    w = _pick_savgol_window(n, window_length)
    if w is None or w <= polyorder:
        return x.copy(), y.copy()
    sx = savgol_filter(x, w, polyorder)
    sy = savgol_filter(y, w, polyorder)
    return sx, sy
