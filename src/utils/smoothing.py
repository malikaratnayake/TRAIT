"""
src/utils/smoothing.py
======================
Coordinate smoothing utilities shared by process_track.py and
smoothing_experiment.py.

Two methods are supported:

  savgol   — Savitzky-Golay filter (default, polynomial fit in a sliding
              window; good at preserving sharp direction changes)
  gaussian — Gaussian convolution (σ in pixels; suppresses high-frequency
              jitter more uniformly; tested in Step 1.2)

Public API
----------
  smooth_coords(x, y, method, **kwargs) -> (sx, sy)   numpy arrays
  smooth_track_df(df, method, **kwargs)  -> df         with sx / sy added
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _interpolate_1d(series: pd.Series, frame_index: pd.Series) -> np.ndarray:
    """Linear interpolation for NaN gaps; returns a float numpy array."""
    s = series.copy().astype(float)
    s.index = frame_index
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy(dtype=float)


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


# ---------------------------------------------------------------------------
# Core smoothing
# ---------------------------------------------------------------------------

def smooth_coords(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "savgol",
    *,
    window_length: int = 11,
    polyorder: int = 2,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth (x, y) coordinate arrays.

    Parameters
    ----------
    x, y          : 1-D float arrays (NaNs allowed — will be interpolated first)
    method        : 'savgol' or 'gaussian'
    window_length : Savitzky-Golay window size (odd integer, savgol only)
    polyorder     : Savitzky-Golay polynomial order (savgol only)
    sigma         : Gaussian σ in frames (gaussian only)

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

    if method == "gaussian":
        sx = gaussian_filter1d(x, sigma=sigma)
        sy = gaussian_filter1d(y, sigma=sigma)
        return sx, sy

    if method == "savgol":
        w = _pick_savgol_window(n, window_length)
        if w is None or w <= polyorder:
            return x.copy(), y.copy()
        sx = savgol_filter(x, w, polyorder)
        sy = savgol_filter(y, w, polyorder)
        return sx, sy

    raise ValueError(f"Unknown smoothing method: {method!r}. Use 'savgol' or 'gaussian'.")


# ---------------------------------------------------------------------------
# DataFrame-level helper
# ---------------------------------------------------------------------------

def smooth_track_df(
    df: pd.DataFrame,
    method: str = "savgol",
    *,
    xcol: str = "x",
    ycol: str = "y",
    frame_col: str = "nframe",
    window_length: int = 11,
    polyorder: int = 2,
    sigma: float = 1.0,
    round_int: bool = False,   # S-1 fix: was True — integer rounding destroys sub-pixel residuals
) -> pd.DataFrame:
    """Add sx / sy columns to a track DataFrame.

    Trims trailing all-NaN rows, interpolates gaps, then smooths.
    If round_int=True the smoothed values are rounded and stored as Int64
    (nullable integer).  Default is False to preserve sub-pixel precision
    needed by wing-beat residual features.
    """
    df = df.copy()

    # Trim rows after the last valid x/y
    last_valid = df[[xcol, ycol]].dropna().index.max()
    if pd.isna(last_valid):
        df["sx"] = pd.NA
        df["sy"] = pd.NA
        return df
    df = df.loc[:last_valid]

    fi = df[frame_col] if frame_col in df.columns else pd.Series(range(len(df)))
    x_interp = _interpolate_1d(df[xcol], fi)
    y_interp = _interpolate_1d(df[ycol], fi)

    sx, sy = smooth_coords(
        x_interp, y_interp,
        method=method,
        window_length=window_length,
        polyorder=polyorder,
        sigma=sigma,
    )

    if round_int:
        df["sx"] = pd.array(np.round(sx).astype(float), dtype=pd.Int64Dtype())
        df["sy"] = pd.array(np.round(sy).astype(float), dtype=pd.Int64Dtype())
    else:
        df["sx"] = sx
        df["sy"] = sy

    return df
