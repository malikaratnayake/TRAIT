"""
src/generate_features.py
========================
Phase 1 — Gap filling, smoothing and extraction of the 125 motion features.

For every track listed in the labels CSV, the raw centroid coordinates are
gap-filled by linear interpolation and smoothed with a Savitzky–Golay filter
(11-frame window, polynomial order 2; src/utils/smoothing.py), and the 125
behaviour-informed features are computed from the smoothed trajectory (paper
Sec. 3.2–3.3; definitions and equations in Section 1 of the supplementary
material). Only the wingbeat-residual features also use the raw coordinates.
No recording metadata (time of day, station, farm) is used as a feature.

The features are organised into six domains and 32 subgroups, declared in
src/models/data_loader.py; this script computes them with reusable formula
blocks and writes them in that canonical order:

  D1 Fundamental locomotion kinematics (26)
     d1_duration, d1_speed, d1_dynamic, d1_distributional, d1_intensity
  D2 Spatial occupancy and path geometry (19)
     d2_path_geometry, d2_spatial_spread, d2_exploration,
     d2_diffusion, d2_sinuosity
  D3 Flight rhythm, oscillation and biomechanics (30)
     d3_h_oscillation, d3_v_oscillation, d3_cross_axis, d3_spectral,
     d3_residual, d3_hover_stab (mean reversion)
  D4 Navigation, orientation and search behaviour (16)
     d4_directional, d4_heading_occ, d4_zigzag, d4_looping, d4_bias
  D5 Sensorimotor control and manoeuvrability (20)
     d5_turning, d5_saccade, d5_fixation, d5_maneuver, d5_power_law,
     d5_levy, d5_aero
  D6 Behavioural state organisation (14)
     d6_hover, d6_stopping, d6_landing, d6_takeoff

Tracks with fewer than 5 points get only the track-support features; all
other features are NaN and are median-imputed from training rows later. (The
released dataset already excludes tracks shorter than five frames; its
shortest track has 10 points.)

Large datasets can be processed in batches (--batch-size, checkpoints in
--ckpt-dir) and merged with --finalize; run_pipeline.sh does this.

Usage
-----
  python src/generate_features.py \\
      --tracks-dir  data/tracks \\
      --labels-csv  data/track_classification.csv \\
      --fps         30 \\
      --output-csv  data/results/trait_results/track_features.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial import ConvexHull
from scipy.signal import welch as _scipy_welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.smoothing import smooth_coords
from src.models.data_loader import ALL_FEATURES, FEATURE_GROUPS


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to [−π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; returns NaN if degenerate."""
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _canonical_feature_order(feats: dict) -> dict:
    """Return six-domain model features first, with optional extras appended."""
    ordered = {name: feats.get(name, np.nan) for name in ALL_FEATURES}
    for name, value in feats.items():
        if name not in ordered:
            ordered[name] = value
    return ordered


# ===========================================================================
# DOMAIN-ORDERED FEATURE HELPERS
# ===========================================================================
# Helper definitions are ordered by the canonical D1 -> D6 hierarchy and by
# subgroup within each domain. A few helpers emit features used by multiple
# subgroups; those helpers are placed at the earliest subgroup they support.

# ---------------------------------------------------------------------------
# Domain 1: Fundamental locomotion kinematics (26)
# ---------------------------------------------------------------------------
# d1_duration: track support
def feat_basic(frames: np.ndarray, fps: float) -> dict:
    """Track support: number of points, frame span and duration in seconds."""
    n = len(frames)
    dur_frames = int(frames[-1] - frames[0] + 1) if n else n
    return {
        "n_points": n,
        "duration_frames": dur_frames,
        "duration_sec": dur_frames / fps,
    }


# d1_speed: core speed statistics
def feat_speed(speed: np.ndarray) -> dict:
    """Speed statistics: mean, SD, max, 90th percentile and coefficient of variation."""
    return {
        "mean_speed": float(np.mean(speed)),
        "std_speed": float(np.std(speed)),
        "max_speed": float(np.max(speed)),
        "p90_speed": float(np.percentile(speed, 90)),
        "cv_speed": float(np.std(speed) / (np.mean(speed) + 1e-9)),
    }


# d1_dynamic: acceleration, jerk, effort, and acceleration envelope
def feat_accel(speed: np.ndarray, fps: float) -> dict:
    """Mean, SD and max of the absolute change in speed (px/s²)."""
    if len(speed) < 2:
        return {"mean_accel": np.nan, "std_accel": np.nan, "max_accel": np.nan}
    accel = np.diff(speed) * fps
    accel_abs = np.abs(accel)
    return {
        "mean_accel": float(np.mean(accel_abs)),
        "std_accel":  float(np.std(accel_abs)),   # std of |accel|, consistent with mean/max
        "max_accel":  float(np.max(accel_abs)),
    }


def feat_mean_abs_jerk(x: np.ndarray, y: np.ndarray, fps: float) -> dict:
    """Mean absolute jerk (rate of change of acceleration).

    Insects show frequent jerk spikes when changing direction abruptly.
    Wind plants produce smooth, sinusoidal motion → low jerk.
    """
    if len(x) < 5:
        return {"mean_abs_jerk": np.nan}
    vx = np.diff(x) * fps
    vy = np.diff(y) * fps
    ax = np.diff(vx) * fps
    ay = np.diff(vy) * fps
    jx = np.diff(ax) * fps
    jy = np.diff(ay) * fps
    jerk_mag = np.sqrt(jx**2 + jy**2)
    return {"mean_abs_jerk": float(np.mean(jerk_mag))}


def feat_kinematic_effort(speed: np.ndarray, x: np.ndarray,
                           y: np.ndarray, fps: float) -> dict:
    """Kinematic energy and effort proxies.

    Powered insect flight requires sustained mechanical effort against drag,
    distinguishing it from passive drift (debris) or wind-forced motion.

      • specific_kinetic_energy : mean(speed²) — weights bursts quadratically
      • power_proxy             : mean(speed × |accel|) — mechanical work rate
      • speed_ratio             : max_speed / mean_speed — burst intensity
      • jerk_fraction           : fraction of steps with jerk > 2σ
                                  (abrupt direction changes specific to insects)
    """
    nan_result = {"specific_kinetic_energy": np.nan, "power_proxy": np.nan,
                  "speed_ratio": np.nan, "jerk_fraction": np.nan}
    if len(speed) < 4:
        return nan_result

    ske   = float(0.5 * np.mean(speed**2))  # ½·mean(v²)
    accel = np.abs(np.diff(speed)) * fps
    power = float(np.mean(speed[:-1] * accel)) if len(accel) > 0 else np.nan

    mean_sp = float(np.mean(speed))
    max_sp  = float(np.max(speed))
    ratio   = float(max_sp / mean_sp) if mean_sp > 0 else np.nan

    # Jerk from positions
    if len(x) >= 5:
        vx = np.diff(x) * fps; vy = np.diff(y) * fps
        ax = np.diff(vx) * fps; ay = np.diff(vy) * fps
        jx = np.diff(ax) * fps; jy = np.diff(ay) * fps
        jmag  = np.sqrt(jx**2 + jy**2)
        jfrac = float(np.mean(jmag > np.mean(jmag) + 2 * np.std(jmag)))
    else:
        jfrac = np.nan

    return {"specific_kinetic_energy": ske, "power_proxy": power,
            "speed_ratio": ratio, "jerk_fraction": jfrac}


def feat_aerodynamic_envelope(x: np.ndarray, y: np.ndarray,
                               speed: np.ndarray, fps: float) -> dict:
    """D1 acceleration envelope and D5 Reynolds-regime proxy.

    accel_speed_envelope_slope:
        OLS slope of |a_i| regressed on v_i.
        Insects: active actuation → high accel at all speeds (slope ≈ 0).
        Wind-driven: a ∝ v² (drag-limited) → positive slope.

    re_proxy:
        mean_speed × sqrt(convex-hull area) — correlates with the actual
        Reynolds number. Insects occupy a narrow Re band (~10²–10⁴); debris
        much higher. (Falls back to sqrt of bbox area if the hull is degenerate.)
    """
    nan_r = {"accel_speed_envelope_slope": np.nan, "re_proxy": np.nan}
    n = len(speed)
    if n < 6:
        return nan_r

    accel_mag = np.abs(np.diff(speed)) * fps      # (n-1,)
    sp_inner  = speed[:-1]                         # aligned

    if len(accel_mag) >= 5 and np.std(sp_inner) > 1e-9:
        slope, _, _, _, _ = sp_stats.linregress(sp_inner, accel_mag)
        env_slope = float(slope)
    else:
        env_slope = np.nan

    # Re proxy: mean_speed × sqrt(hull_area)
    try:
        hull_area = float(ConvexHull(np.column_stack([x, y])).volume)
    except Exception:
        hull_area = float((np.max(x) - np.min(x)) * (np.max(y) - np.min(y)) + 1e-6)
    re_proxy = float(np.mean(speed) * np.sqrt(hull_area + 1e-6))

    return {
        "accel_speed_envelope_slope": env_slope,
        "re_proxy": re_proxy,
    }


# d1_distributional: speed and step distribution shape
# feat_robustness is a shared helper; it is placed here because its first
# canonical outputs are D1 distribution/intensity features.
def feat_speed_shape(speed: np.ndarray) -> dict:
    """Higher-order speed statistics.

    Insects show right-skewed speed (mostly slow, with burst spikes).
    Wind plants show symmetric or weakly-skewed speed.
    Kurtosis: peaky speed distribution = insect bursts.
    """
    if len(speed) < 5:
        return {"speed_skewness": np.nan, "speed_kurtosis": np.nan}
    return {
        "speed_skewness": float(sp_stats.skew(speed)),
        "speed_kurtosis": float(sp_stats.kurtosis(speed)),
    }


def feat_levy_steps(x: np.ndarray, y: np.ndarray) -> dict:
    """Lévy flight step-length statistics.

    Insect foraging follows Lévy-like step distributions (power law).
    Non-insects (debris, noise) follow exponential or Gaussian step lengths.

      • levy_mu            : power-law exponent μ via Clauset ML estimator
                             Lévy walkers: 1 < μ < 3; random walk: μ → ∞
      • step_heavy_tail    : fraction of steps > mean + 2·std
                             (heavy-tailed ↔ insect burst-glide patterns)
      • step_cv            : coefficient of variation of step lengths
    """
    n = len(x)
    nan_result = {"levy_mu": np.nan, "step_heavy_tail": np.nan, "step_cv": np.nan}
    if n < 10:
        return nan_result

    steps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    steps = steps[steps > 0]
    if len(steps) < 5:
        return nan_result

    # Clauset et al. MLE for power-law exponent
    s_min = np.percentile(steps, 10)   # use 10th pct as x_min to avoid zeros
    s_above = steps[steps >= s_min]
    if len(s_above) < 3:
        return nan_result
    log_sum = float(np.sum(np.log(s_above / s_min)))
    mu = float(1 + len(s_above) / log_sum) if log_sum > 1e-12 else np.nan

    mean_s, std_s = np.mean(steps), np.std(steps)
    heavy = float(np.mean(steps > mean_s + 2 * std_s))
    cv    = float(std_s / mean_s) if mean_s > 0 else np.nan

    return {"levy_mu": mu, "step_heavy_tail": heavy, "step_cv": cv}


def feat_robustness(
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    fps: float,
) -> dict:
    """Calculate 12 spectral-and-kinematic robustness features.

    Features
    --------
    benhamou_sinuosity    : Benhamou (2004) sinuosity = 2·sqrt((1−r) / (L̄·(1+r)))
                            where r = mean(cos(turning_angles)) and L̄ = mean step
                            length. Rises with tortuosity (low r → high sinuosity).
    spectral_entropy      : Normalised Shannon entropy of the speed PSD.
    spectral_flatness     : Wiener entropy of speed PSD (geometric/arithmetic mean).
    low_band_psd_ratio    : Fraction of speed PSD power below 1 Hz.
    spectral_peak_Q       : Q-factor of the dominant spectral peak (f_peak / bandwidth).
    speed_burst_fraction  : Fraction of frames where instantaneous speed exceeds
                            2 × median speed.
    heading_decorr_time   : Lag (s) at which the circular heading ACF
                            C(τ)=mean(cos(Δθ)) first drops to 1/e.
    jerk_asymmetry        : Skewness of the signed jerk distribution.
    accel_mad_spike_rate  : Fraction of frames where |accel| exceeds
                            median(|accel|) + 3 × MAD(|accel|) (robust spike rate).
    hover_transition_rate : Transitions per second in/out of hover state
                            (speed < 0.5 px/frame × fps).
    stft_stationarity_idx : Std of power-weighted spectral centroid across STFT windows.
    speed_iqr_over_median : IQR(speed) / median(speed), NaN if median < 1e-6.
    Note: spatial spread is provided separately as "radius_gyration".
    """
    n = len(x)
    out: dict = {}

    # ── Minimal guard ───────────────────────────────────────────────────────
    if n < 5 or len(speed) < 4:
        nans = {k: np.nan for k in [
            "benhamou_sinuosity", "spectral_entropy", "spectral_flatness",
            "low_band_psd_ratio", "spectral_peak_Q", "speed_burst_fraction",
            "heading_decorr_time", "jerk_asymmetry",
            "accel_mad_spike_rate", "hover_transition_rate",
            "stft_stationarity_idx", "speed_iqr_over_median",
        ]}
        return nans

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)          # per-frame step lengths (n-1,)

    # ── Benhamou (2004) sinuosity ────────────────────────────────────────────
    # S = 2 * sqrt((1 - r) / (L̄ * (1 + r)))  where r = mean(cos(turning_angles))
    # and L̄ = mean step length  (Benhamou 2004, Am. Nat. 163:595–605).
    # High S → tortuous (insect foraging); low S → straight / confined (plant).
    try:
        mean_step = float(step.mean())
        if mean_step > 1e-9 and len(dx) >= 3:
            headings_b = np.arctan2(dy, dx)
            dtheta_b   = _wrap(np.diff(headings_b))
            r          = float(np.mean(np.cos(dtheta_b)))
            denom      = mean_step * (1.0 + r)
            out["benhamou_sinuosity"] = (
                float(2.0 * np.sqrt((1.0 - r) / denom)) if denom > 1e-9 else np.nan
            )
        else:
            out["benhamou_sinuosity"] = np.nan
    except Exception:
        out["benhamou_sinuosity"] = np.nan

    # ── Speed PSD (Welch) ────────────────────────────────────────────────────
    speed_demean = speed - speed.mean()
    nperseg = min(len(speed), 64)
    try:
        f_arr, psd = _scipy_welch(speed_demean, fs=fps, nperseg=nperseg)
        psd_sum = psd.sum()
    except Exception:
        f_arr, psd, psd_sum = None, None, 0.0

    # spectral_entropy
    if psd is not None and psd_sum > 1e-30:
        p_norm = psd / psd_sum
        p_norm = p_norm[p_norm > 0]
        H = -float(np.sum(p_norm * np.log(p_norm)))
        H_max = float(np.log(len(p_norm))) if len(p_norm) > 1 else 1.0
        out["spectral_entropy"] = H / H_max if H_max > 0 else np.nan
    else:
        out["spectral_entropy"] = np.nan

    # spectral_flatness  (geometric mean / arithmetic mean of PSD)
    if psd is not None and psd_sum > 1e-30 and np.all(psd >= 0):
        pos = psd[psd > 0]
        if len(pos) > 0:
            log_mean = float(np.mean(np.log(pos)))
            arith_mean = float(np.mean(psd))
            out["spectral_flatness"] = (
                float(np.exp(log_mean)) / arith_mean if arith_mean > 1e-30 else np.nan
            )
        else:
            out["spectral_flatness"] = np.nan
    else:
        out["spectral_flatness"] = np.nan

    # low_band_psd_ratio  — fraction of power at or below 1 Hz
    if psd is not None and psd_sum > 1e-30:
        out["low_band_psd_ratio"] = float(psd[f_arr <= 1.0].sum() / psd_sum)
    else:
        out["low_band_psd_ratio"] = np.nan

    # spectral_peak_Q  — peak frequency / 3 dB bandwidth
    if psd is not None and len(psd) > 2:
        try:
            peak_idx = int(np.argmax(psd))
            f_peak   = f_arr[peak_idx]
            half_pow = psd[peak_idx] / 2.0
            # bandwidth: distance between -3 dB crossings
            above = f_arr[psd >= half_pow]
            bw = float(above[-1] - above[0]) if len(above) > 1 else (f_arr[1] - f_arr[0])
            out["spectral_peak_Q"] = float(f_peak / bw) if bw > 1e-9 else np.nan
        except Exception:
            out["spectral_peak_Q"] = np.nan
    else:
        out["spectral_peak_Q"] = np.nan

    # ── speed_burst_fraction ─────────────────────────────────────────────────
    try:
        med_spd = float(np.median(speed))
        out["speed_burst_fraction"] = (
            float(np.mean(speed > 2.0 * med_spd)) if med_spd > 1e-9 else np.nan
        )
    except Exception:
        out["speed_burst_fraction"] = np.nan

    # ── heading_decorr_time ──────────────────────────────────────────────────
    # Time (s) at which the circular directional autocorrelation first drops to
    # 1/e ≈ 0.368. Headings are circular, so we use C(τ) = mean(cos(θ_{i+τ} − θ_i))
    # which is wraparound-safe (no linear mean-centring of angles).
    INV_E = float(np.exp(-1.0))
    try:
        heading = np.arctan2(dy, dx)          # (n-1,)
        max_lag = min(len(heading) - 1, 60)
        if max_lag >= 2:
            decorr_lag = None
            for lag in range(1, max_lag + 1):
                acf_val = float(np.mean(np.cos(heading[lag:] - heading[:-lag])))
                if acf_val <= INV_E:
                    decorr_lag = lag
                    break
            out["heading_decorr_time"] = (
                float(decorr_lag) / fps if decorr_lag is not None else float(max_lag) / fps
            )
        else:
            out["heading_decorr_time"] = np.nan
    except Exception:
        out["heading_decorr_time"] = np.nan

    # ── jerk_asymmetry ───────────────────────────────────────────────────────
    try:
        accel_mag = np.abs(np.diff(speed)) * fps
        if len(accel_mag) >= 3:
            jerk_signed = np.diff(speed, n=2) * (fps ** 2)
            out["jerk_asymmetry"] = float(sp_stats.skew(jerk_signed))
        else:
            out["jerk_asymmetry"] = np.nan
    except Exception:
        out["jerk_asymmetry"] = np.nan

    # ── accel_mad_spike_rate ─────────────────────────────────────────────────
    try:
        accel_abs = np.abs(np.diff(speed) * fps)
        if len(accel_abs) >= 4:
            med_a = float(np.median(accel_abs))
            mad_a = float(np.median(np.abs(accel_abs - med_a)))
            thr   = med_a + 3.0 * mad_a   # robust spike level above the typical value
            out["accel_mad_spike_rate"] = float(np.mean(accel_abs > thr))
        else:
            out["accel_mad_spike_rate"] = np.nan
    except Exception:
        out["accel_mad_spike_rate"] = np.nan

    # ── hover_transition_rate ────────────────────────────────────────────────
    # Hover = speed < 0.5 px/frame (in px/s, threshold = 0.5 * fps)
    try:
        thr_low = 0.5 * fps          # px/s
        is_hover = (speed < thr_low).astype(int)
        transitions = int(np.sum(np.abs(np.diff(is_hover))))
        dur_sec = len(speed) / fps
        out["hover_transition_rate"] = float(transitions / dur_sec) if dur_sec > 0 else np.nan
    except Exception:
        out["hover_transition_rate"] = np.nan

    # ── stft_stationarity_idx ────────────────────────────────────────────────
    # Std of power-weighted spectral centroid across STFT windows
    try:
        win_len = 16
        hop     = win_len // 2
        centroids = []
        i = 0
        while i + win_len <= len(speed):
            seg = speed[i : i + win_len] - speed[i : i + win_len].mean()
            f_seg, p_seg = _scipy_welch(seg, fs=fps, nperseg=win_len)
            p_tot = p_seg.sum()
            if p_tot > 1e-30:
                centroid = float(np.sum(f_seg * p_seg) / p_tot)
                centroids.append(centroid)
            i += hop
        out["stft_stationarity_idx"] = float(np.std(centroids)) if len(centroids) >= 2 else np.nan
    except Exception:
        out["stft_stationarity_idx"] = np.nan

    # ── speed_iqr_over_median ────────────────────────────────────────────────
    try:
        med_s = float(np.median(speed))
        if med_s > 1e-6:
            q75, q25 = float(np.percentile(speed, 75)), float(np.percentile(speed, 25))
            out["speed_iqr_over_median"] = (q75 - q25) / med_s
        else:
            out["speed_iqr_over_median"] = np.nan
    except Exception:
        out["speed_iqr_over_median"] = np.nan

    return out


# d1_intensity: burst, near-stop, jerk, and spike-rate measures
def feat_stops_bursts(speed: np.ndarray, fps: float) -> dict:
    """Stop and burst fractions from speed thresholds of 1 and max(3, Q75) px/frame."""
    # speed is in px/s, so the px/frame thresholds are multiplied by fps
    stop_thr  = 1.0 * fps
    burst_thr = max(3.0 * fps, float(np.percentile(speed, 75)))
    return {
        "stop_fraction": float(np.mean(speed < stop_thr)),
        "burst_fraction": float(np.mean(speed > burst_thr)),
    }


# ---------------------------------------------------------------------------
# Domain 2: Spatial occupancy and path geometry (19)
# ---------------------------------------------------------------------------
# d2_path_geometry: path length, displacement, straightness, and scales
def feat_trajectory(x: np.ndarray, y: np.ndarray) -> dict:
    """Path length, net displacement and straightness (net displacement / path length)."""
    dx, dy = np.diff(x), np.diff(y)
    step = np.sqrt(dx**2 + dy**2)
    path_len = float(np.sum(step))
    net_disp = float(np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2))
    straightness = net_disp / path_len if path_len > 0 else np.nan
    return {
        "path_length": path_len,
        "net_displacement": net_disp,
        "straightness": straightness,
    }


def feat_multiscale_straightness(x: np.ndarray, y: np.ndarray) -> dict:
    """Straightness computed at two temporal scales (short and long windows).

    Insects are tortuous at short scales (looping near a flower) but
    directed at coarse scales (travelling between patches).  Plant motion
    and debris have low or constant straightness across scales.

    Returns mean straightness at short / long windows and their ratio.
    """
    n = len(x)
    nan_result = {"straight_short": np.nan, "straight_long": np.nan,
                  "straight_scale_ratio": np.nan}
    if n < 10:
        return nan_result

    def _window_straightness(x, y, w):
        """Mean straightness over half-overlapping windows of w points."""
        vals = []
        for i in range(0, n - w, max(1, w // 2)):
            xs, ys = x[i:i+w], y[i:i+w]
            dxs, dys = np.diff(xs), np.diff(ys)
            pl = float(np.sum(np.sqrt(dxs**2 + dys**2)))
            nd = float(np.sqrt((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2))
            if pl > 0:
                vals.append(nd / pl)
        return float(np.nanmean(vals)) if vals else np.nan

    w_short = max(5,  min(10,  n // 4))
    w_long  = max(10, min(50,  n // 2))

    s_short = _window_straightness(x, y, w_short)
    s_long  = _window_straightness(x, y, w_long)
    ratio   = float(s_short / s_long) if (s_long and s_long > 0) else np.nan
    return {"straight_short": s_short, "straight_long": s_long,
            "straight_scale_ratio": ratio}


# d2_spatial_spread: bounding box, convex hull, and centroid spread
def feat_spatial(x: np.ndarray, y: np.ndarray) -> dict:
    """Bounding-box area and aspect ratio, radius of gyration and mean distance to the centroid."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    w, h = xmax - xmin, ymax - ymin
    cx, cy = np.mean(x), np.mean(y)
    rg = float(np.sqrt(np.mean((x - cx)**2 + (y - cy)**2)))
    scale = max(float(w), float(h), 1.0)
    aspect_eps = max(1e-6, 1e-9 * scale)
    return {
        "bbox_area": float(w * h),
        "bbox_aspect": float(w / h) if h > aspect_eps else np.nan,
        "radius_gyration": rg,
        "mean_dist_centroid": float(np.mean(np.sqrt((x - cx)**2 + (y - cy)**2))),
    }


def feat_convex_hull(x: np.ndarray, y: np.ndarray) -> dict:
    """Convex hull area.

    Insects explore 2D space freely → larger hull.
    Wind plants are tethered → small, flattened hull.
    """
    if len(x) < 4:
        return {"convex_hull_area": np.nan, "exploration_ratio": np.nan}
    pts = np.column_stack([x, y])
    # Need at least 3 non-collinear points for ConvexHull
    try:
        hull = ConvexHull(pts)
        area = float(hull.volume)  # .volume is 2D area in scipy
    except Exception:
        area = np.nan
    path_len = float(np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))
    exploration_ratio = area / (path_len + 1e-6) if not np.isnan(area) else np.nan
    return {"convex_hull_area": area, "exploration_ratio": exploration_ratio}


# d2_exploration: range use, confinement, entropy, and revisitation
def feat_track_confinement(x: np.ndarray, y: np.ndarray) -> dict:
    """Track confinement and start-anchoring features.

    track_confinement_ratio = total_path_length / max_displacement_from_start
      Plants: high ratio (long path but spatially confined near anchor)
      Insects: lower ratio (path length ~ displacement)

    max_displacement_from_start: bounded for plants, larger for insects.
    return_to_start_rate: fraction of frames within 10 px of start point.
    """
    if len(x) < 3:
        return {"track_confinement_ratio": np.nan, "max_displacement_from_start": np.nan,
                "return_to_start_rate": np.nan}
    dx, dy = np.diff(x), np.diff(y)
    path_len = float(np.sum(np.sqrt(dx**2 + dy**2)))
    dist_from_start = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    max_disp = float(np.max(dist_from_start))
    # path length / maximum displacement from the start
    confinement = path_len / (max_disp + 1e-6)
    return_rate = float(np.mean(dist_from_start < 10.0))
    return {
        "track_confinement_ratio": confinement,
        "max_displacement_from_start": max_disp,
        "return_to_start_rate": return_rate,
    }


def feat_spatial_entropy(x: np.ndarray, y: np.ndarray) -> dict:
    """Spatial coverage entropy and revisitation rate.

    A uniform forager visits all regions of its range; a transiting insect
    has elongated, sparse coverage; debris drifts with uneven coverage.

      • spatial_entropy    : Shannon entropy of 4×4 grid cell visits (0→1)
      • revisit_rate       : fraction of cells visited more than once
    """
    n = len(x)
    nan_result = {"spatial_entropy": np.nan, "revisit_rate": np.nan}
    if n < 8:
        return nan_result

    grid = 4
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    if xmax == xmin or ymax == ymin:
        return nan_result

    # Map positions to grid cells
    xi = np.clip(((x - xmin) / (xmax - xmin) * grid).astype(int), 0, grid-1)
    yi = np.clip(((y - ymin) / (ymax - ymin) * grid).astype(int), 0, grid-1)
    cell_ids = xi * grid + yi

    counts = np.bincount(cell_ids, minlength=grid*grid).astype(float)
    total  = counts.sum()
    if total == 0:
        return nan_result

    p = counts / total
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)) / np.log(grid * grid))

    revisit = float(np.sum(counts > 1) / (grid * grid))
    return {"spatial_entropy": H, "revisit_rate": revisit}


# d2_diffusion and d2_sinuosity: MSD exponent and path sinuosity
def feat_msd_alpha(x: np.ndarray, y: np.ndarray) -> dict:
    """Mean Squared Displacement power-law exponent α.

    MSD(τ) ∝ τ^α
      α < 1 : sub-diffusive (confined, tethered plant)
      α ≈ 1 : normal diffusion (random walk)
      α > 1 : super-diffusive / ballistic (directed insect flight)

    Expected: plants α ≈ 0.4–0.8, insects α ≈ 1.3–2.0
    """
    n = len(x)
    if n < 10:
        return {"msd_alpha": np.nan}
    max_lag = max(2, n // 4)
    tau = np.arange(1, max_lag + 1)
    msd = np.array([
        np.mean((x[t:] - x[:-t])**2 + (y[t:] - y[:-t])**2)
        for t in tau
    ])
    valid = msd > 0
    if valid.sum() < 3:
        return {"msd_alpha": np.nan}
    log_tau = np.log(tau[valid])
    log_msd = np.log(msd[valid])
    alpha = float(np.polyfit(log_tau, log_msd, 1)[0])
    return {"msd_alpha": alpha}


# ---------------------------------------------------------------------------
# Domain 3: Flight rhythm, oscillation and biomechanics (30)
# ---------------------------------------------------------------------------
# d3_h_oscillation and d3_v_oscillation: axis-wise spectral/autocorr rhythm
def feat_spectral(signal: np.ndarray, fps: float, axis_label: str) -> dict:
    """FFT summary of one coordinate signal, keyed by axis_label ('x' or 'y').

    Dominant frequency, its share of power, normalised spectral entropy,
    share of power at or below 1 Hz, and peak sharpness, over 0.1 Hz to the
    Nyquist frequency.
    """
    prefix = axis_label
    n = len(signal)
    if n < 8:
        return {f"dom_freq_{prefix}": np.nan, f"dom_power_ratio_{prefix}": np.nan,
                f"spec_entropy_{prefix}": np.nan, f"lowfreq_power_ratio_{prefix}": np.nan,
                f"peak_sharpness_{prefix}": np.nan}
    x = signal - np.mean(signal)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    power = np.abs(X)**2
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power[0] = 0.0
    band = (freqs >= 0.1) & (freqs <= fps / 2.0)
    if not np.any(band):
        return {f"dom_freq_{prefix}": np.nan, f"dom_power_ratio_{prefix}": np.nan,
                f"spec_entropy_{prefix}": np.nan, f"lowfreq_power_ratio_{prefix}": np.nan,
                f"peak_sharpness_{prefix}": np.nan}
    pb = power[band]
    fb = freqs[band]
    total = np.sum(pb)
    if total <= 0:
        return {f"dom_freq_{prefix}": np.nan, f"dom_power_ratio_{prefix}": np.nan,
                f"spec_entropy_{prefix}": np.nan, f"lowfreq_power_ratio_{prefix}": np.nan,
                f"peak_sharpness_{prefix}": np.nan}
    idx = np.argmax(pb)
    dom_power = pb[idx]
    p = pb / total
    eps = 1e-12
    H_norm = float(-np.sum(p * np.log(p + eps)) / np.log(len(p) + eps))
    low_band = fb <= 1.0
    sharpness_denom = max(float(np.median(pb)), 1e-6 * float(total), eps)
    return {
        f"dom_freq_{prefix}": float(fb[idx]),
        f"dom_power_ratio_{prefix}": float(dom_power / total),
        f"spec_entropy_{prefix}": H_norm,
        f"lowfreq_power_ratio_{prefix}": float(np.sum(pb[low_band]) / total) if np.any(low_band) else np.nan,
        f"peak_sharpness_{prefix}": float(dom_power / sharpness_denom),
    }


def feat_autocorr(signal: np.ndarray, axis_label: str) -> dict:
    """Lag (frames) and value of the autocorrelation peak of one coordinate signal, up to lag 200."""
    prefix = axis_label
    n = len(signal)
    if n < 10:
        return {f"autocorr_lag_{prefix}": np.nan, f"autocorr_strength_{prefix}": np.nan}
    x = signal - np.mean(signal)
    max_lag = min(200, n - 2)
    ac = np.correlate(x, x, mode="full")[n-1:n+max_lag]
    if ac[0] == 0:
        return {f"autocorr_lag_{prefix}": np.nan, f"autocorr_strength_{prefix}": np.nan}
    ac = ac / ac[0]
    lag = int(np.argmax(ac[1:]) + 1)
    return {f"autocorr_lag_{prefix}": float(lag), f"autocorr_strength_{prefix}": float(ac[lag])}


def feat_oscillation_regularity(x: np.ndarray, y: np.ndarray) -> dict:
    """How metronomic are the reversals?

    Wind plants oscillate at a stable period (low CV of inter-reversal
    intervals → regularity close to 1.0).
    Insects reverse erratically (high CV → regularity close to 0).
    """
    def _regularity(sig: np.ndarray) -> float:
        """1 − CV of the intervals between direction reversals, clipped to [0, 1]."""
        if len(sig) < 10:
            return np.nan
        reversals = np.where(np.diff(np.sign(np.diff(sig))) != 0)[0]
        if len(reversals) < 3:
            return np.nan
        intervals = np.diff(reversals).astype(float)
        mu = np.mean(intervals)
        if mu < 1e-6:
            return np.nan
        cv = np.std(intervals) / mu
        return float(np.clip(1.0 - cv, 0.0, 1.0))  # high = regular, low = erratic

    return {
        "oscillation_regularity_x": _regularity(x),
        "oscillation_regularity_y": _regularity(y),
    }


# d3_cross_axis: horizontal/vertical frequency coherence
def feat_xy_freq_coherence(x: np.ndarray, y: np.ndarray, fps: float) -> dict:
    """Cross-axis frequency coherence.

    Wind oscillates x and y at the same frequency (forced by wind direction).
    Insects move x and y semi-independently.
    Computed as 1 − normalised difference in dominant FFT frequency.
    """
    n = len(x)
    if n < 16:
        return {"xy_freq_coherence": np.nan}

    def _dom_freq(sig: np.ndarray) -> float:
        """Dominant FFT frequency (Hz) of a signal, above 0.1 Hz."""
        s = sig - np.mean(sig)
        X = np.fft.rfft(s * np.hanning(n))
        power = np.abs(X)**2
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        power[0] = 0.0
        band = freqs >= 0.1
        if not np.any(band):
            return np.nan
        return float(freqs[band][np.argmax(power[band])])

    fx = _dom_freq(x)
    fy = _dom_freq(y)
    if np.isnan(fx) or np.isnan(fy):
        return {"xy_freq_coherence": np.nan}
    max_f = max(fx, fy, 1e-6)
    coherence = 1.0 - abs(fx - fy) / max_f
    return {"xy_freq_coherence": float(np.clip(coherence, 0.0, 1.0))}


# d3_residual: raw-vs-smoothed residual micro-motion
def feat_wingbeat_residuals(x_raw: np.ndarray, y_raw: np.ndarray,
                             x_sm: np.ndarray, y_sm: np.ndarray,
                             fps: float) -> dict:
    """Wing-beat micro-oscillation from smoothing residuals.

    The difference between raw and smoothed coordinates captures
    sub-frame micro-motion.  For insects this contains real biological
    signal (wing beats, body oscillation); for non-insects it is pure
    tracker noise with no spectral structure.

      • residual_mean      : mean magnitude of (x_raw-sx, y_raw-sy)
      • residual_std       : variation in residual magnitude
      • residual_max       : maximum residual magnitude
    """
    # mean / std / max of the residual magnitude |ε(t)|
    nan_result = {"residual_mean": np.nan, "residual_std": np.nan,
                  "residual_max": np.nan}
    n = len(x_raw)
    if n < 8 or len(x_sm) != n:
        return nan_result

    rx = x_raw - x_sm
    ry = y_raw - y_sm
    rmag = np.sqrt(rx**2 + ry**2)   # scalar residual magnitude per frame

    return {
        "residual_mean": float(np.mean(rmag)),
        "residual_std":  float(np.std(rmag)),
        "residual_max":  float(np.max(rmag)),
    }


def feat_wingbeat_aliasing(x_raw: np.ndarray, y_raw: np.ndarray,
                            x_sm: np.ndarray, y_sm: np.ndarray,
                            fps: float) -> dict:
    """D3 residual wingbeat-aliasing features.

    At 25–30 fps, wingbeat frequencies (50–250 Hz) alias into the
    5–15 Hz band of the raw-minus-smoothed residual signal.

    aliased_wingbeat_power_frac: power in [5, 15] Hz band / total residual power
    residual_autocorr_peak: max autocorrelation of |r_i| at lags 1–10 frames
                            (non-trivial peak → biological periodicity)
    """
    nan_r = {"aliased_wingbeat_power_frac": np.nan, "residual_autocorr_peak": np.nan}
    n = len(x_raw)
    if n < 16 or len(x_sm) != n:
        return nan_r

    rx   = x_raw - x_sm
    ry   = y_raw - y_sm
    rmag = np.sqrt(rx**2 + ry**2)
    rmag -= np.mean(rmag)

    if np.std(rmag) < 1e-9:
        return nan_r

    # ── FFT power in aliased wingbeat band ───────────────────────────────────
    psd  = np.abs(np.fft.rfft(rmag * np.hanning(n)))**2
    frq  = np.fft.rfftfreq(n, d=1.0 / fps)
    psd[0] = 0
    total_power = np.sum(psd)
    if total_power > 0:
        band = (frq >= 5.0) & (frq <= min(15.0, fps / 2.0))  # 5–15 Hz, capped at the Nyquist frequency
        ali_power = float(np.sum(psd[band]) / total_power) if np.any(band) else np.nan
    else:
        ali_power = np.nan

    # ── Autocorrelation of residual magnitude at lags 1–10 ──────────────────
    max_lag = min(10, n - 2)
    if max_lag >= 1:
        var0 = np.var(rmag)
        if var0 > 1e-12:
            ac_vals = []
            for lag in range(1, max_lag + 1):
                # rmag is already zero-mean (subtracted above); no re-centering needed
                ac = float(np.mean(rmag[:-lag] * rmag[lag:]) / var0)
                ac_vals.append(ac)
            res_ac_peak = float(np.max(ac_vals))
        else:
            res_ac_peak = np.nan
    else:
        res_ac_peak = np.nan

    return {
        "aliased_wingbeat_power_frac": ali_power,
        "residual_autocorr_peak": res_ac_peak,
    }


# d3_hover_stab: mean-reversion and hover stabilisation
def feat_mean_reversion(x: np.ndarray, y: np.ndarray, fps: float) -> dict:
    """Mean reversion strength.

    Physics: a tethered plant has a restoring force toward its anchor.
    This creates a strong negative correlation between (position − mean)
    and velocity.  Insects have no such restoring force → correlation ≈ 0.

    Expected: wind plants ≈ −0.7, insects ≈ 0.
    """
    if len(x) < 5:
        return {"mean_reversion_x": np.nan, "mean_reversion_y": np.nan,
                "mean_reversion_strength": np.nan}
    dt = 1.0 / fps
    pos_dev_x = x[:-1] - np.mean(x)
    vel_x = np.diff(x) / dt
    pos_dev_y = y[:-1] - np.mean(y)
    vel_y = np.diff(y) / dt
    mr_x = _safe_corr(pos_dev_x, vel_x)
    mr_y = _safe_corr(pos_dev_y, vel_y)
    # Combined: average of x and y correlations
    combined = np.nanmean([mr_x, mr_y]) if not (np.isnan(mr_x) and np.isnan(mr_y)) else np.nan
    return {
        "mean_reversion_x": float(mr_x),
        "mean_reversion_y": float(mr_y),
        "mean_reversion_strength": float(combined),
    }


# ---------------------------------------------------------------------------
# Domain 4: Navigation, orientation and search behaviour (16)
# ---------------------------------------------------------------------------
# d4_directional: heading entropy, turn tendency, persistence, and drift
def feat_turning(dx: np.ndarray, dy: np.ndarray) -> dict:
    """Mean absolute and SD of turn angles, and normalised 16-bin heading entropy."""
    if len(dx) < 2:
        return {"turn_mean_abs": np.nan, "turn_std": np.nan, "heading_entropy": np.nan}
    headings = np.arctan2(dy, dx)
    dtheta = _wrap(np.diff(headings))
    # heading entropy (normalised)
    h2 = (headings + 2 * np.pi) % (2 * np.pi)
    hist, _ = np.histogram(h2, bins=16, range=(0, 2 * np.pi), density=True)
    hist = (hist + 1e-12) / np.sum(hist + 1e-12)
    H_norm = float(-np.sum(hist * np.log(hist)) / np.log(16))
    return {
        "turn_mean_abs": float(np.mean(np.abs(dtheta))),
        "turn_std": float(np.std(dtheta)),
        "heading_entropy": H_norm,
    }


def feat_turn_advanced(dx: np.ndarray, dy: np.ndarray) -> dict:
    """Advanced turning statistics.

    Beyond mean/std/entropy of turn angles:
      • turn_kurtosis      : heavy-tailed ↔ sharp reversals mixed with glides
      • uturn_rate         : fraction of turns > 150° — very insect-specific
      • forward_rate       : fraction of turns < 30° — sustained directed flight
      • crw_persistence    : lag-1 autocorrelation of turn angles
                             (CRW λ: +1=always same direction, −1=zigzag, 0=random)
    """
    nan_result = {"turn_kurtosis": np.nan, "uturn_rate": np.nan,
                  "forward_rate": np.nan, "crw_persistence": np.nan}
    if len(dx) < 3:
        return nan_result

    headings = np.arctan2(dy, dx)
    dtheta   = _wrap(np.diff(headings))
    if len(dtheta) < 2:
        return nan_result

    abs_dt = np.abs(dtheta)
    kurt   = float(sp_stats.kurtosis(dtheta, fisher=True))
    uturn  = float(np.mean(abs_dt > (5 * np.pi / 6)))   # > 150°
    fwd    = float(np.mean(abs_dt < (np.pi / 6)))        # < 30°

    # CRW persistence: mean(cos(dtheta)) is the standard estimator
    crw = float(np.mean(np.cos(dtheta)))

    return {"turn_kurtosis": kurt, "uturn_rate": uturn,
            "forward_rate": fwd, "crw_persistence": crw}


# d4_heading_occ: heading levels and dwell entropy
def feat_saccade_fixation(dx: np.ndarray, dy: np.ndarray, fps: float) -> dict:
    """D4 heading-occupancy and D5 saccade/fixation features.

    Many diurnal insects (flies, bees, hoverflies) alternate between:
      • Saccades : rapid angular deflections (|Δθ| > θ_sacc = 45°)
                   with transient neighbourhood |Δθ±1| < θ_fix = 10°
      • Fixations: runs of frames with |Δθ| < θ_fix

    Features
    --------
    saccade_rate            : saccades per second
    mean_inter_saccade_interval : mean gap between saccades (frames)
    saccade_amplitude_mean  : mean |Δθ| at saccade frames (radians)
    saccade_amplitude_cv    : CV of saccade amplitudes
    mean_fixation_length    : mean fixation run length (frames)
    fixation_length_cv      : CV of fixation run lengths
    n_heading_levels        : number of discrete heading clusters (~steps)
    heading_dwell_entropy   : entropy over heading clusters (normalised)
    """
    nan_r = {
        "saccade_rate": np.nan, "mean_inter_saccade_interval": np.nan,
        "saccade_amplitude_mean": np.nan, "saccade_amplitude_cv": np.nan,
        "mean_fixation_length": np.nan, "fixation_length_cv": np.nan,
        "n_heading_levels": np.nan, "heading_dwell_entropy": np.nan,
    }
    if len(dx) < 5:
        return nan_r

    headings = np.arctan2(dy, dx)
    dtheta   = np.abs(_wrap(np.diff(headings)))  # (n-2,)
    if len(dtheta) < 3:
        return nan_r

    THETA_SACC = np.radians(45)   # 45° saccade threshold
    THETA_FIX  = np.radians(10)

    # ── Saccade detection ────────────────────────────────────────────────────
    # A saccade at index i requires:
    #   dtheta[i] > θ_sacc  AND  dtheta[i-1] < θ_fix  AND  dtheta[i+1] < θ_fix
    saccade_mask = np.zeros(len(dtheta), dtype=bool)
    for i in range(1, len(dtheta) - 1):
        if dtheta[i] > THETA_SACC and dtheta[i-1] < THETA_FIX and dtheta[i+1] < THETA_FIX:
            saccade_mask[i] = True
    saccade_idx = np.where(saccade_mask)[0]

    n_frames = len(dtheta)
    dur_sec  = n_frames / fps

    saccade_rate = float(len(saccade_idx) / dur_sec) if dur_sec > 0 else np.nan

    if len(saccade_idx) >= 2:
        isi = float(np.mean(np.diff(saccade_idx)))
        amp = dtheta[saccade_idx]
        sac_amp_mean = float(np.mean(amp))
        sac_amp_cv   = float(np.std(amp) / (np.mean(amp) + 1e-9))
    elif len(saccade_idx) == 1:
        isi = np.nan
        sac_amp_mean = float(dtheta[saccade_idx[0]])
        sac_amp_cv   = np.nan
    else:
        isi, sac_amp_mean, sac_amp_cv = np.nan, np.nan, np.nan

    # ── Fixation run lengths ─────────────────────────────────────────────────
    is_fix = dtheta < THETA_FIX
    fix_lengths = []
    count = 0
    for v in is_fix:
        if v:
            count += 1
        elif count > 0:
            fix_lengths.append(count)
            count = 0
    if count > 0:
        fix_lengths.append(count)

    if len(fix_lengths) >= 2:
        mean_fix = float(np.mean(fix_lengths))
        fix_cv   = float(np.std(fix_lengths) / (np.mean(fix_lengths) + 1e-9))
    elif len(fix_lengths) == 1:
        mean_fix = float(fix_lengths[0])
        fix_cv   = np.nan
    else:
        mean_fix, fix_cv = np.nan, np.nan

    # ── Heading step-and-hold ─────────────────────────────────────────────────
    # Cluster headings into discrete levels using 8-bin histogram,
    # then count occupied bins (heading levels) and their entropy
    h_deg = (headings + np.pi) / (2 * np.pi)     # [0, 1)
    hist_h, _ = np.histogram(h_deg, bins=8, range=(0, 1))
    occ = int(np.sum(hist_h > 0))
    p_h = hist_h / (hist_h.sum() + 1e-12)
    p_h = p_h[p_h > 0]
    H_h = float(-np.sum(p_h * np.log(p_h)) / np.log(8)) if len(p_h) > 0 else np.nan

    return {
        "saccade_rate": saccade_rate,
        "mean_inter_saccade_interval": isi,
        "saccade_amplitude_mean": sac_amp_mean,
        "saccade_amplitude_cv": sac_amp_cv,
        "mean_fixation_length": mean_fix,
        "fixation_length_cv": fix_cv,
        "n_heading_levels": float(occ),
        "heading_dwell_entropy": H_h,
    }


# d4_zigzag: cast-surge, zigzag, decoupling, and reverse translation
def feat_cast_surge(dx: np.ndarray, dy: np.ndarray) -> dict:
    """D4 zigzag/casting search-behaviour features.

    Odour-tracking insects (moths, mosquitoes) produce alternating
    casting (zigzag perpendicular to wind) and surge (directed) phases.

    cast_surge_alternation_index:
        Fraction of consecutive heading triplets (θ_i−1, θ_i, θ_i+1) where
        the middle turn sign differs from both neighbours.
        High = frequent direction reversals (zigzag/cast).

    heading_zigzag_index:
        Normalised cross-correlation of cos(θ) and sin(θ) at lag-1.
        Anti-correlated lag-1 = heading alternates (zigzag structure).
    """
    nan_r = {"cast_surge_alternation_index": np.nan, "heading_zigzag_index": np.nan}
    if len(dx) < 5:
        return nan_r

    headings = np.arctan2(dy, dx)
    dtheta   = _wrap(np.diff(headings))  # signed turn angles

    # ── Cast-surge alternation ────────────────────────────────────────────────
    if len(dtheta) >= 3:
        signs  = np.sign(dtheta)
        # Middle sign differs from both neighbours = reversal in turn direction
        n_alt = 0
        for i in range(1, len(signs) - 1):
            if signs[i] != signs[i-1] and signs[i] != signs[i+1] and signs[i] != 0:
                n_alt += 1
        alt_idx = float(n_alt / (len(signs) - 2)) if len(signs) > 2 else np.nan
    else:
        alt_idx = np.nan

    # ── Heading zigzag index: negative lag-1 autocorrelation of turn angles ──
    # (A cos/sin cross-correlation would be NaN when cos(θ) is constant, e.g. a symmetric zigzag.)
    # Lag-1 autocorrelation of signed dθ: negative = turn sign alternates (zigzag).
    # We flip the sign so high value = strong zigzag.
    if len(dtheta) >= 4:
        dt_dm = dtheta - np.mean(dtheta)
        var_dt = float(np.var(dtheta))
        if var_dt > 1e-9:
            ac1 = float(np.mean(dt_dm[:-1] * dt_dm[1:]) / var_dt)
            zz_idx = float(-ac1)   # positive = sign-alternating zigzag
        else:
            zz_idx = np.nan
    else:
        zz_idx = np.nan

    return {
        "cast_surge_alternation_index": alt_idx,
        "heading_zigzag_index": zz_idx,
    }


def feat_heading_decoupling(x: np.ndarray, y: np.ndarray,
                              dx: np.ndarray, dy: np.ndarray) -> dict:
    """D4 heading-translation decoupling features.

    Insects can fly sideways / backwards; passive objects cannot.

    heading_translation_decoupling:
        1 − mean(cos(θ_instant − θ_coarse)), where θ_coarse is the heading of a
        5-frame moving average of the velocity vector. 0 = instantaneous heading
        tracks the smoothed heading (passive drift); higher = active decoupling.
        Rotation-invariant by construction.

    reverse_translation_fraction:
        Fraction of frames where the instantaneous velocity direction
        is anti-parallel (>120° from) the smoothed heading over the
        previous 5 frames. Indicates active course correction.
    """
    nan_r = {"heading_translation_decoupling": np.nan,
             "reverse_translation_fraction": np.nan}
    n = len(x)
    if n < 12 or len(dx) < 5:
        return nan_r

    headings = np.arctan2(dy, dx)           # instantaneous movement direction
    dtheta   = _wrap(np.diff(headings))     # turn angles

    # ── Heading-translation decoupling: circular mean of alignment angle ──────
    # mean(cos(h_instant − h_coarse)) is rotation-invariant, unlike a Pearson
    # correlation of cos(h), which depends on the camera axes (NaN near 90°).
    if n >= 12 and len(dx) >= 6:
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)
        coarse_cos = np.convolve(cos_h, np.ones(5) / 5, mode='same')
        coarse_sin = np.convolve(sin_h, np.ones(5) / 5, mode='same')
        coarse_heading = np.arctan2(coarse_sin, coarse_cos)
        # mean cos(Δθ): 1 = perfectly aligned (passive), 0 = decoupled (active)
        mean_align = float(np.mean(np.cos(headings - coarse_heading)))
        decoupling = float(1.0 - mean_align)
    else:
        decoupling = np.nan

    # ── Reverse-translation fraction ─────────────────────────────────────────
    WIN = 5
    if n >= WIN + 2:
        instant_heading = np.arctan2(dy, dx)          # (n-1,)
        smooth_heading  = np.array([
            np.arctan2(np.mean(dy[max(0, i-WIN):i+1]),
                       np.mean(dx[max(0, i-WIN):i+1]))
            for i in range(len(dy))
        ])                                             # (n-1,)
        angle_diff = np.abs(_wrap(instant_heading[WIN:] - smooth_heading[WIN:]))
        rev_frac   = float(np.mean(angle_diff > (2 * np.pi / 3)))   # >120°
    else:
        rev_frac = np.nan

    return {
        "heading_translation_decoupling": decoupling,
        "reverse_translation_fraction": rev_frac,
    }


# d4_looping: rotation, circling, winding, and figure-eight structure
def feat_looping(x: np.ndarray, y: np.ndarray,
                 dx: np.ndarray, dy: np.ndarray) -> dict:
    """Looping and circular flight behaviour.

    Insects often fly in loops, figure-eights, or helices near hosts/flowers.

      • net_rotation       : total signed angular displacement / (2π)
                             ±1 = one full loop, 0 = balanced back-and-forth
      • circling_index     : fraction of consecutive steps with the same sign
                             of cross-product — sustained rotation
      • self_intersections : winding-number proxy = |net_rotation| (absolute
                             number of full turns). NOTE: this is a deterministic
                             function of net_rotation (its absolute value), not an
                             independent count of actual segment self-crossings.
    """
    nan_result = {"net_rotation": np.nan, "circling_index": np.nan,
                  "self_intersections": np.nan}
    n = len(x)
    if n < 5 or len(dx) < 2:
        return nan_result

    headings = np.arctan2(dy, dx)
    dtheta   = _wrap(np.diff(headings))

    # Signed winding; self_intersections takes the absolute value separately
    net_rot  = float(np.sum(dtheta) / (2 * np.pi))   # signed: +CW, −CCW dominant

    # Circling index: cross-product sign consistency
    cross = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]
    if len(cross) > 0:
        pos = np.sum(cross > 0)
        neg = np.sum(cross < 0)
        circ = float(max(pos, neg) / len(cross))
    else:
        circ = np.nan

    # Self-intersections proxy: winding number magnitude.
    # The absolute net rotation (in full turns) approximates how many loops
    # the trajectory makes. Actual segment intersection counting is O(n²)
    # in Python loops and too slow for batch processing.
    # |net_rotation| ≥ 0.5 implies at least one near-loop; ≥ 1 = full loop.
    winding_proxy = float(np.abs(net_rot))

    return {"net_rotation": net_rot, "circling_index": circ,
            "self_intersections": winding_proxy}


def feat_maneuvers(dx: np.ndarray, dy: np.ndarray,
                    x: np.ndarray, y: np.ndarray) -> dict:
    """D4 looping and D5 stereotyped maneuver features.

    max_persistent_curving_run:
        Longest run of consecutive same-sign turn angles exceeding 15°.
        ≥ 6 consecutive → orientation flight / courtship spiral.

    figure_eight_count:
        Count of sign changes in cumulative angular displacement
        that occur after at least a half-rotation in each direction.
        Detects figure-8 / lemniscate flight paths.
    """
    nan_r = {"max_persistent_curving_run": np.nan,
             "figure_eight_count": np.nan}
    if len(dx) < 4:
        return nan_r

    headings = np.arctan2(dy, dx)
    dtheta   = _wrap(np.diff(headings))

    # ── Persistent curving runs ───────────────────────────────────────────────
    CURVE_THR = np.radians(15)   # > 15° per frame
    signs = np.sign(dtheta)
    big   = np.abs(dtheta) > CURVE_THR
    max_run = 0
    cur_run = 0
    cur_sign = 0
    for s, b in zip(signs, big):
        if b and s != 0:
            if s == cur_sign:
                cur_run += 1
            else:
                cur_sign = s
                cur_run = 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
            cur_sign = 0

    # ── Figure-eight count ────────────────────────────────────────────────────
    # Track cumulative winding; count zero-crossings after ≥±0.5 turns
    cumsum = np.cumsum(dtheta)
    HALF_TURN = np.pi    # 0.5 full rotation
    fig8 = 0
    last_cross_val = cumsum[0]
    crossed_half = False
    for val in cumsum[1:]:
        delta = val - last_cross_val
        if abs(delta) >= HALF_TURN:
            crossed_half = True
        if crossed_half and np.sign(val) != np.sign(last_cross_val) and np.sign(last_cross_val) != 0:
            fig8 += 1
            last_cross_val = val
            crossed_half = False

    return {
        "max_persistent_curving_run": float(max_run),
        "figure_eight_count": float(fig8),
    }


# d4_bias: upward movement fraction
def feat_vertical(dy: np.ndarray) -> dict:
    """H — Vertical motion bias.

    In image coordinates y increases downward, so upward movement = dy < 0.
    Insects tend to fly upward; wind plants oscillate symmetrically.
    Expected: insects upward_fraction ≈ 0.52, plants ≈ 0.48.
    """
    if len(dy) == 0:
        return {"upward_fraction": np.nan}
    return {"upward_fraction": float(np.mean(dy < 0))}


# ---------------------------------------------------------------------------
# Domain 5: Sensorimotor control and maneuverability (20)
# ---------------------------------------------------------------------------
# d5_turning, d5_maneuver, d5_power_law, d5_levy, and d5_aero helpers
def feat_speed_at_turns(x: np.ndarray, y: np.ndarray, fps: float) -> dict:
    """Mean speed at genuine direction-change frames.

    A pendulum slows to nearly zero at its turning points.
    Insects maintain speed (or accelerate) through direction changes.

    Only speed minima with a heading change above 15° count, so straight
    tracks with varying speed do not register as turns.
    """
    if len(x) < 10:
        return {"speed_at_turn_points": np.nan}
    dx = np.diff(x); dy = np.diff(y)
    speed = np.sqrt(dx**2 + dy**2) * fps
    headings = np.arctan2(dy, dx)
    abs_dtheta = np.abs(_wrap(np.diff(headings)))   # (n-2,)
    # A genuine turn: local speed minimum AND heading change > 15°
    TURN_THR = np.radians(15)
    from scipy.signal import argrelmin
    minima_idx = argrelmin(speed, order=2)[0]        # indices into speed (n-1,)
    # abs_dtheta has one fewer element; align safely
    turn_idx = [i for i in minima_idx if i < len(abs_dtheta) and abs_dtheta[i] > TURN_THR]
    if not turn_idx:
        return {"speed_at_turn_points": np.nan}
    return {"speed_at_turn_points": float(np.mean(speed[turn_idx]))}


def feat_motor_control(x: np.ndarray, y: np.ndarray,
                        speed: np.ndarray, fps: float) -> dict:
    """D5 maneuverability and motor control features.

    (a) Speed-increment power law: log|δv| ~ β·log(v).
        β≈1 indicates proportional (muscle-driven) acceleration;
        β≈0 indicates constant-magnitude noise (passive drift).
        Features: power_law_slope (β), power_law_r2.

    (b) Min-turn-radius at high speed:
        r_i = 1 / κ_i, where κ_i is instantaneous curvature.
        At the speed-percentile-90 bin, insects achieve tighter radii
        (active aerodynamic control) than passive ballistic objects.
        Feature: min_turn_radius_at_max_speed.

    (c) Centripetal/tangential acceleration decomposition:
        Decompose a_i into a_∥ (tangential) and a_⊥ (centripetal).
        centripetal_fraction = mean(|a_⊥| / (|a| + ε))
        lateral_accel_burstiness = std(|a_⊥|) / (mean(|a_⊥|) + ε)
    """
    nan_r = {
        "power_law_slope": np.nan, "power_law_r2": np.nan,
        "min_turn_radius_at_max_speed": np.nan,
        "centripetal_fraction": np.nan, "lateral_accel_burstiness": np.nan,
    }
    n = len(x)
    if n < 10:
        return nan_r

    dx  = np.diff(x)
    dy  = np.diff(y)
    dt  = 1.0 / fps

    # ── velocity and speed at interior frames ────────────────────────────────
    vx = dx / dt                     # (n-1,)
    vy = dy / dt
    sp = np.sqrt(vx**2 + vy**2)     # = speed passed in but recomputed here

    # ── curvature κ via cross/norm formula ───────────────────────────────────
    # Need (n-2,) second derivatives
    if n < 6:
        kappa = np.ones(len(dx) - 1) * np.nan
    else:
        # Multiply by fps (divide by dt) to get acceleration d²x/dt²
        ax_ = np.diff(vx) * fps                    # (n-2,) — true acceleration px/s²
        ay_ = np.diff(vy) * fps
        vx_ = vx[:-1]
        vy_ = vy[:-1]
        sp_ = sp[:-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cross = np.abs(vx_ * ay_ - vy_ * ax_)
            denom = sp_**3
            kappa = np.where(denom > 1e-6, cross / denom, np.nan)  # units: 1/px

    # ── (a) Power-law regression: log|δv| ~ β·log(v) ───────────────────────
    # β≈1: proportional (muscle-driven) acceleration; β≈0: constant-magnitude noise
    if len(sp) >= 5:
        dv = np.abs(np.diff(sp))       # |speed increment| (n-2,)
        v_mid = sp[:-1]                # speed at start of each interval (n-2,)
        pl_mask = (dv > 1e-8) & (v_mid > 1e-8) & np.isfinite(dv) & np.isfinite(v_mid)
        if pl_mask.sum() >= 5:
            ldv = np.log(dv[pl_mask])
            lv  = np.log(v_mid[pl_mask])
            slope, _, r, _, _ = sp_stats.linregress(lv, ldv)
            r2 = float(r**2)
            power_law_slope = float(slope)
        else:
            power_law_slope, r2 = np.nan, np.nan
    else:
        power_law_slope, r2 = np.nan, np.nan

    # ── (b) Min turn radius at max speed ────────────────────────────────────
    # Radius of curvature = 1/κ
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radius = np.where(kappa > 1e-8, 1.0 / (kappa + 1e-12), np.nan)  # units: px
    if np.any(np.isfinite(radius)):
        p90_sp = np.percentile(sp, 90)
        high_speed_mask = (sp[:len(radius)] >= p90_sp) & np.isfinite(radius)
        if high_speed_mask.sum() >= 2:
            min_r = float(np.nanmin(radius[high_speed_mask]))
        else:
            min_r = float(np.nanmin(radius))
    else:
        min_r = np.nan

    # ── (c) Centripetal / tangential decomposition ───────────────────────────
    if n >= 6:
        ax_ = np.diff(vx)
        ay_ = np.diff(vy)
        a_mag = np.sqrt(ax_**2 + ay_**2)
        vx_n  = vx[:-1] / (sp[:-1] + 1e-9)
        vy_n  = vy[:-1] / (sp[:-1] + 1e-9)
        # tangential component magnitude
        a_par = np.abs(ax_ * vx_n + ay_ * vy_n)
        # centripetal = sqrt(|a|² - a_∥²)
        a_perp = np.sqrt(np.maximum(a_mag**2 - a_par**2, 0.0))
        valid = a_mag > 1e-9
        cent_frac = float(np.mean(a_perp[valid] / a_mag[valid])) if valid.sum() > 0 else np.nan
        lat_burst = float(np.std(a_perp) / (np.mean(a_perp) + 1e-9)) if len(a_perp) > 0 else np.nan
    else:
        cent_frac, lat_burst = np.nan, np.nan

    return {
        "power_law_slope": power_law_slope,
        "power_law_r2": r2,
        "min_turn_radius_at_max_speed": min_r,
        "centripetal_fraction": cent_frac,
        "lateral_accel_burstiness": lat_burst,
    }


def feat_levy_tail_advanced(x: np.ndarray, y: np.ndarray,
                             speed: np.ndarray, fps: float) -> dict:
    """D5 Levy-tail and D6 inter-burst burstiness features.

    Complements existing levy_mu (which uses the full step distribution via
    the Clauset ML estimator).  Here we use the Hill estimator on the top 20%
    of steps for robustness to the bulk distribution, and measure temporal
    burstiness of burst events.

    levy_tail_exponent: power-law exponent μ = 1 + 1/H_k (Hill estimator on
                        top 20% of steps).  Comparable to levy_mu: Lévy
                        walkers have 1 < μ < 3; random walks μ → ∞.
    levy_tail_r2: Pareto CCDF R² — goodness of tail fit (high = true power law)
    interburst_burstiness: B = (σ−μ)/(σ+μ) of inter-burst intervals.
        B→+1: heavy-tailed (insect) · B→−1: periodic (wind plant)
    """
    nan_r = {"levy_tail_exponent": np.nan, "levy_tail_r2": np.nan,
             "interburst_burstiness": np.nan}
    n = len(x)
    if n < 15:
        return nan_r

    steps = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    steps = steps[steps > 0]
    if len(steps) < 8:
        return nan_r

    # ── Hill estimator (top 20% of steps) ───────────────────────────────────
    k = max(2, int(0.2 * len(steps)))
    X_sorted = np.sort(steps)
    X_k = X_sorted[-(k + 1)] if len(X_sorted) > k else X_sorted[0]
    top_k = X_sorted[-k:]
    if X_k > 1e-8 and len(top_k) >= 2 and np.all(top_k > 1e-8):
        H_k = float(np.mean(np.log(top_k / (X_k + 1e-12))))
        if H_k > 1e-9:
            alpha_hat = 1.0 / H_k          # Pareto shape α: P(X>x) ∝ x^{-α}
            levy_exp  = float(1.0 + alpha_hat)   # μ = α + 1, consistent with levy_mu
        else:
            alpha_hat = levy_exp = np.nan
        # R² of Pareto fit: both CCDFs conditional on X > X_k, so scale by k not len(steps)
        if not np.isnan(levy_exp):
            empirical_ccdf = np.arange(k, 0, -1) / k
            fitted_ccdf    = (X_k / top_k) ** alpha_hat   # Pareto CCDF uses α
            ss_res  = float(np.sum((empirical_ccdf - fitted_ccdf) ** 2))
            ss_tot  = float(np.sum((empirical_ccdf - np.mean(empirical_ccdf)) ** 2))
            levy_r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        else:
            levy_r2 = np.nan
    else:
        levy_exp, levy_r2 = np.nan, np.nan

    # ── Inter-burst burstiness ────────────────────────────────────────────────
    # speed is in px/s; threshold converted from 3 px/frame
    burst_thr = max(3.0 * fps, float(np.percentile(speed, 75)))
    is_burst  = speed > burst_thr
    # Find inter-burst intervals (gap frames between burst runs)
    burst_ends   = np.where(np.diff(is_burst.astype(int)) == -1)[0]
    burst_starts = np.where(np.diff(is_burst.astype(int)) ==  1)[0]
    if len(burst_starts) >= 2 and len(burst_ends) >= 1:
        # Align: inter-burst = start[i+1] - end[i]
        ibi = []
        for i in range(len(burst_ends)):
            later_starts = burst_starts[burst_starts > burst_ends[i]]
            if len(later_starts) > 0:
                ibi.append(int(later_starts[0] - burst_ends[i]))
        if len(ibi) >= 3:
            mu_b  = float(np.mean(ibi))
            sig_b = float(np.std(ibi))
            B = float((sig_b - mu_b) / (sig_b + mu_b + 1e-9))
        else:
            B = np.nan
    else:
        B = np.nan

    return {
        "levy_tail_exponent": levy_exp,
        "levy_tail_r2": levy_r2,
        "interburst_burstiness": B,
    }


# ---------------------------------------------------------------------------
# Domain 6: Behavioural state organisation (14)
# ---------------------------------------------------------------------------
# d6_hover, d6_stopping, d6_landing, and d6_takeoff helpers
def feat_hover_burst(speed: np.ndarray, fps: float) -> dict:
    """Hover-burst transition count and durations.

    Insects alternate between hovering (slow inspection) and bursting
    (fast flight to next location).  Wind plants stay slow or drift
    steadily — few or no state transitions.

    Thresholds (speed is in px/s; thresholds converted from px/frame):
      hover  : speed < 0.5 px/frame  →  0.5 * fps  px/s
      burst  : speed > max(3.0 * fps, 75th-percentile speed)  px/s
               (unified with burst_fraction / interburst_burstiness so every
                "burst" feature uses one operational definition)
    """
    if len(speed) < 3:
        return {"hover_burst_transitions": np.nan, "mean_hover_duration": np.nan,
                "mean_burst_duration": np.nan}

    # Multiply by fps because speed is in px/s, not px/frame
    # Unified burst threshold: max(3 px/frame, Q75) — matches feat_stops_bursts
    # and feat_landing_kinematics so the "burst" state is defined consistently.
    hover_thr = 0.5 * fps
    burst_thr = max(3.0 * fps, float(np.percentile(speed, 75)))

    is_hover = speed < hover_thr
    is_burst = speed > burst_thr
    # Unified state: 0=neutral, 1=hover, 2=burst
    state = np.where(is_hover, 1, np.where(is_burst, 2, 0))
    transitions = int(np.sum(np.diff(state) != 0))

    # Run-length encoding for hover and burst durations
    def _run_lengths(mask: np.ndarray) -> list[int]:
        """Lengths of consecutive True runs in mask."""
        runs = []
        count = 0
        for v in mask:
            if v:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        return runs

    hover_runs = _run_lengths(is_hover)
    burst_runs = _run_lengths(is_burst)
    mean_hover = float(np.mean(hover_runs)) if hover_runs else 0.0
    mean_burst = float(np.mean(burst_runs)) if burst_runs else 0.0

    return {
        "hover_burst_transitions": float(transitions),
        "mean_hover_duration": mean_hover,
        "mean_burst_duration": mean_burst,
    }


def feat_landing_kinematics(x: np.ndarray, y: np.ndarray,
                              speed: np.ndarray, fps: float) -> dict:
    """D6 stopping, landing, and takeoff organisation features.

    Pollinators: decelerating approach → hover → contact → rapid takeoff.
    Wind plants: no such structure (long steady stops or continuous sway).

    Features
    --------
    n_decel_to_stop_events   : # times a >0.5s monotonic deceleration ends in stop
    max_decel_to_stop_duration: max duration (s) of such decelerations
    n_impulsive_starts       : # times rapid acceleration from near-stop
    max_takeoff_accel        : max speed gain (px/s²) in an impulsive start
    pre_stop_vertical_fraction: |dy| / |d| in last 5 frames before a stop
    stop_interval_count      : number of stop intervals (stop_fraction > 0)
    mean_stop_duration       : mean duration of stop intervals (frames)
    stop_duration_cv         : CV of stop interval durations
    """
    nan_r = {
        "n_decel_to_stop_events": np.nan, "max_decel_to_stop_duration": np.nan,
        "n_impulsive_starts": np.nan, "max_takeoff_accel": np.nan,
        "pre_stop_vertical_fraction": np.nan,
        "stop_interval_count": np.nan, "mean_stop_duration": np.nan,
        "stop_duration_cv": np.nan,
    }
    n = len(speed)
    if n < 8:
        return nan_r

    # Speed thresholds in px/s (speed = step * fps).
    STOP_THR  = 1.0 * fps   # 1 px/frame → px/s   (compared against speed)
    # The impulsive-start test compares an acceleration (acc, px/s²), so its
    # threshold must also be an acceleration. 3 px/frame² → 3 * fps² px/s². This
    # value corresponds exactly to the speed gaining 3 px/frame in a single frame.
    BURST_ACC_THR = 3.0 * fps * fps   # 3 px/frame² → px/s²  (compared against acc)
    MIN_DECEL_FRAMES = max(2, int(0.5 * fps))   # ≥ 0.5 s

    # ── Stop intervals ────────────────────────────────────────────────────────
    is_stop = speed < STOP_THR
    stop_runs, stop_starts = [], []
    count = 0
    start_i = None
    for i, v in enumerate(is_stop):
        if v:
            if count == 0:
                start_i = i
            count += 1
        elif count > 0:
            stop_runs.append(count)
            stop_starts.append(start_i)
            count = 0
    if count > 0:
        stop_runs.append(count)
        stop_starts.append(start_i)

    n_stops = len(stop_runs)
    if n_stops > 0:
        mean_stop  = float(np.mean(stop_runs))
        cv_stop    = float(np.std(stop_runs) / (np.mean(stop_runs) + 1e-9))
    else:
        mean_stop = 0.0
        cv_stop   = np.nan

    # ── Deceleration-to-stop events ──────────────────────────────────────────
    n_decel_events = 0
    max_decel_dur  = 0.0

    for ss in stop_starts:
        if ss < MIN_DECEL_FRAMES:
            continue
        # Walk backwards from stop start looking for monotonic deceleration
        seg = speed[max(0, ss - 30) : ss + 1]
        if len(seg) < MIN_DECEL_FRAMES:
            continue
        # Find longest monotonically-decreasing tail ending at stop
        dlen = 0
        for k in range(len(seg) - 2, -1, -1):
            if seg[k] >= seg[k + 1]:
                dlen += 1
            else:
                break
        if dlen >= MIN_DECEL_FRAMES:
            n_decel_events += 1
            max_decel_dur = max(max_decel_dur, dlen / fps)

    # ── Impulsive starts (rapid acceleration from stop) ───────────────────────
    n_starts   = 0
    max_tk_acc = 0.0

    for ss, slen in zip(stop_starts, stop_runs):
        after = ss + slen
        if after + 3 >= n:
            continue
        # Start the window one frame early (at the last stop frame, speed ≈ 0) so a
        # burst that begins exactly at the stop→move boundary is captured: its onset
        # acceleration lives in the transition between the last stop frame and the
        # first moving frame. after - 1 = ss + slen - 1 ≥ 0 always (slen ≥ 1).
        seg_after = speed[after - 1 : after + int(fps)]   # onset frame + 1 s window
        if len(seg_after) < 2:
            continue
        acc = np.diff(seg_after) * fps        # px/s² (acceleration)
        if np.any(acc > BURST_ACC_THR):
            n_starts += 1
            max_tk_acc = max(max_tk_acc, float(np.max(acc)))

    # ── Pre-stop vertical fraction ────────────────────────────────────────────
    pre_stop_vf = np.nan
    dy = np.diff(y)
    dx_ = np.diff(x)
    if len(stop_starts) > 0:
        vfracs = []
        for ss in stop_starts:
            win_end   = min(ss, len(dy))
            win_start = max(0, win_end - 5)
            seg_dy = dy[win_start:win_end]
            seg_dx = dx_[win_start:win_end]
            seg_d  = np.sqrt(seg_dx**2 + seg_dy**2)
            total_d = np.sum(seg_d)
            if total_d > 0:
                vfracs.append(float(np.sum(np.abs(seg_dy)) / total_d))
        pre_stop_vf = float(np.mean(vfracs)) if vfracs else np.nan

    return {
        "n_decel_to_stop_events": float(n_decel_events),
        "max_decel_to_stop_duration": float(max_decel_dur),
        "n_impulsive_starts": float(n_starts),
        "max_takeoff_accel": float(max_tk_acc),
        "pre_stop_vertical_fraction": pre_stop_vf,
        "stop_interval_count": float(n_stops),
        "mean_stop_duration": mean_stop,
        "stop_duration_cv": cv_stop,
    }


# ===========================================================================
# MASTER FEATURE EXTRACTION
# ===========================================================================

def extract_all_features(
    df: pd.DataFrame,
    fps: float,
    xcol: str = "sx",
    ycol: str = "sy",
    fcol: str = "nframe",
    xcol_raw: str = "x",
    ycol_raw: str = "y",
    filename: str = "",
) -> dict:
    """Extract all features from a single track DataFrame.

    Parameters
    ----------
    df                    : track DataFrame (must have xcol, ycol, fcol)
    fps                   : frames per second
    xcol, ycol            : smoothed coordinate columns (default: sx, sy)
    fcol                  : frame index column
    xcol_raw, ycol_raw    : raw coordinate columns for residual features
                            (x, y) — used by feat_wingbeat_residuals

    Returns
    -------
    dict containing 125 motion-derived features. No metadata-derived field
    (time-of-day, station ID, season/farm) is produced.
    """
    # Keep a reference to the full df for raw coordinate access
    df_full = df.copy()

    needed = [fcol, xcol, ycol]
    df = df[needed].dropna().sort_values(fcol).drop_duplicates(subset=[fcol])
    frames = df[fcol].to_numpy()
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    n = len(df)

    if n < 5:
        # Use the frame span, not the point count (consistent with feat_basic)
        dur_frames = int(frames[-1] - frames[0] + 1) if n >= 2 else n
        feats = {"n_points": n, "duration_frames": dur_frames, "duration_sec": dur_frames / fps}
        return _canonical_feature_order(feats)

    dx, dy = np.diff(x), np.diff(y)
    step = np.sqrt(dx**2 + dy**2)
    speed = step * fps

    # -----------------------------------------------------------------------
    # Formula helper pass
    # -----------------------------------------------------------------------
    # Helpers are named for calculation families. Some helpers return features
    # used by more than one biological subgroup, so we first collect all values
    # into a calculation pool and then assemble the canonical hierarchy below.
    calculated: dict = {}

    def collect(values: dict) -> None:
        """Add computed values to the shared calculation pool."""
        calculated.update(values)

    collect(feat_basic(frames, fps))
    collect(feat_trajectory(x, y))
    collect(feat_speed(speed))
    collect(feat_accel(speed, fps))
    collect(feat_stops_bursts(speed, fps))
    collect(feat_turning(dx, dy))
    collect(feat_spatial(x, y))
    collect(feat_spectral(x, fps, "x"))
    collect(feat_spectral(y, fps, "y"))
    collect(feat_autocorr(x, "x"))
    collect(feat_autocorr(y, "y"))
    collect(feat_mean_reversion(x, y, fps))
    collect(feat_hover_burst(speed, fps))
    collect(feat_track_confinement(x, y))
    collect(feat_convex_hull(x, y))
    collect(feat_vertical(dy))
    collect(feat_oscillation_regularity(x, y))
    collect(feat_xy_freq_coherence(x, y, fps))
    collect(feat_speed_at_turns(x, y, fps))
    collect(feat_speed_shape(speed))
    collect(feat_msd_alpha(x, y))
    collect(feat_mean_abs_jerk(x, y, fps))
    collect(feat_multiscale_straightness(x, y))
    collect(feat_turn_advanced(dx, dy))
    collect(feat_looping(x, y, dx, dy))
    collect(feat_levy_steps(x, y))
    collect(feat_spatial_entropy(x, y))
    collect(feat_kinematic_effort(speed, x, y, fps))

    # Wingbeat residual helpers re-access raw columns from df_full, aligned to
    # the same frame indices as the smoothed coordinates.
    x_raw_aligned = None
    y_raw_aligned = None
    x_sm_aligned  = None
    y_sm_aligned  = None

    if xcol_raw in df_full.columns and ycol_raw in df_full.columns:
        df_raw = df_full[[fcol, xcol_raw, ycol_raw]].dropna() \
                        .sort_values(fcol).drop_duplicates(subset=[fcol])
        common = np.intersect1d(df[fcol].to_numpy(), df_raw[fcol].to_numpy())
        if len(common) >= 8:
            mask_sm  = np.isin(df[fcol].to_numpy(),  common)
            x_raw_aligned = df_raw.loc[df_raw[fcol].isin(common), xcol_raw].to_numpy(dtype=float)
            y_raw_aligned = df_raw.loc[df_raw[fcol].isin(common), ycol_raw].to_numpy(dtype=float)
            x_sm_aligned  = x[mask_sm]
            y_sm_aligned  = y[mask_sm]
            collect(feat_wingbeat_residuals(x_raw_aligned, y_raw_aligned,
                                             x_sm_aligned, y_sm_aligned, fps))
        else:
            collect({"residual_mean": np.nan, "residual_std": np.nan,
                     "residual_max": np.nan})
    else:
        collect({"residual_mean": np.nan, "residual_std": np.nan,
                 "residual_max": np.nan})

    collect(feat_motor_control(x, y, speed, fps))
    collect(feat_saccade_fixation(dx, dy, fps))
    collect(feat_levy_tail_advanced(x, y, speed, fps))
    if x_raw_aligned is not None and len(x_raw_aligned) >= 16:
        collect(feat_wingbeat_aliasing(x_raw_aligned, y_raw_aligned,
                                        x_sm_aligned, y_sm_aligned, fps))
    else:
        collect({"aliased_wingbeat_power_frac": np.nan,
                 "residual_autocorr_peak": np.nan})
    collect(feat_landing_kinematics(x, y, speed, fps))
    collect(feat_cast_surge(dx, dy))
    collect(feat_heading_decoupling(x, y, dx, dy))
    collect(feat_aerodynamic_envelope(x, y, speed, fps))
    collect(feat_maneuvers(dx, dy, x, y))
    collect(feat_robustness(x, y, speed, fps))

    missing_calculations = [name for name in ALL_FEATURES if name not in calculated]
    if missing_calculations:
        missing = ", ".join(missing_calculations)
        raise RuntimeError(f"Feature calculation did not produce: {missing}")

    # --- Metadata-derived fields intentionally excluded ---
    # time_of_day_bucket, station ID and season/farm are NOT used as features.
    # The classifier must rely only on intrinsic motion, not recording context.

    # -----------------------------------------------------------------------
    # Canonical six-domain assembly
    # -----------------------------------------------------------------------
    feats: dict = {}

    def add_group(group_name: str) -> None:
        """Copy one subgroup's features from the pool into the output, in canonical order."""
        for feature_name in FEATURE_GROUPS[group_name]:
            feats[feature_name] = calculated[feature_name]

    # Domain 1: Fundamental locomotion kinematics (26)
    # Track support, speed summaries, acceleration/effort, speed distribution,
    # and high-intensity burst or spike behaviour.
    add_group("d1_duration")
    add_group("d1_speed")
    add_group("d1_dynamic")
    add_group("d1_distributional")
    add_group("d1_intensity")

    # Domain 2: Spatial occupancy and path geometry (19)
    # Path directness, occupied region, exploratory range use, diffusion-like
    # displacement scaling, and Benhamou sinuosity.
    add_group("d2_path_geometry")
    add_group("d2_spatial_spread")
    add_group("d2_exploration")
    add_group("d2_diffusion")
    add_group("d2_sinuosity")

    # Domain 3: Flight rhythm, oscillation and biomechanics (30)
    # Axis-wise oscillation, cross-axis frequency agreement, global speed
    # spectrum, raw-vs-smoothed residual micro-motion, and hover stabilisation.
    add_group("d3_h_oscillation")
    add_group("d3_v_oscillation")
    add_group("d3_cross_axis")
    add_group("d3_spectral")
    add_group("d3_residual")
    add_group("d3_hover_stab")

    # Domain 4: Navigation, orientation and search behaviour (16)
    # Directional persistence, heading occupancy, zigzag/casting patterns,
    # looping or circling structure, and image-upward movement bias.
    add_group("d4_directional")
    add_group("d4_heading_occ")
    add_group("d4_zigzag")
    add_group("d4_looping")
    add_group("d4_bias")

    # Domain 5: Sensorimotor control and maneuverability (20)
    # Turning kinematics, saccade/fixation structure, maneuverability,
    # speed-change scaling, Levy tail statistics, and aerodynamic proxy.
    add_group("d5_turning")
    add_group("d5_saccade")
    add_group("d5_fixation")
    add_group("d5_maneuver")
    add_group("d5_power_law")
    add_group("d5_levy")
    add_group("d5_aero")

    # Domain 6: Behavioural state organisation (14)
    # Hover-burst cycling, stopping behaviour, deceleration-to-stop events,
    # and impulsive takeoff-like starts.
    add_group("d6_hover")
    add_group("d6_stopping")
    add_group("d6_landing")
    add_group("d6_takeoff")

    return _canonical_feature_order(feats)


# ===========================================================================
# BATCH PIPELINE
# ===========================================================================

def build_feature_table(
    tracks_dir: Path,
    labels_csv: Path,
    fps: float,
    xcol: str = "sx",
    ycol: str = "sy",
    fcol: str = "nframe",
    file_col: str = "filepath",
    label_col: str = "label",
    sg_window: int = 11,
    sg_poly: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """Build the feature table.

    If ``xcol`` (default 'sx') is absent from a track CSV but raw 'x'/'y'
    columns are present, a Savitzky-Golay filter (window=sg_window,
    polyorder=sg_poly) is applied on-the-fly to produce smoothed coordinates.
    The raw 'x'/'y' are retained as the source for wingbeat-residual features.
    """
    labels = pd.read_csv(labels_csv)
    # Drop stray header row
    labels = labels[labels[label_col] != label_col].reset_index(drop=True)

    rows = []
    stats = {"total": len(labels), "processed": 0, "missing": 0, "error": 0, "no_cols": 0}

    for _, row in labels.iterrows():
        fname = Path(str(row[file_col])).name
        path = tracks_dir / fname
        if not path.exists():
            stats["missing"] += 1
            continue

        lbl_raw = str(row[label_col]).strip().lower()
        y_bin = {"insect": 1, "not_insect": 0, "non_insect": 0, "1": 1, "0": 0}.get(lbl_raw)
        if y_bin is None:
            continue

        try:
            df = pd.read_csv(path, sep=None, engine="python")
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            stats["error"] += 1
            continue

        # Auto-smooth: if smoothed columns are absent but raw x/y are present,
        # apply Savitzky-Golay on-the-fly (float precision, no rounding).
        if xcol not in df.columns and "x" in df.columns and "y" in df.columns:
            x_raw = df["x"].to_numpy(dtype=float)
            y_raw = df["y"].to_numpy(dtype=float)
            # Interpolate NaNs before smoothing
            idx = np.arange(len(x_raw))
            for arr in (x_raw, y_raw):
                nans = np.isnan(arr)
                if nans.any() and (~nans).any():
                    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
            sx, sy = smooth_coords(
                x_raw, y_raw,
                window_length=sg_window,
                polyorder=sg_poly,
            )
            df = df.copy()
            df["sx"] = sx
            df["sy"] = sy

        if not {xcol, ycol, fcol}.issubset(df.columns):
            stats["no_cols"] += 1
            continue

        feats = extract_all_features(
            df, fps=fps, xcol=xcol, ycol=ycol, fcol=fcol,
            xcol_raw="x", ycol_raw="y",
            filename=fname,
        )
        feats["label"] = y_bin
        feats["filename"] = fname
        rows.append(feats)
        stats["processed"] += 1

    return pd.DataFrame(rows), stats


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    p = argparse.ArgumentParser(
        description="Phase 1 — Extract the 125 motion features from track CSVs."
    )
    p.add_argument("--tracks-dir", required=True)
    p.add_argument("--labels-csv", required=True)
    p.add_argument("--fps", type=float, required=True, help="Frames per second.")
    p.add_argument("--output-csv", default="data/results/trait_results/track_features.csv")
    p.add_argument("--xcol", default="sx")
    p.add_argument("--ycol", default="sy")
    p.add_argument("--fcol", default="nframe")
    p.add_argument("--no-save", action="store_true", help="Skip saving output CSV.")
    # Savitzky-Golay on-the-fly smoothing (used when sx/sy absent from CSVs)
    p.add_argument("--sg-window", type=int, default=11,
                   help="SG filter window length (odd). Default: 11")
    p.add_argument("--sg-poly", type=int, default=2,
                   help="SG filter polynomial order. Default: 2")
    # Batch / checkpoint mode
    p.add_argument("--batch-size", type=int, default=0,
                   help="Process N tracks per call and save checkpoint. 0 = all at once.")
    p.add_argument("--ckpt-dir", default="data/results/trait_results/feature_ckpts",
                   help="Directory for per-batch checkpoint CSVs.")
    p.add_argument("--finalize", action="store_true",
                   help="Concatenate all checkpoints into --output-csv and exit.")
    return p.parse_args()


def main() -> None:
    """Extract features for all labelled tracks and write the feature CSV.

    With --batch-size, processes one batch per call and saves a checkpoint;
    --finalize merges the checkpoints into --output-csv.
    """
    args = parse_args()
    tracks_dir = Path(args.tracks_dir).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()

    if not tracks_dir.exists():
        raise SystemExit(f"Tracks directory not found: {tracks_dir}")
    if not labels_csv.exists():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")

    output_csv = Path(args.output_csv)

    # ── Finalize: merge all checkpoints ────────────────────────────────────
    if args.finalize:
        ckpt_dir = Path(args.ckpt_dir)
        ckpts = sorted(ckpt_dir.glob("batch_*.csv"))
        if not ckpts:
            raise SystemExit(f"No checkpoint files found in {ckpt_dir}")
        dfs = [pd.read_csv(p) for p in ckpts]
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["filename"])
        merged.to_csv(output_csv, index=False)
        n_feat = len([c for c in merged.columns if c not in ("label", "filename")])
        print(f"Merged {len(ckpts)} checkpoints -> {len(merged):,} tracks · {n_feat} features")
        print(f"Saved → {output_csv}")
        return

    # ── Batch / checkpoint mode ────────────────────────────────────────────
    if args.batch_size > 0:
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        labels = pd.read_csv(labels_csv)
        labels = labels[labels["label"] != "label"].reset_index(drop=True)

        # Find already-processed filenames
        done_files: set = set()
        for p in ckpt_dir.glob("batch_*.csv"):
            try:
                done_files.update(pd.read_csv(p, usecols=["filename"])["filename"].tolist())
            except Exception:
                pass

        pending = labels[~labels["filepath"].apply(lambda p: Path(p).name).isin(done_files)]
        total   = len(labels)
        n_done  = len(done_files)

        if len(pending) == 0:
            print(f"All {total:,} tracks already processed. Run --finalize to merge.")
            return

        batch = pending.iloc[: args.batch_size]
        print(f"  {n_done:,}/{total:,} done. Processing next {len(batch):,} tracks …")

        rows = []
        for _, row in batch.iterrows():
            fname = Path(str(row["filepath"])).name
            path = tracks_dir / fname
            if not path.exists():
                continue
            lbl_raw = str(row["label"]).strip().lower()
            y_bin = {"insect": 1, "not_insect": 0, "non_insect": 0, "1": 1, "0": 0}.get(lbl_raw)
            if y_bin is None:
                continue
            try:
                df = pd.read_csv(path, sep=None, engine="python")
                df.columns = [str(c).strip() for c in df.columns]
            except Exception:
                continue
            xcol, ycol, fcol = args.xcol, args.ycol, args.fcol
            if xcol not in df.columns and "x" in df.columns and "y" in df.columns:
                x_raw = df["x"].to_numpy(dtype=float)
                y_raw = df["y"].to_numpy(dtype=float)
                idx = np.arange(len(x_raw))
                for arr in (x_raw, y_raw):
                    nans = np.isnan(arr)
                    if nans.any() and (~nans).any():
                        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
                sx, sy = smooth_coords(x_raw, y_raw,
                                       window_length=args.sg_window, polyorder=args.sg_poly)
                df = df.copy()
                df["sx"] = sx
                df["sy"] = sy
            if not {xcol, ycol, fcol}.issubset(df.columns):
                continue
            feats = extract_all_features(
                df, fps=args.fps, xcol=xcol, ycol=ycol, fcol=fcol,
                xcol_raw="x", ycol_raw="y",
                filename=fname,
            )
            feats["label"] = y_bin
            feats["filename"] = fname
            rows.append(feats)

        if rows:
            batch_df = pd.DataFrame(rows)
            n_done_now = n_done + len(rows)
            ckpt_path = ckpt_dir / f"batch_{n_done_now:06d}.csv"
            batch_df.to_csv(ckpt_path, index=False)
            print(f"  Saved {len(rows):,} rows → {ckpt_path.name}")
            remaining = total - n_done_now
            if remaining > 0:
                print(f"  {remaining:,} tracks remaining — run again to continue.")
            else:
                print(f"  All tracks processed. Run --finalize to merge.")
        return

    # ── Standard (all-at-once) mode ────────────────────────────────────────
    print(f"Extracting features from {tracks_dir} …")
    print(f"SG filter: window={args.sg_window}, polyorder={args.sg_poly} "
          f"(applied on-the-fly if sx/sy absent)")
    features_df, stats = build_feature_table(
        tracks_dir=tracks_dir,
        labels_csv=labels_csv,
        fps=args.fps,
        xcol=args.xcol,
        ycol=args.ycol,
        fcol=args.fcol,
        sg_window=args.sg_window,
        sg_poly=args.sg_poly,
    )

    n_feat = len([c for c in features_df.columns if c not in ("label", "filename")])
    print(f"Tracks processed : {stats['processed']}")
    print(f"Missing files    : {stats['missing']}")
    print(f"Errors           : {stats['error']}")
    print(f"Features per track: {n_feat}")
    print(f"Class balance    : {(features_df['label'] == 1).sum()} insect / "
          f"{(features_df['label'] == 0).sum()} not_insect")

    if not args.no_save:
        out = Path(args.output_csv).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out, index=False)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
