"""
src/models/data_loader.py
============================
Shared data-loading utilities for all pipeline phases.

Defines the 6-domain biologically-informed feature hierarchy (125 features,
with time_of_day_bucket excluded) and exposes helper functions used across
Phase 2–5 modules.

Revision (2026-06-02): two redundant features removed — drift_consistency
(identical to crw_persistence) and n_hairpin_reversals (identical event to
uturn_rate). Contract reduced 127 → 125.

Public API
----------
  ALL_FEATURES  : list[str]          — all 125 features in canonical order
  DOMAIN_GROUPS : dict[str, list]    — 6 biological domains
  FEATURE_GROUPS: dict[str, list]    — fine-grained biological subgroups (LOO ablation)
  DOMAIN_TO_FEATURES: dict[str, list] — alias: same as DOMAIN_GROUPS

  load_data(features_csv, splits_csv, audit_csv=None) -> (df, feature_cols, domain_groups)
  get_fold_split(df, fold, feature_cols)    -> (X_tv, y_tv, X_val, y_val)
  get_test_split(df, feature_cols)          -> (X_tr, y_tr, X_te, y_te)
  get_cross_farm_split(df, train_farm, feature_cols) -> (X_tr, y_tr, X_te, y_te)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Domain 1 — Fundamental Locomotion Kinematics  (26 features)
# The supplied six-domain specification lists the 23 dynamics features below.
# Track duration/support completes the 125-feature motion contract here.
# ---------------------------------------------------------------------------
_DOMAIN1_DURATION = ["n_points", "duration_frames", "duration_sec"]
_DOMAIN1_SPEED = ["mean_speed", "std_speed", "max_speed", "p90_speed", "cv_speed"]
_DOMAIN1_DYNAMIC = [
    "mean_accel", "std_accel", "max_accel", "mean_abs_jerk",
    "specific_kinetic_energy", "power_proxy", "accel_speed_envelope_slope",
]
_DOMAIN1_DISTRIBUTIONAL = [
    "speed_skewness", "speed_kurtosis",
    "speed_iqr_over_median", "step_cv", "step_heavy_tail",
]
_DOMAIN1_INTENSITY = [
    "burst_fraction", "speed_burst_fraction", "speed_ratio",
    "jerk_fraction", "accel_mad_spike_rate", "jerk_asymmetry",
]

# ---------------------------------------------------------------------------
# Domain 2 — Spatial Occupancy & Path Geometry  (19 features)
# ---------------------------------------------------------------------------
_DOMAIN2_PATH_GEOMETRY = [
    "path_length", "net_displacement", "straightness",
    "straight_short", "straight_long", "straight_scale_ratio",
]
_DOMAIN2_SPATIAL_SPREAD = [
    "bbox_area", "bbox_aspect", "convex_hull_area",
    "radius_gyration", "mean_dist_centroid",
]
_DOMAIN2_EXPLORATION = [
    "exploration_ratio", "track_confinement_ratio",
    "max_displacement_from_start", "return_to_start_rate",
    "spatial_entropy", "revisit_rate",
]
_DOMAIN2_DIFFUSION = ["msd_alpha"]
_DOMAIN2_SINUOSITY = ["benhamou_sinuosity"]

# ---------------------------------------------------------------------------
# Domain 3 — Flight Rhythm, Oscillation & Biomechanics  (30 features)
# ---------------------------------------------------------------------------
_DOMAIN3_H_OSC = [
    "dom_freq_x", "dom_power_ratio_x", "spec_entropy_x",
    "lowfreq_power_ratio_x", "peak_sharpness_x",
    "autocorr_lag_x", "autocorr_strength_x", "oscillation_regularity_x",
]
_DOMAIN3_V_OSC = [
    "dom_freq_y", "dom_power_ratio_y", "spec_entropy_y",
    "lowfreq_power_ratio_y", "peak_sharpness_y",
    "autocorr_lag_y", "autocorr_strength_y", "oscillation_regularity_y",
]
_DOMAIN3_CROSS = ["xy_freq_coherence"]
_DOMAIN3_SPECTRAL = [
    "spectral_entropy", "spectral_flatness",
    "low_band_psd_ratio", "spectral_peak_Q", "stft_stationarity_idx",
]
_DOMAIN3_RESIDUAL = [
    "residual_mean", "residual_std", "residual_max",
    "aliased_wingbeat_power_frac", "residual_autocorr_peak",
]
_DOMAIN3_HOVER_STAB = ["mean_reversion_x", "mean_reversion_y", "mean_reversion_strength"]

# ---------------------------------------------------------------------------
# Domain 4 — Navigation, Orientation & Search Behaviour  (16 features)
# ---------------------------------------------------------------------------
_DOMAIN4_DIRECTIONAL = [
    "heading_entropy", "forward_rate", "uturn_rate",
    "crw_persistence", "heading_decorr_time",
]
_DOMAIN4_HEADING_OCC = ["n_heading_levels", "heading_dwell_entropy"]
_DOMAIN4_ZIGZAG = [
    "cast_surge_alternation_index", "heading_zigzag_index",
    "heading_translation_decoupling", "reverse_translation_fraction",
]
_DOMAIN4_LOOPING = ["net_rotation", "circling_index", "self_intersections", "figure_eight_count"]
_DOMAIN4_BIAS = ["upward_fraction"]

# ---------------------------------------------------------------------------
# Domain 5 — Sensorimotor Control & Maneuverability  (20 features)
# ---------------------------------------------------------------------------
_DOMAIN5_TURNING = ["turn_mean_abs", "turn_std", "turn_kurtosis", "speed_at_turn_points"]
_DOMAIN5_SACCADE = [
    "saccade_rate", "mean_inter_saccade_interval",
    "saccade_amplitude_mean", "saccade_amplitude_cv",
]
_DOMAIN5_FIXATION = ["mean_fixation_length", "fixation_length_cv"]
_DOMAIN5_MANEUVER = [
    "min_turn_radius_at_max_speed", "centripetal_fraction",
    "lateral_accel_burstiness", "max_persistent_curving_run",
]
_DOMAIN5_POWER_LAW = ["power_law_slope", "power_law_r2"]
_DOMAIN5_LEVY = ["levy_mu", "levy_tail_exponent", "levy_tail_r2"]
_DOMAIN5_AERO = ["re_proxy"]

# ---------------------------------------------------------------------------
# Domain 6 — Behavioural State Organisation  (14 features)
# NOTE: time_of_day_bucket is intentionally excluded from all feature lists.
# ---------------------------------------------------------------------------
_DOMAIN6_HOVER = [
    "hover_burst_transitions", "hover_transition_rate", "mean_hover_duration",
    "mean_burst_duration", "interburst_burstiness",
]
_DOMAIN6_STOPPING = ["stop_fraction", "stop_interval_count", "mean_stop_duration", "stop_duration_cv"]
_DOMAIN6_LANDING = [
    "n_decel_to_stop_events", "max_decel_to_stop_duration", "pre_stop_vertical_fraction",
]
_DOMAIN6_TAKEOFF = ["n_impulsive_starts", "max_takeoff_accel"]

# ---------------------------------------------------------------------------
# Public mappings
# ---------------------------------------------------------------------------

DOMAIN_GROUPS: dict[str, list[str]] = {
    "Domain1": (
        _DOMAIN1_DURATION + _DOMAIN1_SPEED + _DOMAIN1_DYNAMIC
        + _DOMAIN1_DISTRIBUTIONAL + _DOMAIN1_INTENSITY
    ),
    "Domain2": (
        _DOMAIN2_PATH_GEOMETRY + _DOMAIN2_SPATIAL_SPREAD + _DOMAIN2_EXPLORATION
        + _DOMAIN2_DIFFUSION + _DOMAIN2_SINUOSITY
    ),
    "Domain3": (
        _DOMAIN3_H_OSC + _DOMAIN3_V_OSC + _DOMAIN3_CROSS
        + _DOMAIN3_SPECTRAL + _DOMAIN3_RESIDUAL + _DOMAIN3_HOVER_STAB
    ),
    "Domain4": (
        _DOMAIN4_DIRECTIONAL + _DOMAIN4_HEADING_OCC + _DOMAIN4_ZIGZAG
        + _DOMAIN4_LOOPING + _DOMAIN4_BIAS
    ),
    "Domain5": (
        _DOMAIN5_TURNING + _DOMAIN5_SACCADE + _DOMAIN5_FIXATION
        + _DOMAIN5_MANEUVER + _DOMAIN5_POWER_LAW + _DOMAIN5_LEVY + _DOMAIN5_AERO
    ),
    "Domain6": (
        _DOMAIN6_HOVER + _DOMAIN6_STOPPING + _DOMAIN6_LANDING + _DOMAIN6_TAKEOFF
    ),
}

# Fine-grained biological subgroups — used for leave-one-group-out (LOO) ablation
FEATURE_GROUPS: dict[str, list[str]] = {
    # ── Domain 1 ────────────────────────────────────────────────────────────
    "d1_duration":       _DOMAIN1_DURATION,
    "d1_speed":          _DOMAIN1_SPEED,
    "d1_dynamic":        _DOMAIN1_DYNAMIC,
    "d1_distributional": _DOMAIN1_DISTRIBUTIONAL,
    "d1_intensity":      _DOMAIN1_INTENSITY,
    # ── Domain 2 ────────────────────────────────────────────────────────────
    "d2_path_geometry":  _DOMAIN2_PATH_GEOMETRY,
    "d2_spatial_spread": _DOMAIN2_SPATIAL_SPREAD,
    "d2_exploration":    _DOMAIN2_EXPLORATION,
    "d2_diffusion":      _DOMAIN2_DIFFUSION,
    "d2_sinuosity":      _DOMAIN2_SINUOSITY,
    # ── Domain 3 ────────────────────────────────────────────────────────────
    "d3_h_oscillation":  _DOMAIN3_H_OSC,
    "d3_v_oscillation":  _DOMAIN3_V_OSC,
    "d3_cross_axis":     _DOMAIN3_CROSS,
    "d3_spectral":       _DOMAIN3_SPECTRAL,
    "d3_residual":       _DOMAIN3_RESIDUAL,
    "d3_hover_stab":     _DOMAIN3_HOVER_STAB,
    # ── Domain 4 ────────────────────────────────────────────────────────────
    "d4_directional":    _DOMAIN4_DIRECTIONAL,
    "d4_heading_occ":    _DOMAIN4_HEADING_OCC,
    "d4_zigzag":         _DOMAIN4_ZIGZAG,
    "d4_looping":        _DOMAIN4_LOOPING,
    "d4_bias":           _DOMAIN4_BIAS,
    # ── Domain 5 ────────────────────────────────────────────────────────────
    "d5_turning":        _DOMAIN5_TURNING,
    "d5_saccade":        _DOMAIN5_SACCADE,
    "d5_fixation":       _DOMAIN5_FIXATION,
    "d5_maneuver":       _DOMAIN5_MANEUVER,
    "d5_power_law":      _DOMAIN5_POWER_LAW,
    "d5_levy":           _DOMAIN5_LEVY,
    "d5_aero":           _DOMAIN5_AERO,
    # ── Domain 6 ────────────────────────────────────────────────────────────
    "d6_hover":          _DOMAIN6_HOVER,
    "d6_stopping":       _DOMAIN6_STOPPING,
    "d6_landing":        _DOMAIN6_LANDING,
    "d6_takeoff":        _DOMAIN6_TAKEOFF,
}

# Alias for consumers that need the domain-to-feature mapping explicitly.
DOMAIN_TO_FEATURES = DOMAIN_GROUPS

# Canonical ordered list of all 125 motion-derived model features.
# time_of_day_bucket is contextual metadata and intentionally excluded.
# Note: radius_of_gyration removed (orphan name — feat_spatial outputs radius_gyration).
ALL_FEATURES: list[str] = (
    DOMAIN_GROUPS["Domain1"]
    + DOMAIN_GROUPS["Domain2"]
    + DOMAIN_GROUPS["Domain3"]
    + DOMAIN_GROUPS["Domain4"]
    + DOMAIN_GROUPS["Domain5"]
    + DOMAIN_GROUPS["Domain6"]
)

assert len(ALL_FEATURES) == 125, (
    f"Six-domain biological hierarchy must contain 125 features, got {len(ALL_FEATURES)}."
)
assert len(ALL_FEATURES) == len(set(ALL_FEATURES)), (
    "Duplicate feature names in ALL_FEATURES!"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    features_csv: "Path | str",
    splits_csv:   "Path | str",
    audit_csv:    "Path | str | None" = None,
) -> "tuple[pd.DataFrame, list[str], dict[str, list[str]]]":
    """Load the merged feature + splits DataFrame.

    Parameters
    ----------
    features_csv : path to track_features.csv
    splits_csv   : path to splits_folds.csv
    audit_csv    : optional, path to data_audit_report.csv (filters bad tracks)

    Returns
    -------
    df           : merged DataFrame with split metadata and feature values
    feature_cols : list of feature column names present in df (subset of ALL_FEATURES)
    domain_groups: DOMAIN_GROUPS filtered to features present in df
    """
    features_csv = Path(features_csv)
    splits_csv   = Path(splits_csv)

    feat_df   = pd.read_csv(features_csv)
    splits_df = pd.read_csv(splits_csv)

    # Normalise join key
    for d in (feat_df, splits_df):
        if "filename" not in d.columns and "filepath" in d.columns:
            d["filename"] = d["filepath"].apply(lambda p: Path(str(p)).name)

    # Drop duplicate columns from feat_df that splits_df already provides
    # (e.g. 'label', 'filepath') to avoid _x/_y suffixes after merge.
    drop_from_feat = [c for c in feat_df.columns
                      if c in splits_df.columns and c != "filename"]
    feat_df = feat_df.drop(columns=drop_from_feat, errors="ignore")

    df = splits_df.merge(feat_df, on="filename", how="inner")

    # Binary label
    if "y" not in df.columns:
        if "label" in df.columns:
            df["y"] = (df["label"] == "insect").astype(int)
        elif "label_x" in df.columns:
            df["y"] = (df["label_x"] == "insect").astype(int)
            df["label"] = df["label_x"]
        else:
            raise ValueError("Neither 'y' nor 'label' column found in data.")

    # Farm identity (used for cross-farm experiments)
    if "cross_farm_role" not in df.columns:
        if "farm_name" in df.columns:
            df["cross_farm_role"] = df["farm_name"]
        elif "farm_id" in df.columns:
            def _map_farm(fid: str) -> str:
                fid = str(fid).upper()
                if fid.startswith("FARMB"):
                    return "Farm_B"
                if fid.startswith("FARMA"):
                    return "Farm_A"
                return "UNKNOWN"
            df["cross_farm_role"] = df["farm_id"].apply(_map_farm)
        else:
            df["cross_farm_role"] = "UNKNOWN"

    # Filter bad tracks if audit provided
    if audit_csv is not None:
        audit_path = Path(audit_csv)
        if audit_path.exists():
            audit = pd.read_csv(audit_path)
            if "any_flag" in audit.columns and "filename" in audit.columns:
                bad = set(audit.loc[audit["any_flag"], "filename"])
                before = len(df)
                df = df[~df["filename"].isin(bad)].copy()
                print(f"  Audit filter: removed {before - len(df)} flagged tracks")

    # Identify feature columns actually present in data
    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features not found in CSV "
              f"(first 5: {missing[:5]}). Proceeding with {len(feature_cols)}.")

    # Replace ±inf with NaN so imputation handles them uniformly.
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Drop features that are entirely NaN (cannot be imputed). This is a
    # structural decision based only on a column being globally empty, so it
    # introduces no train/test information leakage.
    all_nan = [c for c in feature_cols if df[c].isna().all()]
    if all_nan:
        print(f"  Dropping {len(all_nan)} all-NaN features: {all_nan[:5]}"
              f"{'…' if len(all_nan) > 5 else ''}")
        feature_cols = [c for c in feature_cols if c not in all_nan]

    # NOTE: NaNs are deliberately NOT imputed here. Median imputation is now
    # performed inside the split helpers (get_fold_split / get_test_split /
    # get_cross_farm_split) using *training-row* medians only, so the held-out
    # validation / test / cross-farm partitions never contribute to the
    # imputation statistics. See impute_train_eval().

    # Build domain groups restricted to present features
    domain_groups_present = {
        domain: [f for f in feats if f in df.columns]
        for domain, feats in DOMAIN_GROUPS.items()
    }

    df = df.reset_index(drop=True)
    return df, feature_cols, domain_groups_present


# ---------------------------------------------------------------------------
# Train-only imputation
# ---------------------------------------------------------------------------

def impute_train_eval(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """Median-impute NaNs using *training-row* medians only.

    The per-column median is computed from ``X_train`` and applied to both
    ``X_train`` and ``X_eval`` (validation / test / cross-farm). This prevents
    the held-out partition from leaking into the imputation statistics.
    Columns that are entirely NaN within the training rows fall back to 0.0.

    Both arrays are modified in place (they are fresh float copies produced by
    the split helpers) and also returned for convenience.
    """
    X_train = np.asarray(X_train, dtype="float32")
    X_eval = np.asarray(X_eval, dtype="float32")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype("float32")

    tr_nan = np.isnan(X_train)
    if tr_nan.any():
        X_train[tr_nan] = np.take(med, np.where(tr_nan)[1])
    ev_nan = np.isnan(X_eval)
    if ev_nan.any():
        X_eval[ev_nan] = np.take(med, np.where(ev_nan)[1])
    return X_train, X_eval


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def get_fold_split(
    df: pd.DataFrame,
    fold: int,
    feature_cols: list,
):
    """(X_train, y_train, X_val, y_val) for inner-fold CV on trainval rows.

    NaNs are median-imputed using training-fold medians only.
    """
    tv = df[df["split_role"] == "trainval"]
    val_mask = tv["fold"] == fold
    tr_mask  = ~val_mask

    X_tr  = tv.loc[tr_mask,  feature_cols].values.astype("float32")
    y_tr  = tv.loc[tr_mask,  "y"].values.astype(int)
    X_val = tv.loc[val_mask, feature_cols].values.astype("float32")
    y_val = tv.loc[val_mask, "y"].values.astype(int)
    X_tr, X_val = impute_train_eval(X_tr, X_val)
    return X_tr, y_tr, X_val, y_val


def get_test_split(
    df: pd.DataFrame,
    feature_cols: list,
):
    """(X_train, y_train, X_test, y_test) — trainval vs held-out test fold.

    NaNs are median-imputed using trainval medians only.
    """
    train_mask = df["split_role"] == "trainval"
    test_mask  = df["split_role"] == "test"

    X_tr = df.loc[train_mask, feature_cols].values.astype("float32")
    y_tr = df.loc[train_mask, "y"].values.astype(int)
    X_te = df.loc[test_mask,  feature_cols].values.astype("float32")
    y_te = df.loc[test_mask,  "y"].values.astype(int)
    X_tr, X_te = impute_train_eval(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te


def get_cross_farm_split(
    df: pd.DataFrame,
    train_farm: str,
    feature_cols: list,
):
    """(X_train, y_train, X_test, y_test) for cross-farm transfer.

    All rows from train_farm are used for training; all rows from the other
    farm are the test set. NaNs are median-imputed using the training farm's
    medians only.
    """
    train_mask = df["cross_farm_role"] == train_farm
    test_mask  = ~train_mask

    X_tr = df.loc[train_mask, feature_cols].values.astype("float32")
    y_tr = df.loc[train_mask, "y"].values.astype(int)
    X_te = df.loc[test_mask,  feature_cols].values.astype("float32")
    y_te = df.loc[test_mask,  "y"].values.astype(int)
    X_tr, X_te = impute_train_eval(X_tr, X_te)
    return X_tr, y_tr, X_te, y_te
