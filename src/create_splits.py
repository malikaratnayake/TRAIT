"""
src/create_splits.py
====================
Step 1.3 — Create grouped, leakage-free cross-validation splits.

WHY this matters
----------------
The existing pipeline calls sklearn's train_test_split(stratify=y), which
splits at the individual track level.  Tracks from the same farm / station /
recording session therefore appear in both train and test — a form of data
leakage that inflates reported performance and invalidates grouped evaluation.

This script groups by farm_id + station_id so that every track from one
physical camera-station lands in exactly one fold.

Fold design
-----------
  5-fold StratifiedGroupKFold by default on group_key = farm_id + '_' + station_id,
  stratified by farm-location × class and selected by deterministic seed search.
  Optional legacy GroupKFold is available with --fold-strategy group.
  Fold 0–3  : standard train / val splits (used in cross-validation)
  Fold 4    : reserved held-out TEST set (never seen during model selection)

  Additionally, a cross-farm split is created:
    cross_farm_train : all Farm A tracks
    cross_farm_test  : all Farm B tracks
  (and the symmetric direction)
  These are labelled in the 'cross_farm_role' column.

Outputs
-------
  data/splits_folds.csv      — one row per track, fold assignment + metadata
  data/split_statistics.csv  — summary of each fold (counts, class balance)

Usage
-----
  python src/create_splits.py \\
      --labels-csv  data/track_classification.csv \\
      --output-dir  data \\
      [--n-folds 5] [--fold-strategy stratified_group] [--random-state 0]
      [--seed-search-trials 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.metadata import parse_dataframe


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------

def create_splits(
    labels_csv: Path,
    output_dir: Path,
    n_folds: int = 5,
    fold_strategy: str = "stratified_group",
    random_state: int = 0,
    seed_search_trials: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build grouped CV folds and cross-farm splits.

    Returns
    -------
    splits_df   : track-level DataFrame with fold and role columns
    stats_df    : fold-level summary DataFrame
    """
    print(f"Loading labels: {labels_csv}")
    labels = pd.read_csv(labels_csv)

    # Drop the stray header row (artifact of incremental CSV appending)
    labels = labels[labels["label"] != "label"].reset_index(drop=True)

    # Attach metadata from filenames
    labels = parse_dataframe(labels, filepath_col="filepath")

    # Drop the single unparseable row (UNKNOWN farm)
    unknown_mask = labels["farm_id"] == "UNKNOWN"
    if unknown_mask.any():
        print(f"  Dropping {unknown_mask.sum()} row(s) with unparseable filenames.")
        labels = labels[~unknown_mask].reset_index(drop=True)

    n_total = len(labels)
    print(f"  {n_total} tracks across {labels['group_key'].nunique()} station-groups")

    # -----------------------------------------------------------------------
    # 5-fold grouped split
    # -----------------------------------------------------------------------
    labels["fold"] = -1
    groups = labels["group_key"].to_numpy()
    X_dummy = np.zeros(n_total)
    y = _binary_labels(labels["label"])

    if fold_strategy == "stratified_group":
        # Stratify on site-location × class jointly. The previous binary
        # comparison against the first farm_id accidentally made the strata
        # "first location vs everything else"; factorizing all farm_id values
        # preserves all Farm A/B location strata.
        site_codes, site_levels = pd.factorize(
            labels["farm_id"].astype(str),
            sort=True,
        )
        y_strat = site_codes * 2 + y
        fold_stratification = "farm_id_class_seed_search_v1"
    elif fold_strategy == "group":
        splitter = GroupKFold(n_splits=n_folds)
        y_strat = y
        fold_stratification = "group_v1"
    else:
        raise ValueError(
            f"Unknown fold_strategy={fold_strategy!r}; "
            "expected 'stratified_group' or 'group'."
        )

    print(f"  Fold strategy: {fold_strategy}")
    if fold_strategy == "stratified_group":
        print(
            "  Stratification: farm_id × class "
            f"({len(site_levels) * 2} strata: {', '.join(site_levels)} × insect/not)"
        )
        last_seed = random_state + max(seed_search_trials, 1) - 1
        print(f"  Stratification seed search: {random_state}..{last_seed}")
        labels["fold"], random_state = _best_stratified_group_folds(
            labels=labels,
            y=y,
            y_strat=y_strat,
            groups=groups,
            X_dummy=X_dummy,
            n_folds=n_folds,
            random_state=random_state,
            seed_search_trials=seed_search_trials,
        )
        print(f"  Selected stratification seed: {random_state}")
    else:
        for fold_idx, (_, val_idx) in enumerate(
            splitter.split(X_dummy, y_strat, groups)
        ):
            labels.loc[val_idx, "fold"] = fold_idx

    # Fold 4 = held-out test (never used for model selection)
    labels["split_role"] = labels["fold"].apply(
        lambda f: "test" if f == (n_folds - 1) else "trainval"
    )

    # -----------------------------------------------------------------------
    # Cross-farm roles
    # -----------------------------------------------------------------------
    # Farm A and Farm B are pseudonymous site labels.
    # The cross-farm test is: train on Farm A, test on Farm B (and vice versa).
    labels["cross_farm_role"] = labels["farm_name"]   # 'Farm_A' or 'Farm_B'
    labels["fold_strategy"] = fold_strategy
    labels["fold_stratification"] = fold_stratification
    labels["fold_random_state"] = random_state if fold_strategy == "stratified_group" else pd.NA
    labels["fold_seed_search_trials"] = (
        max(int(seed_search_trials), 1) if fold_strategy == "stratified_group" else pd.NA
    )

    # -----------------------------------------------------------------------
    # Leakage validation
    # -----------------------------------------------------------------------
    _validate_no_leakage(labels, n_folds)

    # -----------------------------------------------------------------------
    # Save splits
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the columns we need; drop the raw date object if present
    out_cols = [
        "filepath", "label", "fold", "split_role", "cross_farm_role",
        "fold_strategy", "fold_stratification", "fold_random_state",
        "fold_seed_search_trials",
        "farm_name", "farm_id", "camera_unit", "location_num",
        "station_id", "station_num",
        "date_str", "group_key", "filename",
    ]
    out_cols = [c for c in out_cols if c in labels.columns]
    splits_df = labels[out_cols].copy()

    splits_path = output_dir / "splits_folds.csv"
    splits_df.to_csv(splits_path, index=False)
    print(f"Saved splits   → {splits_path}")

    # -----------------------------------------------------------------------
    # Fold statistics
    # -----------------------------------------------------------------------
    stats_df = _build_stats(splits_df, n_folds)
    stats_path = output_dir / "split_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved stats    → {stats_path}")

    _print_summary(splits_df, stats_df, n_folds)
    return splits_df, stats_df


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------

def _binary_labels(labels: pd.Series) -> np.ndarray:
    """Return binary labels for stratification, with explicit validation."""
    mapped = labels.astype(str).str.strip().str.lower().map({
        "insect": 1,
        "not_insect": 0,
        "non_insect": 0,
        "1": 1,
        "0": 0,
    })
    if mapped.isna().any():
        bad = sorted(labels[mapped.isna()].astype(str).unique())
        raise ValueError(f"Unrecognized label value(s): {bad[:5]}")
    return mapped.to_numpy(dtype=int)


def _best_stratified_group_folds(
    labels: pd.DataFrame,
    y: np.ndarray,
    y_strat: np.ndarray,
    groups: np.ndarray,
    X_dummy: np.ndarray,
    n_folds: int,
    random_state: int,
    seed_search_trials: int,
) -> tuple[np.ndarray, int]:
    """Choose the best deterministic StratifiedGroupKFold shuffle."""
    n_trials = max(int(seed_search_trials), 1)
    best_key: tuple[float, float, float] | None = None
    best_folds: np.ndarray | None = None
    best_seed = random_state

    for seed in range(random_state, random_state + n_trials):
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        folds = np.full(len(labels), -1, dtype=int)
        for fold_idx, (_, val_idx) in enumerate(
            splitter.split(X_dummy, y_strat, groups)
        ):
            folds[val_idx] = fold_idx

        key = _fold_balance_key(labels, y, folds, n_folds)
        if best_key is None or key < best_key:
            best_key = key
            best_folds = folds
            best_seed = seed

    if best_folds is None:
        raise RuntimeError("StratifiedGroupKFold did not produce any candidate folds.")
    return best_folds, best_seed


def _fold_balance_key(
    labels: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    n_folds: int,
) -> tuple[float, float, float]:
    """Score fold balance; lower is better."""
    n_total = len(labels)
    ideal_n = n_total / n_folds
    global_class_rate = y.mean()

    farm_codes, farm_levels = pd.factorize(labels["farm_name"].astype(str), sort=True)
    global_farm_props = (
        np.bincount(farm_codes, minlength=len(farm_levels)).astype(float) / n_total
    )

    losses = []
    for fold in range(n_folds):
        mask = folds == fold
        n_fold = int(mask.sum())
        if n_fold == 0:
            return (float("inf"), float("inf"), float("inf"))

        size_loss = abs(n_fold - ideal_n) / ideal_n
        class_loss = abs(y[mask].mean() - global_class_rate)
        farm_props = (
            np.bincount(farm_codes[mask], minlength=len(farm_levels)).astype(float)
            / n_fold
        )
        farm_loss = np.abs(farm_props - global_farm_props).sum() / 2.0
        losses.append(2.0 * size_loss + 3.0 * class_loss + 3.0 * farm_loss)

    return (max(losses), sum(losses), losses[-1])


def _validate_no_leakage(df: pd.DataFrame, n_folds: int) -> None:
    """Assert that no station-group spans more than one fold."""
    errors = []

    # Each group_key should map to exactly one fold
    group_folds = df.groupby("group_key")["fold"].nunique()
    multi_fold_groups = group_folds[group_folds > 1]
    if not multi_fold_groups.empty:
        errors.append(
            f"LEAKAGE: {len(multi_fold_groups)} station-groups span multiple folds:\n"
            + multi_fold_groups.to_string()
        )

    # No track should appear more than once
    dup_paths = df["filepath"].duplicated()
    if dup_paths.any():
        errors.append(f"LEAKAGE: {dup_paths.sum()} duplicate filepath(s) in splits.")

    # All folds should be assigned (no -1 remaining)
    unassigned = (df["fold"] == -1).sum()
    if unassigned:
        errors.append(f"LEAKAGE: {unassigned} tracks have fold=-1 (unassigned).")

    if errors:
        raise RuntimeError("\n".join(errors))

    print("  ✓ Leakage validation passed — no station-group spans multiple folds.")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _build_stats(splits_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    rows = []
    binary = pd.Series(_binary_labels(splits_df["label"]), index=splits_df.index)

    for fold in range(n_folds):
        mask = splits_df["fold"] == fold
        role = "test" if fold == (n_folds - 1) else "val"
        train_mask = (~mask) & (splits_df["split_role"] == "trainval")

        def _stats(m: pd.Series, label: str) -> dict:
            n = m.sum()
            n_ins = binary[m].sum()
            pct = 100 * n_ins / n if n else 0
            farms = splits_df.loc[m, "farm_id"].nunique()
            stations = splits_df.loc[m, "station_id"].nunique()
            return {
                "fold": fold,
                "partition": label,
                "n_tracks": n,
                "n_insect": int(n_ins),
                "n_not_insect": int(n - n_ins),
                "pct_insect": round(pct, 1),
                "n_farms": farms,
                "n_stations": stations,
            }

        rows.append(_stats(train_mask, f"fold{fold}_train"))
        rows.append(_stats(mask, f"fold{fold}_{role}"))

    # Cross-farm rows
    for role_label in ["Farm_A", "Farm_B"]:
        m = splits_df["cross_farm_role"] == role_label
        n = m.sum()
        n_ins = binary[m].sum()
        rows.append({
            "fold": -1,
            "partition": f"cross_farm_{role_label}",
            "n_tracks": n,
            "n_insect": int(n_ins),
            "n_not_insect": int(n - n_ins),
            "pct_insect": round(100 * n_ins / n if n else 0, 1),
            "n_farms": splits_df.loc[m, "farm_id"].nunique(),
            "n_stations": splits_df.loc[m, "station_id"].nunique(),
        })

    return pd.DataFrame(rows)


def _print_summary(splits_df: pd.DataFrame, stats_df: pd.DataFrame, n_folds: int) -> None:
    print()
    print("=" * 65)
    print("SPLIT SUMMARY")
    print("=" * 65)
    print(f"{'Partition':<28} {'Tracks':>7} {'Insect':>7} {'%Ins':>6} {'Farms':>6} {'Stations':>9}")
    print("-" * 65)
    for _, row in stats_df.iterrows():
        print(
            f"  {row['partition']:<26} {row['n_tracks']:>7,} {row['n_insect']:>7,} "
            f"{row['pct_insect']:>5.1f}% {row['n_farms']:>6} {row['n_stations']:>9}"
        )
    print("=" * 65)

    print("\nTrack distribution by farm × fold:")
    group_fold = splits_df.groupby(["farm_name", "fold"]).size().unstack(fill_value=0)
    print(group_fold.to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1.3 — Create grouped, leakage-free CV splits."
    )
    parser.add_argument(
        "--labels-csv", default="data/track_classification.csv",
        help="Labels CSV. Default: data/track_classification.csv",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Output directory for split CSVs. Default: data",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of grouped folds. Default: 5",
    )
    parser.add_argument(
        "--fold-strategy",
        choices=["stratified_group", "group"],
        default="stratified_group",
        help="Grouped split strategy. Default: stratified_group",
    )
    parser.add_argument(
        "--random-state", type=int, default=0,
        help="Random state for stratified_group. Ignored by group. Default: 0",
    )
    parser.add_argument(
        "--seed-search-trials", type=int, default=100,
        help=(
            "Number of deterministic StratifiedGroupKFold shuffles to score, "
            "starting at --random-state. Use 1 to disable search. Default: 100"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not labels_csv.exists():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")

    create_splits(
        labels_csv=labels_csv,
        output_dir=output_dir,
        n_folds=args.n_folds,
        fold_strategy=args.fold_strategy,
        random_state=args.random_state,
        seed_search_trials=args.seed_search_trials,
    )


if __name__ == "__main__":
    main()
