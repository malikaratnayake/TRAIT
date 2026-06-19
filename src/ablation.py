"""
src/ablation.py
=========================
Phase 3 — Feature-domain ablation and feature-importance analysis.

Studies
-------
  Phase 3a  Cumulative domain ablation
              Domain 1 → Domain 1-2 → Domain 1-3 → Domain 1-4 → Domain 1-5 → Domain 1-6
  Phase 3b  Leave-one-domain-out ablation
              Full model → exclude Domain 1 → ... → exclude Domain 6
  Phase 3c  Single-domain-only ablation
              Domain 1 only → Domain 2 only → ... → Domain 6 only
  Phase 3d  Leave-one-feature-group-out (LOO) ablation
              Drop one biological subgroup at a time, measure F1 drop.
  Phase 3e  Feature importance analysis
              XGBoost gain importance + permutation importance.

All ablations use XGBoost trained on trainval folds 0-3 and evaluated
on the held-out station-disjoint test set (fold 4).

Public API
----------
  run_cumulative_domain_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)
  run_leave_one_domain_out_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)
  run_single_domain_only_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)
  run_group_loo_ablation(df, feature_cols, xgb_params, out_dir)
  run_feature_importance(df, feature_cols, xgb_params, out_dir)
  run_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)  # runs all studies

Usage (CLI)
-----------
  python src/ablation.py \\
      --features-csv data/track_features.csv \\
      --splits-csv   data/splits_folds.csv \\
      --output-dir   data/results/six_domain_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import f1_score

from src.models.data_loader import (
    load_data, get_fold_split, get_test_split, DOMAIN_GROUPS, FEATURE_GROUPS,
)
from src.models.evaluate import (
    compute_metrics, bootstrap_ci, save_results,
)

SEED   = 42
N_BOOT = 500   # reduced for speed in ablation experiments

# Decision threshold for the FULL 125-feature configuration. When set from the
# Phase-2 XGBoost PR-AUC result, fixed-threshold ablations reuse the main model's
# operating point. Single-domain-only ablations use their own OOF threshold
# because their score scales can differ substantially from the full model.
_FULL_SET_THRESHOLD: "float | None" = None


# ---------------------------------------------------------------------------
# Shared XGBoost helper
# ---------------------------------------------------------------------------

def _params_from_xgb_result(path: Path) -> dict:
    """Load effective XGBoost params from a saved result/config JSON."""
    with open(path) as f:
        data = json.load(f)
    params = dict(data.get("best_params", data))
    selected_n = data.get("selected_n_estimators")
    if selected_n is not None:
        params["n_estimators"] = int(selected_n)
    return params


def _train_xgb(X_tr, y_tr, params, spw=None):
    import xgboost as xgb
    if spw is None:
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    params = dict(params)
    params.setdefault("n_jobs", int(os.environ.get("TRAIT_N_JOBS", "-1")))
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def _find_f1_threshold(y_true, y_score) -> float:
    """Decision threshold (0.05–0.95) that maximises F1 on the given scores."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f > best_f1:
            best_thr, best_f1 = float(thr), float(f)
    return round(best_thr, 3)


def _oof_threshold(df, keep_cols, feature_cols, xgb_params) -> float:
    """F1-optimal decision threshold from out-of-fold validation predictions.

    Mirrors the Phase-2 PR-AUC protocol: for each configuration, XGBoost is
    trained on the inner training folds and scored on the held-out validation
    fold; the pooled out-of-fold scores are used to choose the F1-optimal
    threshold. This replaces the fixed 0.5 default so every ablation row reports
    F1/precision/recall at a tuned, leakage-free operating point (the full
    125-feature row then matches the main comparison table).
    """
    keep_idx = [feature_cols.index(c) for c in keep_cols]
    oof_scores, oof_true = [], []
    for fold in range(4):
        X_tv, y_tv, X_val, y_val = get_fold_split(df, fold, feature_cols)
        model = _train_xgb(X_tv[:, keep_idx], y_tv, xgb_params)
        oof_scores.append(model.predict_proba(X_val[:, keep_idx])[:, 1])
        oof_true.append(y_val)
    return _find_f1_threshold(np.concatenate(oof_true), np.concatenate(oof_scores))


def _eval_subset(df, keep_cols, feature_cols, xgb_params, threshold_mode="fixed"):
    """Train XGBoost on keep_cols and evaluate on the held-out test set.

    threshold_mode controls the decision threshold:
      "fixed" — score at the deployed full-model operating point
                (_FULL_SET_THRESHOLD, e.g. 0.56). Every row shares one threshold,
                so F1/precision/recall changes reflect the feature set alone.
                Used for the cumulative-domain and leave-one-group-out studies,
                and makes the full 125-feature row match the main comparison
                table exactly. Falls back to OOF tuning if no deployed threshold
                is available.
      "oof"   — tune the threshold per configuration on out-of-fold validation
                F1. Used for the single-domain study, where each domain's score
                scale differs and a shared cut would be unfair to weak domains.
    """
    keep_idx = [feature_cols.index(c) for c in keep_cols]
    if threshold_mode == "fixed" and _FULL_SET_THRESHOLD is not None:
        threshold = _FULL_SET_THRESHOLD
        effective_threshold_mode = "fixed_full_model"
    else:
        threshold = _oof_threshold(df, keep_cols, feature_cols, xgb_params)
        effective_threshold_mode = "oof"
    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    X_tr_k = X_tr[:, keep_idx]
    X_te_k = X_te[:, keep_idx]
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model  = _train_xgb(X_tr_k, y_tr, xgb_params, spw)
    y_prob = model.predict_proba(X_te_k)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    m  = compute_metrics(y_te, y_pred, y_prob)
    m["threshold"] = round(float(threshold), 4)
    m["threshold_mode"] = effective_threshold_mode
    ci = bootstrap_ci(y_te, y_pred, y_prob, n_boot=N_BOOT, seed=SEED)
    return m, ci, model


# ---------------------------------------------------------------------------
# Phase 3a — Cumulative domain ablation
# ---------------------------------------------------------------------------

def run_cumulative_domain_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    domain_groups: dict[str, list[str]],
    xgb_params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Cumulative domain ablation: train with Domain1, then Domain1+2, ..., Domain1-6.

    This quantifies each domain's incremental contribution to F1 and PR-AUC,
    answering: *how much does each layer of biological feature engineering add?*
    """
    print("\n── Phase 3a  Cumulative domain ablation ───────────────────────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping cumulative domain ablation")
        return pd.DataFrame()

    domain_names = ["Domain1", "Domain2", "Domain3", "Domain4", "Domain5", "Domain6"]
    rows = []
    cumulative_cols: list[str] = []

    for domain in domain_names:
        domain_feats = domain_groups.get(domain, [])
        # Only include features actually present in data
        new_feats = [f for f in domain_feats if f in feature_cols]
        cumulative_cols = cumulative_cols + new_feats

        if not cumulative_cols:
            continue

        label = "+".join(domain_names[:domain_names.index(domain) + 1])
        n_feat = len(cumulative_cols)

        m, ci, _ = _eval_subset(df, cumulative_cols, feature_cols, xgb_params)

        print(f"  {label:<20}  {n_feat:>3} features  "
              f"F1={m['f1']:.4f} [{ci['f1']['lo']:.4f}–{ci['f1']['hi']:.4f}]  "
              f"PR-AUC={m.get('pr_auc', float('nan')):.4f}")

        rows.append({
            "label":      label,
            "domain":     domain,
            "n_features": n_feat,
            "f1":         m["f1"],
            "f1_lo":      ci["f1"]["lo"],
            "f1_hi":      ci["f1"]["hi"],
            "precision":  m["precision"],
            "recall":     m["recall"],
            "pr_auc":     m.get("pr_auc"),
            "mcc":        m.get("mcc"),
            "threshold":  m.get("threshold"),
            "threshold_mode": m.get("threshold_mode"),
        })

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "ablation" / "domain_ablation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    # Incremental F1 gain per domain
    print("\n  Incremental F1 gain per domain:")
    for i, row in enumerate(rows):
        prev_f1 = rows[i - 1]["f1"] if i > 0 else 0.0
        delta = row["f1"] - prev_f1
        print(f"    +{row['domain']:7s}  ΔF1={delta:+.4f}")

    print(f"\n  Saved → {out_path.name}")
    return result_df


# ---------------------------------------------------------------------------
# Phase 3b — Leave-one-domain-out ablation
# ---------------------------------------------------------------------------

def run_leave_one_domain_out_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    domain_groups: dict[str, list[str]],
    xgb_params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """Train on all features except one full domain per round."""
    print("\n── Phase 3b  Leave-one-domain-out ablation ────────────────────")

    domain_names = ["Domain1", "Domain2", "Domain3", "Domain4", "Domain5", "Domain6"]
    m_base, ci_base, _ = _eval_subset(df, feature_cols, feature_cols, xgb_params)
    base_f1 = m_base["f1"]
    base_pr_auc = m_base.get("pr_auc")
    rows = [{
        "excluded_domain": "NONE",
        "excluded_n_features": 0,
        "n_features": len(feature_cols),
        "f1": base_f1,
        "f1_lo": ci_base["f1"]["lo"],
        "f1_hi": ci_base["f1"]["hi"],
        "precision": m_base["precision"],
        "recall": m_base["recall"],
        "pr_auc": base_pr_auc,
        "roc_auc": m_base.get("roc_auc"),
        "mcc": m_base.get("mcc"),
        "tp": m_base["tp"],
        "fp": m_base["fp"],
        "tn": m_base["tn"],
        "fn": m_base["fn"],
        "n_total": m_base["n_total"],
        "n_positive": m_base["n_positive"],
        "threshold": m_base.get("threshold"),
        "threshold_mode": m_base.get("threshold_mode"),
        "f1_drop": 0.0,
        "pr_auc_drop": 0.0,
        "roc_auc_drop": 0.0,
    }]

    print(f"  {'Excluded':<12} {'Kept':>5} {'F1':>8} {'ΔF1':>8} {'PR-AUC':>8} {'ΔPR-AUC':>9}")
    print(f"  {'NONE':<12} {len(feature_cols):>5} {base_f1:>8.4f} {0.0:>+8.4f} "
          f"{base_pr_auc:>8.4f} {0.0:>+9.4f}")

    for domain in domain_names:
        excluded = {f for f in domain_groups.get(domain, []) if f in feature_cols}
        keep_cols = [f for f in feature_cols if f not in excluded]
        if not excluded:
            continue

        m, ci, _ = _eval_subset(df, keep_cols, feature_cols, xgb_params)
        f1_drop = base_f1 - m["f1"]
        pr_auc_drop = base_pr_auc - m["pr_auc"]
        roc_auc_drop = m_base["roc_auc"] - m["roc_auc"]
        print(f"  {domain:<12} {len(keep_cols):>5} {m['f1']:>8.4f} {f1_drop:>+8.4f} "
              f"{m['pr_auc']:>8.4f} {pr_auc_drop:>+9.4f}")
        rows.append({
            "excluded_domain": domain,
            "excluded_n_features": len(excluded),
            "n_features": len(keep_cols),
            "f1": m["f1"],
            "f1_lo": ci["f1"]["lo"],
            "f1_hi": ci["f1"]["hi"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "mcc": m.get("mcc"),
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "n_total": m["n_total"],
            "n_positive": m["n_positive"],
            "threshold": m.get("threshold"),
            "threshold_mode": m.get("threshold_mode"),
            "f1_drop": f1_drop,
            "pr_auc_drop": pr_auc_drop,
            "roc_auc_drop": roc_auc_drop,
        })

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "ablation" / "domain_leave_one_out_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path.name}")
    return result_df


# ---------------------------------------------------------------------------
# Phase 3c — Single-domain-only ablation
# ---------------------------------------------------------------------------

def run_single_domain_only_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    domain_groups: dict[str, list[str]],
    xgb_params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """Train one model per domain using features from only that domain."""
    print("\n── Phase 3c  Single-domain-only ablation ──────────────────────")

    domain_names = ["Domain1", "Domain2", "Domain3", "Domain4", "Domain5", "Domain6"]
    m_base, _, _ = _eval_subset(df, feature_cols, feature_cols, xgb_params)
    base_f1 = m_base["f1"]
    base_pr_auc = m_base["pr_auc"]
    base_roc_auc = m_base["roc_auc"]
    base_threshold = m_base.get("threshold")
    base_threshold_mode = m_base.get("threshold_mode")
    rows = []

    print(f"  Full reference ({len(feature_cols)} features): "
          f"F1={base_f1:.4f}  PR-AUC={base_pr_auc:.4f}")
    print(f"\n  {'Only domain':<12} {'Used':>5} {'F1':>8} {'Gap F1':>8} "
          f"{'PR-AUC':>8} {'Gap PR':>8}")

    for domain in domain_names:
        keep_cols = [f for f in domain_groups.get(domain, []) if f in feature_cols]
        if not keep_cols:
            continue

        # Single-domain models have differently-scaled scores, so each tunes its
        # own out-of-fold threshold rather than sharing the full-model cut.
        m, ci, _ = _eval_subset(df, keep_cols, feature_cols, xgb_params,
                                threshold_mode="oof")
        f1_gap = base_f1 - m["f1"]
        pr_auc_gap = base_pr_auc - m["pr_auc"]
        roc_auc_gap = base_roc_auc - m["roc_auc"]
        print(f"  {domain:<12} {len(keep_cols):>5} {m['f1']:>8.4f} {f1_gap:>+8.4f} "
              f"{m['pr_auc']:>8.4f} {pr_auc_gap:>+8.4f}")
        rows.append({
            "domain": domain,
            "n_features": len(keep_cols),
            "f1": m["f1"],
            "f1_lo": ci["f1"]["lo"],
            "f1_hi": ci["f1"]["hi"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "mcc": m["mcc"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "n_total": m["n_total"],
            "n_positive": m["n_positive"],
            "threshold": m.get("threshold"),
            "threshold_mode": m.get("threshold_mode"),
            "full_reference_f1": base_f1,
            "full_reference_pr_auc": base_pr_auc,
            "full_reference_roc_auc": base_roc_auc,
            "full_reference_threshold": base_threshold,
            "full_reference_threshold_mode": base_threshold_mode,
            "f1_gap_to_full": f1_gap,
            "pr_auc_gap_to_full": pr_auc_gap,
            "roc_auc_gap_to_full": roc_auc_gap,
        })

    result_df = pd.DataFrame(rows)
    out_path = out_dir / "ablation" / "domain_only_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path.name}")
    return result_df


# ---------------------------------------------------------------------------
# Phase 3d — Leave-one-feature-group-out (LOO) ablation
# ---------------------------------------------------------------------------

def run_group_loo_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """
    For each biological feature subgroup, retrain XGBoost with that group
    removed and measure the change in F1 ('contribution').

    A large positive contribution means that group is critical.
    A negative contribution means removing the group *improves* F1 —
    suggesting the features are noisy or redundant for this dataset.
    """
    print("\n── Phase 3d  Leave-one-feature-group-out (LOO) ablation ───────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping LOO ablation")
        return pd.DataFrame()

    # Baseline: all features
    m_base, ci_base, base_model = _eval_subset(df, feature_cols, feature_cols, xgb_params)
    base_f1 = m_base["f1"]
    print(f"  Full ({len(feature_cols)} features)  "
          f"F1={base_f1:.4f} [{ci_base['f1']['lo']:.4f}–{ci_base['f1']['hi']:.4f}]")

    rows = [{
        "group":        "FULL",
        "domain":       "—",
        "n_features":   len(feature_cols),
        "dropped":      0,
        "f1":           base_f1,
        "f1_lo":        ci_base["f1"]["lo"],
        "f1_hi":        ci_base["f1"]["hi"],
        "pr_auc":       m_base.get("pr_auc"),
        "threshold":    m_base.get("threshold"),
        "threshold_mode": m_base.get("threshold_mode"),
        "f1_drop":      0.0,
        "contribution": 0.0,
    }]

    for group_name, group_feats in FEATURE_GROUPS.items():
        drop_set  = set(group_feats)
        keep_cols = [c for c in feature_cols if c not in drop_set]
        n_dropped = len(feature_cols) - len(keep_cols)

        if n_dropped == 0:
            continue  # group not present in data

        # Identify the parent domain from subgroup names such as "d1_dynamic".
        domain_label = group_name.split("_")[0].upper()
        domain = f"Domain{domain_label[1]}" if len(domain_label) >= 2 else "?"

        m, ci, _ = _eval_subset(df, keep_cols, feature_cols, xgb_params)
        drop         = round(base_f1 - m["f1"], 4)
        contribution = round(drop, 4)   # positive = group is useful
        pct_loss     = round(100 * drop / base_f1, 2) if base_f1 > 0 else 0.0

        flag = ""
        if contribution < -0.005:
            flag = "  ← noisy / redundant"
        elif contribution > 0.02:
            flag = "  ← IMPORTANT"

        print(f"  Drop {group_name:<25} ({n_dropped:>2} feats)  "
              f"F1={m['f1']:.4f}  Δ={drop:+.4f} ({pct_loss:.1f}%){flag}")

        rows.append({
            "group":        group_name,
            "domain":       domain,
            "n_features":   len(keep_cols),
            "dropped":      n_dropped,
            "f1":           m["f1"],
            "f1_lo":        ci["f1"]["lo"],
            "f1_hi":        ci["f1"]["hi"],
            "pr_auc":       m.get("pr_auc"),
            "threshold":    m.get("threshold"),
            "threshold_mode": m.get("threshold_mode"),
            "f1_drop":      drop,
            "contribution": contribution,
            "pct_loss":     pct_loss,
        })

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("contribution", ascending=False)

    out_path = out_dir / "ablation" / "group_loo_ablation.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\n  Top 5 most contributing groups:")
    top5 = result_df[result_df["group"] != "FULL"].nlargest(5, "contribution")
    for _, r in top5.iterrows():
        print(f"    {r['group']:<25}  contribution={r['contribution']:+.4f}")

    print(f"\n  Groups with negative contribution (removal helps):")
    neg = result_df[(result_df["group"] != "FULL") & (result_df["contribution"] < 0)]
    for _, r in neg.iterrows():
        print(f"    {r['group']:<25}  contribution={r['contribution']:+.4f}")

    print(f"\n  Saved → {out_path.name}")
    save_results(result_df.to_dict(orient="records"), out_dir / "ablation", "group_loo_ablation")
    return result_df


# ---------------------------------------------------------------------------
# Phase 3e — Feature importance analysis
# ---------------------------------------------------------------------------

def run_feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Analyse feature importance via:
      (a) XGBoost gain-based importance
      (b) Permutation importance (F1 drop when a feature is shuffled)

    Also computes redundancy: features with high pairwise |Pearson r|
    are flagged as potentially redundant.
    """
    print("\n── Phase 3e  Feature importance analysis ──────────────────────")
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not installed — skipping")
        return pd.DataFrame()

    X_tr, y_tr, X_te, y_te = get_test_split(df, feature_cols)
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = _train_xgb(X_tr, y_tr, xgb_params, spw)

    # ── (a) XGBoost gain importance ──────────────────────────────────────────
    gain_imp = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": f, "xgb_gain": gain_imp.get(f"f{i}", 0.0)}
        for i, f in enumerate(feature_cols)
    ])

    # ── (b) Permutation importance ───────────────────────────────────────────
    if _FULL_SET_THRESHOLD is not None:
        threshold = float(_FULL_SET_THRESHOLD)
        threshold_mode = "fixed_full_model"
    else:
        threshold = float(_oof_threshold(df, feature_cols, feature_cols, xgb_params))
        threshold_mode = "oof"
    base_prob = model.predict_proba(X_te)[:, 1]
    base_pred = (base_prob >= threshold).astype(int)
    base_metrics = compute_metrics(y_te, base_pred, base_prob)
    base_f1 = base_metrics["f1"]
    base_pr_auc = base_metrics["pr_auc"]
    rng = np.random.default_rng(SEED)
    perm_rows = []
    for i, feat in enumerate(feature_cols):
        X_perm = X_te.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_prob = model.predict_proba(X_perm)[:, 1]
        perm_pred = (perm_prob >= threshold).astype(int)
        perm_metrics = compute_metrics(y_te, perm_pred, perm_prob)
        perm_rows.append({
            "feature": feat,
            "perm_f1_drop": round(base_f1 - perm_metrics["f1"], 5),
            "perm_pr_auc_drop": round(base_pr_auc - perm_metrics["pr_auc"], 5),
            "threshold": round(threshold, 4),
            "threshold_mode": threshold_mode,
        })

    perm_df = pd.DataFrame(perm_rows)
    imp_df = imp_df.merge(perm_df, on="feature")

    # Identify domain for each feature
    feat_to_domain = {}
    for domain, feats in DOMAIN_GROUPS.items():
        for f in feats:
            feat_to_domain[f] = domain
    imp_df["domain"] = imp_df["feature"].map(feat_to_domain).fillna("?")

    # Identify group for each feature
    feat_to_group = {}
    for grp, feats in FEATURE_GROUPS.items():
        for f in feats:
            feat_to_group[f] = grp
    imp_df["group"] = imp_df["feature"].map(feat_to_group).fillna("?")

    imp_df = imp_df.sort_values("perm_f1_drop", ascending=False)

    print(f"\n  Top 15 features by permutation importance (F1 drop):")
    for _, r in imp_df.head(15).iterrows():
        print(f"    {r['feature']:<40} {r['perm_f1_drop']:+.5f}  [{r['domain']}]")

    # ── Redundancy: high-correlation pairs ──────────────────────────────────
    sample = df[feature_cols].sample(min(1000, len(df)), random_state=SEED)
    corr = sample.corr().abs()
    redundant_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if r > 0.90:
                redundant_pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "pearson_r": round(float(r), 4),
                })
    redund_df = pd.DataFrame(
        redundant_pairs,
        columns=["feature_a", "feature_b", "pearson_r"],
    )
    if not redund_df.empty:
        redund_df = redund_df.sort_values("pearson_r", ascending=False)
    if len(redund_df):
        print(f"\n  Highly correlated pairs (|r| > 0.90): {len(redund_df)}")
        for _, r in redund_df.head(10).iterrows():
            print(f"    {r['feature_a']:<35} ↔  {r['feature_b']:<35}  r={r['pearson_r']:.3f}")

    # ── Biological-group contribution summary ────────────────────────────────
    group_imp = (
        imp_df.groupby("group")
        .agg(n_feats=("feature", "count"),
             mean_perm_drop=("perm_f1_drop", "mean"),
             total_perm_drop=("perm_f1_drop", "sum"))
        .sort_values("total_perm_drop", ascending=False)
        .reset_index()
    )
    print(f"\n  Feature-group importance summary:")
    for _, r in group_imp.iterrows():
        print(f"    {r['group']:<25} n={r['n_feats']:<3}  "
              f"total_drop={r['total_perm_drop']:+.4f}  "
              f"mean_drop={r['mean_perm_drop']:+.5f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir_abl = out_dir / "ablation"
    out_dir_abl.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(out_dir_abl / "feature_importance.csv", index=False)
    group_imp.to_csv(out_dir_abl / "group_importance_summary.csv", index=False)
    if len(redund_df):
        redund_df.to_csv(out_dir_abl / "redundant_feature_pairs.csv", index=False)

    # Save the feature-importance fit with the ablation outputs. The active
    # Phase 2 XGBoost model lives under prauc_protocol/xgboost_prauc/.
    model_dir = out_dir_abl
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "xgb_feature_importance_model.ubj"))
    with open(model_dir / "xgb_feature_importance.json", "w") as f:
        json.dump(imp_df.to_dict(orient="records"), f, indent=2)

    print(f"\n  Saved → feature_importance.csv, group_importance_summary.csv")
    return imp_df


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_ablation(
    df: pd.DataFrame,
    feature_cols: list[str],
    domain_groups: dict[str, list[str]],
    xgb_params: dict,
    out_dir: Path,
) -> dict:
    """Run all Phase 3 ablation studies and return combined results dict."""
    domain_df = run_cumulative_domain_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)
    domain_loo_df = run_leave_one_domain_out_ablation(
        df, feature_cols, domain_groups, xgb_params, out_dir
    )
    domain_only_df = run_single_domain_only_ablation(
        df, feature_cols, domain_groups, xgb_params, out_dir
    )
    loo_df = run_group_loo_ablation(df, feature_cols, xgb_params, out_dir)
    imp_df = run_feature_importance(df, feature_cols, xgb_params, out_dir)

    return {
        "domain_ablation": domain_df,
        "domain_loo_ablation": domain_loo_df,
        "domain_only_ablation": domain_only_df,
        "loo_ablation": loo_df,
        "feature_importance": imp_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 — Feature-domain ablation studies")
    p.add_argument("--features-csv",    default="data/track_features.csv")
    p.add_argument("--splits-csv",      default="data/splits_folds.csv")
    p.add_argument("--audit-csv",       default=None)
    p.add_argument("--output-dir",      default="data/results/six_domain_v1")
    p.add_argument("--full-threshold",  type=float, default=None,
                   help="Decision threshold for the full-feature row (defaults to the "
                        "Phase-2 XGBoost PR-AUC threshold so Tables 5/6 match Table 3).")
    p.add_argument("--xgb-params",      default=None,
                   help="JSON string of XGBoost params, e.g. '{\"max_depth\":5}'")
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("TRAIT_N_JOBS", "-1")),
                   help="Parallel jobs for XGBoost. Use 1 for final reproducible runs.")
    p.add_argument("--leave-one-domain-out-only", action="store_true",
                   help="Run only full-domain leave-one-out ablation")
    p.add_argument("--single-domain-only", action="store_true",
                   help="Run only single-domain classification ablation")
    return p.parse_args()


def main():
    args       = parse_args()
    out_dir    = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df, feature_cols, domain_groups = load_data(
        args.features_csv, args.splits_csv, args.audit_csv
    )
    print(f"  {len(df):,} tracks · {len(feature_cols)} features")

    if args.xgb_params:
        xgb_params = json.loads(args.xgb_params)
    else:
        xgb_res_path = out_dir / "prauc_protocol" / "xgboost_prauc" / "results.json"
        if not xgb_res_path.exists():
            raise SystemExit(
                f"Phase 2 XGBoost PR-AUC result not found: {xgb_res_path}. "
                "Run Phase 2 first or pass --xgb-params explicitly."
            )
        xgb_params = _params_from_xgb_result(xgb_res_path)
        print(f"  Loaded Phase 2 XGBoost PR-AUC params from {xgb_res_path}")
    xgb_params["n_jobs"] = int(args.n_jobs)

    # Anchor the full-feature ablation row to the main model's operating point.
    global _FULL_SET_THRESHOLD
    if args.full_threshold is not None:
        _FULL_SET_THRESHOLD = float(args.full_threshold)
    else:
        xgb_res_path = out_dir / "prauc_protocol" / "xgboost_prauc" / "results.json"
        if xgb_res_path.exists():
            try:
                _FULL_SET_THRESHOLD = float(json.load(open(xgb_res_path)).get("threshold"))
            except (TypeError, ValueError, json.JSONDecodeError):
                _FULL_SET_THRESHOLD = None
    if _FULL_SET_THRESHOLD is not None:
        print(f"  Full-feature row pinned to main model threshold = {_FULL_SET_THRESHOLD}")
    else:
        print("  No main threshold found — full-feature row will use its own OOF threshold")

    if args.leave_one_domain_out_only:
        run_leave_one_domain_out_ablation(
            df, feature_cols, domain_groups, xgb_params, out_dir
        )
    elif args.single_domain_only:
        run_single_domain_only_ablation(
            df, feature_cols, domain_groups, xgb_params, out_dir
        )
    else:
        run_ablation(df, feature_cols, domain_groups, xgb_params, out_dir)
    print("\n✓ Phase 3 ablation complete")


if __name__ == "__main__":
    main()
