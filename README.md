# TRAIT — Trajectory Recognition for Automated Insect Tagging

TRAIT is a reproducible pipeline for classifying insect trajectories recorded by camera traps. Given raw (x, y) centroid track CSV files, it applies Savitzky-Golay smoothing, extracts 125 biologically-informed motion features across six domains, and trains and evaluates multiple classifiers.

This repository accompanies the paper:

> **Seeing Insects Through Motion: Behaviour-Informed Trajectory Classification for Camera-Trap Pollinator Monitoring** — *[Authors anonymised for double-blind review]*, *WACV 2027*
> DOI: [to be added upon acceptance]

---

## Repository structure

```
TRAIT/
├── run_pipeline.sh                    # Original end-to-end pipeline driver (Phases 1–5)
├── run_pipeline_review.sh             # Camera-ready revision pipeline — mirrors
│                                       # run_pipeline.sh plus a sequence-baseline
│                                       # preprocessing-parity fix and two new
│                                       # baselines (Transformer, ConvTran). Use this
│                                       # one; it supersedes run_pipeline.sh.
├── preflight_check.sh                 # Verify all optional deps (torch/aeon/xgboost)
│                                       # are importable before a long run — missing
│                                       # ones are skipped silently by design, so run
│                                       # this first
├── data/                              # Dataset included in this bundle (see "Dataset")
│   ├── tracks/                        # 28,295 raw track CSVs (Farm A + Farm B)
│   └── track_classification.csv       # filepath, label (insect / not_insect)
└── src/
    ├── generate_features.py           # Phase 1: SG smoothing + 125-feature extraction
    ├── create_splits.py               # Phase 2: station-disjoint grouped CV splits
    ├── models/                        # Phase 3: model training and evaluation utilities
    │   ├── data_loader.py             # Feature loading, fold/cross-farm splitting
    │   ├── evaluate.py                # Metrics, bootstrap CI, latency measurement
    │   ├── majority_baseline.py       # Majority baseline trainer
    │   └── tabular_models.py          # LR, RF, XGBoost with PR-AUC selection
    ├── train_fcn.py                   # Phase 3: 1D-CNN baseline
    ├── train_rocket.py                # Phase 3: miniROCKET baseline
    ├── train_transformer.py           # Phase 3: Transformer encoder baseline
    ├── train_convtran.py              # Phase 3: ConvTran baseline (Foumani et al. 2024)
    ├── cross_farm.py                  # Phase 4: cross-farm generalisation
    ├── cross_farm_threshold_study.py  # Supplementary: cross-farm threshold sensitivity
    │                                   # (run manually)
    ├── convtran_batch_size_study.py   # Supplementary: ConvTran batch-size sensitivity
    │                                   # (pipeline step 3.4d via
    │                                   #  --convtran-batch-sweep, or standalone)
    ├── ablation.py                    # Phase 5: domain and group ablation studies
    ├── figures/
    │   ├── plot_pr_curves.py          # Phase 3: precision-recall curves figure
    │   └── plot_group_loo.py          # Phase 5: group leave-one-out figure
    └── utils/
        ├── smoothing.py               # Savitzky-Golay smoothing utilities
        └── metadata.py                # Track filename parser (anonymized site/station/date)
```

---

## Dataset

The dataset consists of 28,295 manually annotated 2D centroid trajectory recordings from two geographically separated sites. `Farm A` and `Farm B` are pseudonymous site labels; raw site identities are not included in this repository.

| Site   | Stations | Tracks |
|--------|----------|--------|
| Farm A | 117      | 18,821 |
| Farm B | 36       | 9,474  |
| **Total** | **153** | **28,295** |

Insect prevalence is 12.9% across the full dataset. Each track is a short CSV of `(nframe, x, y)` pixel detections produced by a camera-trap system.

### Availability

The full dataset is **included in this bundle** under `data/`, so the pipeline runs without any additional download:

```
data/
├── tracks/                  # One CSV per track (nframe, x, y)
└── track_classification.csv # filepath, label  (insect / not_insect)
```

It is also archived (CC BY 4.0) at the anonymised link below for public reference:

> https://figshare.com/s/03bd981a356bfc62584d

Track filenames follow the canonical format:

```
FarmA_L1_S10_YYMMDD_HHMMSS_TRACKID.csv
```

where `Farm{A|B}` is the pseudonymous site, `L{N}` the anonymized location, `S{N}` the trap station, and `TRACKID` a unique identifier.

---

## Feature hierarchy

TRAIT extracts 125 intrinsic motion features organised into six biological domains. No recording metadata (time of day, station ID, site identity) is used by any classifier.

| Domain | # Features | Description |
|--------|-----------|-------------|
| D1 | 26 | Fundamental locomotion kinematics (speed, acceleration, jerk) |
| D2 | 19 | Spatial occupancy and path geometry (hull area, MSD, sinuosity) |
| D3 | 30 | Flight rhythm, oscillation and biomechanics (spectral, residuals) |
| D4 | 16 | Navigation, orientation and search behaviour (turning, looping) |
| D5 | 20 | Sensorimotor control and manoeuvrability (power law, Lévy tail) |
| D6 | 14 | Behavioural state organisation (hover, stopping, takeoff) |

---

## Environment setup

### 1. Create the main `trait-env` environment

Python 3.10 or later is required. If `pip` reports that SciPy is only
available up to `1.13.1`, the active environment is using Python 3.9 or older;
recreate the environment with Python 3.10+.

```bash
python3.10 -m venv trait-env
source trait-env/bin/activate          # macOS / Linux
# trait-env\Scripts\activate           # Windows
python -m pip install -r requirements.txt
```

This installs all dependencies for Phases 1–5 (feature extraction, CV splits, LR / RF / XGBoost / 1D-CNN / miniROCKET / Transformer / ConvTran training, cross-farm generalisation, and ablation studies) into a single environment.

### 2. Activate before running the pipeline

```bash
source trait-env/bin/activate
bash run_pipeline_review.sh --fps 30
```

If PyTorch, aeon, or xgboost fail to import (e.g. on an unsupported platform),
the pipeline automatically **skips** the steps that depend on them (1D-CNN,
miniROCKET, Transformer, ConvTran; xgboost's absence also skips Phase 4 and
Phase 5) and continues — silently, by design, so a step is marked complete
even though no result was produced. Before a long or unattended run (e.g. a
cluster job), run the dependency preflight check first:

```bash
bash preflight_check.sh
```

It verifies numpy/scipy/pandas/scikit-learn/xgboost/matplotlib/torch/aeon/numba
are all importable and reports exactly which results you'd silently lose if
any are missing.

---

## Reproducing results

`run_pipeline_review.sh` is the current, maintained pipeline — it mirrors
`run_pipeline.sh` exactly (same five phases, same CLI, same checkpointing)
but adds a preprocessing-parity fix for the sequence baselines and two new
Phase 3 models (Transformer, ConvTran); see [Camera-ready revision
additions](#camera-ready-revision-additions) below. Use it unless you
specifically need the earlier `run_pipeline.sh` behaviour.

Run the full pipeline end-to-end:

```bash
bash run_pipeline_review.sh --fps 30
```

Run the final-report reproducibility mode:

```bash
bash run_pipeline_review.sh --fps 30 --test-run --run-name final_report
```

`--test-run` forces single-threaded RF/XGBoost/miniROCKET, deterministic
PyTorch settings, and CPU FCN/Transformer/ConvTran execution. It is slower,
but is intended for the canonical result table — it's also what keeps
latency figures for the deep-learning baselines comparable to XGBoost's
CPU-only numbers.

Resume from a specific phase (e.g. after Phase 2 splits already exist):

```bash
bash run_pipeline_review.sh --fps 30 --from-phase 3
```

Save results to a named folder:

```bash
bash run_pipeline_review.sh --fps 30 --run-name my_run
```

All checkpoints are written to `data/results/<run>/ckpts/`. Re-running the script automatically skips completed steps.

### Pipeline phases

**Phase 1 — Feature generation**

Applies Savitzky-Golay smoothing (window=11, poly=2) to raw (x, y) coordinates and extracts 125 motion features across six biological domains. Tracks shorter than 5 frames are discarded.

**Phase 2 — CV splits**

Creates station-disjoint 5-fold grouped splits (group = farm + location + station). `StratifiedGroupKFold` is the default, using farm-location × class strata and a deterministic seed search to choose a balanced grouped split; legacy `GroupKFold` can be selected with `--fold-strategy group`. Fold 4 is held out as the fixed test set and is never used for model selection or threshold tuning.

**Phase 3 — Model performance**

Trains and evaluates all classifiers on the held-out test set:

- Majority baseline (prevalence reference)
- Logistic Regression (L2-regularised)
- Random Forest (100 trees)
- XGBoost (depth=5, scale_pos_weight tuned for class imbalance)
- 1D-CNN (temporal FCN with PR-AUC early stopping)
- miniROCKET (fixed convolutional kernels + ridge classifier)
- Transformer (linear projection + learnable positional embedding + standard
  `nn.TransformerEncoder`, linear LR warmup + cosine decay)
- ConvTran (Foumani et al., DMKD 2024 — disjoint-CNN convolutional tokeniser,
  tAPE positional encoding, eRPE relative-position attention)

1D-CNN, miniROCKET, Transformer, and ConvTran all consume an identical
gap-filled, Savitzky-Golay-smoothed trajectory (`src/utils/smoothing.py`),
matching the preprocessing used to build the 125-feature set, so the
comparison isolates model architecture rather than input preprocessing.

Model selection and hyperparameter tuning use validation PR-AUC only; decision thresholds are tuned by validation F1. Output: `phase3_model_comparison.csv` and `figures/phase3_precision_recall_curves.png`.

**Phase 4 — Cross-farm generalisation**

Trains XGBoost on all tracks from one farm and evaluates on the other (A→B and B→A), with no domain adaptation or shared normalisation.

**Phase 5 — Ablation studies**

Three complementary ablation experiments, all using XGBoost on the held-out test set:

- *Cumulative domain ablation*: D1 → D1+D2 → D1+D2+D3 → … → D1–D6
- *Single-domain-only ablation*: each domain Di evaluated independently
- *Leave-one-group-out ablation*: each of 32 biological feature subgroups left out in turn

---

## Camera-ready revision additions

Two supplementary studies accompany the main results. Neither runs by
default, and neither modifies the reported numbers.

**`src/cross_farm_threshold_study.py`** — cross-farm decision-threshold
sensitivity. Phase 4's cross-farm PR-AUC is threshold-independent, so it
can't say whether the single fixed decision threshold (tuned on the pooled
training distribution) still lands at a sensible precision/recall point on
an unseen farm. This script retrains XGBoost with the same specification as
Phase 4, for each of the four evaluations (within-A, within-B, A→B, B→A),
and reports precision/recall/F1 at the deployed threshold alongside that
evaluation's own locally-optimal threshold.

```bash
python src/cross_farm_threshold_study.py \
    --features-csv data/results/<run>/track_features.csv \
    --output-dir   data/results/<run>
```

Output: `data/results/<run>/cross_farm_threshold_study/cross_farm_threshold_summary.csv`
(+ per-track predictions).

**ConvTran batch-size sensitivity** — `src/train_convtran.py` uses ConvTran's
own published architecture defaults but trains at `batch_size=128` rather
than the official recipe's 16, for parity with this project's other sequence
baselines. The sweep trains ConvTran at several batch sizes to check the
reported result is stable across that choice rather than an artefact of one
setting.

It is built into the pipeline as an **opt-in** step (3.4d) — omitted by
default, so a normal run is unaffected:

```bash
bash run_pipeline_review.sh --fps 30 --from-phase 3 \
    --convtran-batch-sweep "16 32 64 96 128 192"
```

The same sweep can also be run standalone, without the pipeline:

```bash
python src/convtran_batch_size_study.py \
    --batch-sizes 16 32 64 96 128 192 \
    --output-dir data/results/<run>
```

Output (either route): `data/results/<run>/convtran_batch_size_study/convtran_batch_size_summary.csv`.
The reported ConvTran result in `convtran_prauc/` is never touched — each
batch size writes to its own `bs_<N>/` subdirectory.

Each batch size is a full training run — smaller batch sizes take
substantially longer (more optimiser steps per epoch), so budget accordingly
at production settings (`max_len=512`, the full training set). The sweep
resumes automatically (skips batch sizes with existing results; pass
`--overwrite` to force a re-run) and logs each run to its own `train.log`
rather than aborting if one batch size fails.

A no-LR-warmup ablation of the Transformer encoder was also run during
development (`train_transformer.py --warmup-ratio 0`) to rule out
under-optimisation as an explanation for its result — PR-AUC 0.8068 vs.
0.8002 with warmup, well within each other's confidence interval. It is not
part of the automated pipeline or the reported results, but the
`--warmup-ratio` flag remains available on `train_transformer.py` for anyone
who wants to reproduce that check.

---

## Key outputs

| File | Description |
|------|-------------|
| `data/splits_folds.csv` | Station-disjoint fold assignments (generated by Phase 2) |
| `data/results/<run>/track_features.csv` | 125-feature matrix for all tracks |
| `data/results/<run>/phase3_model_comparison.csv` | Model comparison table (all 8 models) |
| `data/results/<run>/figures/phase3_precision_recall_curves.png` | PR curves figure |
| `data/results/<run>/transformer_warmup_prauc/transformer_results.csv` | Transformer result |
| `data/results/<run>/convtran_prauc/convtran_results.csv` | ConvTran result |
| `data/results/<run>/cross_farm/cross_farm_results.csv` | Cross-farm results (Phase 4) |
| `data/results/<run>/ablation/domain_ablation_results.csv` | Cumulative domain ablation |
| `data/results/<run>/ablation/domain_only_results.csv` | Single-domain-only ablation |
| `data/results/<run>/ablation/group_loo_ablation.csv` | Group leave-one-out ablation |
| `data/results/<run>/figures/group_loo_top_positive_negative.png` | Group LOO figure |
| `data/results/<run>/cross_farm_threshold_study/cross_farm_threshold_summary.csv` | Supplementary: threshold sensitivity (manual run) |
| `data/results/<run>/convtran_batch_size_study/convtran_batch_size_summary.csv` | Supplementary: batch-size sensitivity (manual run) |

---

## Citation

If you use TRAIT in your work, please cite:

```bibtex
@inproceedings{[key],
  title     = {[Title]},
  author    = {[Authors]},
  booktitle = {[Venue]},
  year      = {[Year]},
  doi       = {[DOI]}
}
```

---

## License

[To be added upon acceptance]
