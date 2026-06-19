# TRAIT — Trajectory Recognition for Automated Insect Tagging

TRAIT is a reproducible pipeline for classifying insect trajectories recorded by optical traps. Given raw (x, y) centroid track CSV files, it applies Savitzky-Golay smoothing, extracts 125 biologically-informed motion features across six domains, and trains and evaluates multiple classifiers.

This repository accompanies the paper:

> **[Title]** — *[Authors]*, *[Venue, Year]*
> DOI: [doi]

---

## Repository structure

```
TRAIT/
├── run_pipeline.sh              # End-to-end pipeline driver (Phases 1–5)
├── data/
│   ├── tracks/                  # 28,295 raw track CSVs (Farm A + Farm B)
│   └── track_classification.csv # filepath, label (insect / not_insect)
└── src/
    ├── generate_features.py     # Phase 1: SG smoothing + 125-feature extraction
    ├── create_splits.py         # Phase 2: station-disjoint grouped CV splits
    ├── models/                  # Phase 3: model training and evaluation utilities
    │   ├── data_loader.py       # Feature loading, fold/cross-farm splitting
    │   ├── evaluate.py          # Metrics, bootstrap CI, latency measurement
    │   ├── majority_baseline.py # Majority baseline trainer
    │   └── tabular_models.py    # LR, RF, XGBoost with PR-AUC selection
    ├── train_fcn.py             # Phase 3: 1D-CNN baseline
    ├── train_rocket.py          # Phase 3: miniROCKET baseline
    ├── cross_farm.py            # Phase 4: cross-farm generalisation
    ├── ablation.py              # Phase 5: domain and group ablation studies
    ├── figures/
    │   ├── plot_pr_curves.py    # Phase 3: precision-recall curves figure
    │   └── plot_group_loo.py    # Phase 5: group leave-one-out figure
    └── utils/
        ├── smoothing.py         # Savitzky-Golay smoothing utilities
        └── metadata.py          # Track filename parser (anonymized site/station/date)
```

---

## Dataset

The dataset consists of 28,295 manually annotated 2D centroid trajectory recordings from two geographically separated sites. `Farm A` and `Farm B` are pseudonymous site labels; raw site identities are not included in this repository.

| Site   | Stations | Tracks |
|--------|----------|--------|
| Farm A | 117      | 18,821 |
| Farm B | 36       | 9,474  |
| **Total** | **153** | **28,295** |

Insect prevalence is 12.9% across the full dataset. Each track is a short CSV of `(nframe, x, y)` pixel detections produced by an optical trapping system.

**Directory layout (included in this repo):**

```
data/
├── tracks/                  # One CSV per track (nframe, x, y)
└── track_classification.csv # filepath, label  (insect / not_insect)
```

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

This installs all dependencies for Phases 1–5 (feature extraction, CV splits, LR / RF / XGBoost / 1D-CNN / miniROCKET training, cross-farm generalisation, and ablation studies) into a single environment.

### 2. Activate before running the pipeline

```bash
source trait-env/bin/activate
bash run_pipeline.sh --fps 30
```

If PyTorch or aeon fail to import (e.g. on an unsupported platform), the pipeline automatically skips the 1D-CNN / miniROCKET steps and continues.

---

## Reproducing results

Run the full pipeline end-to-end:

```bash
bash run_pipeline.sh --fps 30
```

Run the final-report reproducibility mode:

```bash
bash run_pipeline.sh --fps 30 --test-run --run-name final_report
```

`--test-run` forces single-threaded RF/XGBoost/miniROCKET, deterministic
PyTorch settings, and CPU FCN execution. It is slower, but is intended for the
canonical result table.

Resume from a specific phase (e.g. after Phase 2 splits already exist):

```bash
bash run_pipeline.sh --fps 30 --from-phase 3
```

Save results to a named folder:

```bash
bash run_pipeline.sh --fps 30 --run-name my_run
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

Model selection and hyperparameter tuning use validation PR-AUC only; decision thresholds are tuned by validation F1. Output: `phase3_model_comparison.csv` and `figures/phase3_precision_recall_curves.png`.

**Phase 4 — Cross-farm generalisation**

Trains XGBoost on all tracks from one farm and evaluates on the other (A→B and B→A), with no domain adaptation or shared normalisation.

**Phase 5 — Ablation studies**

Three complementary ablation experiments, all using XGBoost on the held-out test set:

- *Cumulative domain ablation*: D1 → D1+D2 → D1+D2+D3 → … → D1–D6
- *Single-domain-only ablation*: each domain Di evaluated independently
- *Leave-one-group-out ablation*: each of 32 biological feature subgroups left out in turn

---

## Key outputs

| File | Description |
|------|-------------|
| `data/splits_folds.csv` | Station-disjoint fold assignments (generated by Phase 2) |
| `data/results/<run>/track_features.csv` | 125-feature matrix for all tracks |
| `data/results/<run>/phase3_model_comparison.csv` | Model comparison table (all 6 models) |
| `data/results/<run>/figures/phase3_precision_recall_curves.png` | PR curves figure |
| `data/results/<run>/cross_farm/cross_farm_results.csv` | Cross-farm results (Phase 4) |
| `data/results/<run>/ablation/domain_ablation_results.csv` | Cumulative domain ablation |
| `data/results/<run>/ablation/domain_only_results.csv` | Single-domain-only ablation |
| `data/results/<run>/ablation/group_loo_ablation.csv` | Group leave-one-out ablation |
| `data/results/<run>/figures/group_loo_top_positive_negative.png` | Group LOO figure |

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
