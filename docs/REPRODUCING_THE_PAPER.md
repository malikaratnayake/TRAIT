# Reproducing the paper

This guide walks through the experiments in *Detecting Insect Flight Trajectories for
Pollinator Monitoring with Camera Traps* (WACV 2027), from downloading the tracks to
comparing your results with the reference run.

[Dataset](https://doi.org/10.26180/32737458)

For installation, a small working example or guidance on your own tracks, start with
the [README](../README.md). The standard pipeline covers the main experiments,
ablations and operating points. Two additional studies from the supplement are
[run separately](#supplementary-studies).

## Contents

- [Dataset](#dataset)
- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Running the pipeline](#running-the-pipeline)
- [Computing requirements and reproducibility](#computing-requirements-and-reproducibility)
- [Reported results](#reported-results)
- [Comparing with the published results](#comparing-with-the-published-results)
- [Where each result comes from](#where-each-result-comes-from)
- [Supplementary studies](#supplementary-studies)

---

## Dataset

The dataset contains 28,295 manually annotated 2D centroid tracks from two commercial
blueberry farms in Australia (Farm A and Farm B), covering eight semi-protected cropping
tunnels and 153 camera recording stations. Videos were recorded at 30 fps and
1920 × 1080 pixels by a track-mounted robotic monitoring platform with the camera
70–100 cm from the crop, and processed with Polytrack (Ratnayake et al., IJCV 2023)
to extract one CSV per track.

| Subset       | Stations | Tracks | Insect | Not insect |
|--------------|---------:|-------:|-------:|-----------:|
| Farm A       | 117      | 18,821 | 2,194  | 16,627     |
| Farm B       | 36       | 9,474  | 1,450  | 8,024      |
| **Total**    | **153**  | **28,295** | **3,644** | **24,651** |
| Train/val (folds 0–3) | 122 | 22,839 | 2,846 | 19,993 |
| Test (fold 4) | 31      | 5,456  | 798    | 4,658      |

Each track was labelled by overlaying it on its video clip. `insect` covers honeybees and
other insects. `not_insect` covers wind-blown vegetation, shadows, camera vibration,
birds and people. When a track contained several motion sources, it was given the label
of the source present for more than 50% of its duration.

### Download

The dataset is published on Figshare under CC BY 4.0: [doi:10.26180/32737458](https://doi.org/10.26180/32737458).
Download `tracks.zip`, `track_classification.csv` and `splits_folds.csv`, put the two CSV
files in `data/` and extract `tracks.zip` there, so the layout is:

```
data/
├── tracks/                    # 28,295 CSV files, columns: nframe, x, y (pixels)
├── track_classification.csv   # columns: filepath, label (insect | not_insect)
└── splits_folds.csv           # the five-fold assignment used in the paper
```

The Figshare record also has `README.txt`, which describes the data (files, columns,
collection, labelling, splits, limitations), and `SHA256SUMS.txt` with a checksum of every
file. A copy of the same files, `TRAIT_dataset.zip`, is attached to the
[`wacv2027` release](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027) for
convenience; the Figshare record is the one to cite.

### File naming

```
FarmA_L1_S10_250729_093716_100005000023.csv
│     │  │   │      │      └─ track ID
│     │  │   │      └─ recording start time (HHMMSS)
│     │  │   └─ recording date (YYMMDD)
│     │  └─ station within the location
│     └─ location (cropping tunnel) within the farm
└─ farm (A or B)
```

The farm, location and station together (`FarmA_L1_S10`) identify a camera station.
They are used to build station-disjoint splits but are never used as model inputs.

Track rows follow consecutive frame numbers in time order, with empty coordinates for
missed detections. This matters because the preprocessing code interpolates over row
positions. For another track format, insert any omitted frame rows before running it.

---

## Method overview

1. **Gap filling and smoothing.** Missing coordinates are filled by linear interpolation
   along the consecutive frame rows. Tracks shorter than five frames were excluded when
   the dataset was built. Coordinates are then smoothed with a Savitzky–Golay filter
   (11-frame window, polynomial order 2). Shorter tracks use the largest valid odd window.
2. **Feature extraction.** 125 features in six domains and 32 subgroups are computed from
   the smoothed track. The wingbeat-residual features also use the raw coordinates.
   Full definitions and equations are in Section 1 of the supplementary material, and
   the feature names in the code match those tables.

   | Domain | Focus | Features | Subgroups |
   |---|---|---:|---|
   | D1 | Fundamental locomotion kinematics | 26 | track support, speed statistics, dynamics, speed distribution, motion intensity |
   | D2 | Spatial occupancy and path geometry | 19 | path geometry, spatial spread, exploration, diffusion, sinuosity |
   | D3 | Flight rhythm, oscillation and biomechanics | 30 | horizontal and vertical oscillation, XY coherence, global spectral, wingbeat residuals, mean reversion |
   | D4 | Navigation, orientation and search behaviour | 16 | directionality, heading use, zigzag and casting, circling, directional bias |
   | D5 | Sensorimotor control and manoeuvrability | 20 | turning, saccades, fixation structure, manoeuvrability, speed scaling, Lévy statistics, aerodynamics |
   | D6 | Behavioural state organisation | 14 | hover–burst, stops, landing, takeoff |

3. **Station-disjoint evaluation.** Tracks are split into five folds with
   `StratifiedGroupKFold`, grouped by camera station and stratified by farm location and
   class. A deterministic search over seeds 0–99 selects the most balanced split
   (selected seed 47). Fold 4 is reserved for final evaluation, with model selection,
   early stopping and deployed threshold selection carried out on folds 0–3.
4. **Classification.** Logistic Regression, Random Forest and XGBoost are trained on the
   features. The 1D-CNN (FCN), miniROCKET and ConvTran baselines are trained on the same
   gap-filled, smoothed trajectories, using four channels (normalised x, y and speed, and
   heading change) padded or truncated to 512 steps. Hyperparameters and early stopping
   use validation PR-AUC. Each model's decision threshold maximises validation or
   out-of-fold F1 and is then fixed for testing.

---

## Repository structure

```
TRAIT/
├── run_pipeline.sh                     # Runs the full pipeline (Phases 1–5)
├── clean.sh                            # Deletes one run folder (dry run unless --confirm)
├── requirements.txt                    # Pinned dependencies used for the paper
├── CITATION.cff                        # How to cite this software and the paper
├── LICENSE                             # GPL-3.0
├── examples/                           # Two labelled tracks for a small first run
├── data/                               # Put the downloaded dataset here
└── src/
    ├── generate_features.py            # Phase 1: gap filling, smoothing, 125 features
    ├── create_splits.py                # Phase 2: station-disjoint five-fold splits
    ├── models/
    │   ├── data_loader.py              # Feature hierarchy (6 domains, 32 subgroups), data loading
    │   ├── evaluate.py                 # Metrics, bootstrap confidence intervals, latency
    │   ├── majority_baseline.py        # Majority-class baseline
    │   └── tabular_models.py           # Logistic Regression, Random Forest, XGBoost
    ├── train_fcn.py                    # 1D-CNN (FCN) and shared sequence preprocessing
    ├── train_rocket.py                 # miniROCKET
    ├── train_convtran.py               # ConvTran
    ├── cross_farm.py                   # Phase 4: within- and cross-farm evaluation
    ├── ablation.py                     # Phase 5: ablations and permutation importance
    ├── operating_points.py             # XGBoost operating points (Supp. Table 10)
    ├── cross_farm_threshold_study.py   # Threshold robustness (Supp. Table 9)
    ├── convtran_batch_size_study.py    # ConvTran batch-size sweep (Supp. Table 11)
    ├── compare_results.py              # Compares a run with the published results
    ├── figures/
    │   ├── plot_session_trajectories.py        # Figure 2
    │   ├── plot_class_trajectory_overlays.py   # Figure 3
    │   ├── plot_pr_curves.py                   # Figure 4
    │   └── plot_group_loo.py                   # Supp. Figure 1
    └── utils/
        ├── smoothing.py                # Gap filling and Savitzky–Golay smoothing
        └── metadata.py                 # Parses farm, location and station from file names
```

To start a run again from scratch, delete its folder with `bash clean.sh <run-name> --confirm`
(without `--confirm` it only lists what would be deleted). It never touches the dataset, the
split files or run folders that contain read-only files.

---

## Running the pipeline

Use the source archive from the
[`wacv2027` release](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027)
for the paper version, or select the tag in a clean clone before installing the
[pinned dependencies](../requirements.txt):

```bash
git checkout wacv2027
```

Use Python 3.10, as in the paper. For later runs, keep a note of the commit returned by
`git rev-parse HEAD` alongside your results. This helps distinguish changes in the code
from differences in the computing environment.

Run the main experiments and ablations in deterministic mode:

```bash
bash run_pipeline.sh --fps 30 --test-run --run-name paper
```

Outputs are written to `data/results/paper/`. `--test-run` runs every model on one CPU
thread, uses deterministic PyTorch with seed 42, and keeps the 1D-CNN and ConvTran on the
CPU. Despite the name, `--test-run` runs the full experiments; it is not a shortened
check. Without this option, model fitting uses the available cores and the neural
baselines can use CUDA when available. This can speed up a run, but may change the results.

Phase 2 regenerates `data/splits_folds.csv` and `data/split_statistics.csv`; the regenerated
fold assignment is identical to the one in the dataset download.

| Option | Effect |
|---|---|
| `--fps 30` | Frame rate used for feature extraction. Required on the first run of each run folder, then cached in its `pipeline.cfg`. |
| `--run-name NAME` | Write results to `data/results/NAME/`. |
| `--test-run` | Deterministic, single-threaded, CPU-only mode used for the paper. |
| `--from-phase N` | Start at phase N (1–5), skipping earlier phases and honouring existing checkpoints. It does not force completed steps to run again. |
| `--overwrite-results` | Delete the run folder and start again. |
| `--dry-run` | Print the commands without running them. |
| `-h`, `--help` | List all options. |

The pipeline has five phases:

| Phase | What it does | Script(s) |
|---|---|---|
| 1 | Gap filling, smoothing and extraction of the 125 features | `src/generate_features.py` |
| 2 | Station-disjoint five-fold splits | `src/create_splits.py` |
| 3 | Train and test all models; comparison table and PR curves | `src/models/`, `src/train_fcn.py`, `src/train_rocket.py`, `src/train_convtran.py` |
| 4 | Within-farm and cross-farm generalisation (XGBoost) | `src/cross_farm.py` |
| 5 | Domain, subgroup and feature-level ablations; trajectory figures | `src/ablation.py`, `src/figures/` |

Use a new `--run-name` when you want a fresh run while keeping an earlier one for
comparison. `--overwrite-results` replaces the selected run folder. Split files are
shared by default, so use a separate `--splits-csv` path when working with another dataset.

## Computing requirements and reproducibility

The reference experiments ran on Linux x86-64 with a CPU, 16 GB RAM, Python 3.10 and
the versions in `requirements.txt`. A similar environment is the best starting point
for a close match. Deterministic settings reduce variation, but do not guarantee identical
floating-point results across different processors, operating systems or library builds.
Differences of up to 0.004 in cross-farm PR-AUC have been observed on macOS with Apple
silicon; those differences exceed the comparison tool's default tolerance of 0.0005.

The downloaded tracks and one saved run together occupy roughly 300 MB in a local
checkout. Allow additional disk space for the Python environment, package downloads,
reference archive and any extra runs. The environment size depends on the platform
and whether GPU libraries are installed.

The reference CPU model and timings for each phase are not documented here,
so there is no reliable wall-clock estimate to quote. The full pipeline trains several
models and repeats XGBoost fits for the ablations; the optional batch-size study adds
five full ConvTran training runs. Start with the example in the README to check your
installation, then allow time for a full run. Pipeline logs in `logs/` record progress
with timestamps, and checkpoints let you resume completed work.

## Reported results

Model comparison on the held-out test set of the paper (fold 4, 5,456 tracks, 14.6% insect), for
checking a reproduction. PR-AUC confidence intervals use 1,000 bootstrap resamples of
the test set. Latency is the mean time for the classifier call on a precomputed feature
vector or preprocessed sequence. It excludes video processing, tracking, gap filling,
smoothing and feature extraction, and depends on the hardware.

| Model | PR-AUC (95% CI) | τ | F1 | Recall | Precision | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.146 (0.138–0.155) | – | 0.000 | 0.000 | 0.000 | 0.18 |
| Logistic Regression | 0.908 (0.892–0.923) | 0.76 | 0.839 | 0.846 | 0.832 | 0.18 |
| Random Forest | 0.919 (0.904–0.933) | 0.28 | 0.853 | 0.846 | 0.861 | 12.49 |
| **XGBoost** | **0.929 (0.915–0.942)** | 0.53 | **0.866** | 0.833 | **0.902** | 0.53 |
| 1D-CNN (FCN) | 0.848 (0.824–0.871) | 0.86 | 0.776 | 0.721 | 0.839 | 5.35 |
| miniROCKET | 0.817 (0.794–0.841) | 0.30 | 0.738 | 0.772 | 0.707 | 6.07 |
| ConvTran | 0.847 (0.823–0.872) | 0.88 | 0.781 | 0.724 | 0.848 | 19.66 |

---

## Comparing with the published results

The outputs of the run reported in the paper are attached to the
[`wacv2027` release](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027) as
`WACV2027_paper_results.zip`. They include every table, the per-track test predictions of
each model, the feature table, the fitted LR, Random Forest and XGBoost models, and the
outputs of the two supplementary studies, in the same layout as a run folder.

Unzip the archive into `WACV2027_paper_results/`, then compare the main results:

```bash
python src/compare_results.py \
    --run       data/results/paper \
    --reference WACV2027_paper_results \
    --features \
    --splits-run data/splits_folds.csv --splits-reference WACV2027_paper_results/splits_folds.csv
```

The comparison checks the listed metrics and selected model settings behind the tables,
along with test scores, labels and track identities for all seven classifiers. The two
flags in the example also check the feature table and fold assignments. It checks saved
values rather than comparing figures or fitted model files.

Missing files, columns or rows in a requested comparison cause a failure, including
when the reference archive is incomplete. A `MATCH` means all requested checks passed;
the exit status is 0 for a match and 1 for missing or different results. Latencies are
shown for context and do not affect the outcome.

By default, the two separately run supplementary studies are left out. After running
[both studies](#supplementary-studies), add `--include-supplementary` to the comparison
command to include Tables 9 and 11. Their outputs must then be present in both folders.

`--tol` sets the absolute tolerance for numeric metrics and prediction scores
(default 0.0005). Predicted labels, true labels and fold assignments must agree exactly.
Feature values use a separate relative tolerance of `1e-9`, with matching missing values
allowed. Use `--details-csv comparison.csv` to save the comparisons. If your environment
produces a difference, inspect it before changing the tolerance; accepting a larger
difference does not establish that the runs are equivalent.

---

## Where each result comes from

All paths are relative to `data/results/<run-name>/` unless they start with `data/`.

| Paper item | Output |
|---|---|
| Table 1 — dataset summary | `data/split_statistics.csv` |
| Figure 2 — all tracks from one recording | `figures/session_trajectories_rank1.png` |
| Figure 3 — class-conditional trajectory overlays | `figures/class_trajectory_overlays.png` |
| Table 3 — model comparison | `phase3_model_comparison.csv` |
| Figure 4 — precision–recall curves | `figures/phase3_precision_recall_curves.png` |
| Table 4 — within- and cross-farm results | `cross_farm/cross_farm_results.csv` |
| Table 5 — cumulative domain ablation | `ablation/domain_ablation_results.csv` |
| Table 6 — single-domain ablation | `ablation/domain_only_results.csv` |
| Supp. Table 8 — selected hyperparameters | `prauc_protocol/*/results.json` (LR, RF, XGBoost); `fcn_prauc/fcn_results.csv`, `rocket_prauc/rocket_results.csv`, `convtran_prauc/convtran_results.csv` |
| Supp. Table 9 — threshold robustness across farms | `cross_farm_threshold_study/cross_farm_threshold_summary.csv` (see below) |
| Supp. Table 10 — XGBoost operating points | `prauc_protocol/xgboost_prauc/operating_points.csv` |
| Supp. Table 11 — ConvTran batch-size sensitivity | `convtran_batch_size_study/convtran_batch_size_summary.csv` (see below) |
| Supp. Table 12 — leave-one-subgroup-out ablation | `ablation/group_loo_ablation.csv` |
| Supp. Figure 1 — largest subgroup contributions | `figures/group_loo_prauc_top_positive_negative.png` |
| Supp. Table 13 — permutation feature importance | `ablation/feature_importance.csv` |

The features for every track are in `track_features.csv`, and the fold assignments in
`data/splits_folds.csv`.

---

## Supplementary studies

These two studies are not part of a default run.

**Threshold robustness across farms (Supp. Table 9).** Retrains the Phase 4 XGBoost
models and compares F1 at the fixed threshold (τ = 0.53) with F1 at each evaluation's own
best threshold. The latter is a diagnostic calculated from test labels, rather than a
threshold to select for deployment. Run it after the pipeline has finished, with one
thread to match the reference configuration:

```bash
python src/cross_farm_threshold_study.py \
    --features-csv data/results/paper/track_features.csv \
    --splits-csv   data/splits_folds.csv \
    --output-dir   data/results/paper \
    --n-jobs 1
```

**ConvTran batch-size sensitivity (Supp. Table 11).** The reported ConvTran is trained
with batch size 128, matching the other sequence baselines, whereas the published ConvTran
recipe uses 16. This study retrains ConvTran at several batch sizes. Each batch size is a
full training run, and smaller batches take longer. Results go to their own folder, and
the reported ConvTran result is not changed.

```bash
bash run_pipeline.sh --fps 30 --test-run --run-name paper --from-phase 3 \
    --convtran-batch-sweep "16 32 64 96 128"
```

---
