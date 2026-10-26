# TRAIT — Trajectory Recognition for Automated Insect Tagging

TRAIT classifies 2D centroid tracks from camera traps as **insect** or **not insect** using
only motion. Each track is gap-filled and smoothed, described by 125 interpretable,
behaviour-informed motion features organised into six domains, and classified with
XGBoost. No appearance information, image data or GPU is required.

This repository contains the code and instructions to reproduce all results in:

> Malika Nisal Ratnayake and Alan Dorin.
> **Detecting Insect Flight Trajectories for Pollinator Monitoring with Camera Traps.**
> *IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2027.

On a held-out, station-disjoint test set, XGBoost on the 125 features reaches
**PR-AUC 0.929** and **F1 0.866** at 0.53 ms per track on a single CPU thread, ahead of the
1D-CNN, miniROCKET and ConvTran sequence baselines trained on the same trajectories.

---

## Contents

- [Dataset](#dataset)
- [Method overview](#method-overview)
- [Installation](#installation)
- [Reproducing the paper](#reproducing-the-paper)
- [Where each result comes from](#where-each-result-comes-from)
- [Supplementary studies](#supplementary-studies)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

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

The dataset is released under CC BY 4.0 at **[dataset DOI — to be added]**. Download it
and extract it into `data/` so the layout is:

```
data/
├── tracks/                    # 28,295 CSV files, columns: nframe, x, y (pixels)
└── track_classification.csv   # columns: filepath, label (insect | not_insect)
```

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

---

## Method overview

1. **Gap filling and smoothing.** Missed detections are filled by linear interpolation
   over the frame index. Tracks shorter than five frames are discarded. Coordinates are
   then smoothed with a Savitzky–Golay filter (11-frame window, polynomial order 2). For
   shorter tracks the largest valid odd window is used.
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
   class. A deterministic seed search over 100 candidates selects the most balanced split
   (`random_state = 47`). Fold 4 is the held-out test set. It is used once for final
   evaluation and never for model selection, early stopping or threshold tuning.
4. **Classification.** Logistic Regression, Random Forest and XGBoost are trained on the
   features. The 1D-CNN (FCN), miniROCKET and ConvTran baselines are trained on the same
   gap-filled, smoothed trajectories, using four channels (normalised x, y and speed, and
   heading change) padded or truncated to 512 steps. Hyperparameters and early stopping
   use validation PR-AUC. Each model's decision threshold maximises validation or
   out-of-fold F1 and is then fixed for testing.

---

## Installation

Python 3.10 or later is required. The pinned versions in `requirements.txt` are the
versions used for the paper.

```bash
git clone https://github.com/malikaratnayake/TRAIT.git
cd TRAIT
python3.10 -m venv trait-env
source trait-env/bin/activate          # Windows: trait-env\Scripts\activate
python -m pip install -r requirements.txt
```

Check that every dependency imports before starting a long run:

```bash
python -c "import numpy, scipy, pandas, sklearn, xgboost, matplotlib, torch, aeon; print('OK')"
```

If pip cannot find SciPy 1.15.3, the environment is using Python 3.9 or older. Recreate
it with Python 3.10 or later.

---

## Reproducing the paper

Run the whole pipeline in deterministic mode, as used for the paper:

```bash
bash run_pipeline.sh --fps 30 --test-run --run-name paper
```

Outputs are written to `data/results/paper/`. `--test-run` runs every model on one CPU
thread, uses deterministic PyTorch with seed 42, and keeps the 1D-CNN and ConvTran on the
CPU. This is the configuration behind the reported results and latencies (Linux, 16 GB
RAM, CPU only). It is slow, so plan for a long run. Without `--test-run`, model fitting
uses all available cores and is faster, but the last decimal places of some results may
differ.

Each step writes a checkpoint when it finishes. Re-running the same command skips
completed steps, so an interrupted run can simply be restarted.

| Option | Effect |
|---|---|
| `--fps 30` | Frame rate used for feature extraction. Required on the first run of each run folder, then cached in its `pipeline.cfg`. |
| `--run-name NAME` | Write results to `data/results/NAME/`. |
| `--test-run` | Deterministic, single-threaded, CPU-only mode used for the paper. |
| `--from-phase N` | Re-run from phase N (1–5) onwards. |
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

### Expected results

Held-out test set (fold 4, 5,456 tracks, 14.6% insect). Latencies depend on the hardware.

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
best threshold. Run it after the pipeline has finished:

```bash
python src/cross_farm_threshold_study.py \
    --features-csv data/results/paper/track_features.csv \
    --splits-csv   data/splits_folds.csv \
    --output-dir   data/results/paper
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

## Repository structure

```
TRAIT/
├── run_pipeline.sh                     # Runs the full pipeline (Phases 1–5)
├── clean.sh                            # Deletes generated outputs (dry run unless --confirm)
├── requirements.txt                    # Pinned dependencies used for the paper
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
    ├── figures/
    │   ├── plot_session_trajectories.py        # Figure 2
    │   ├── plot_class_trajectory_overlays.py   # Figure 3
    │   ├── plot_pr_curves.py                   # Figure 4
    │   └── plot_group_loo.py                   # Supp. Figure 1
    └── utils/
        ├── smoothing.py                # Gap filling and Savitzky–Golay smoothing
        └── metadata.py                 # Parses farm, location and station from file names
```

To delete all generated outputs and start from scratch, run `bash clean.sh --confirm`.
The downloaded dataset is kept.

---

## Citation

If you use the code or dataset, please cite:

```bibtex
@inproceedings{ratnayake2027trait,
  title     = {Detecting Insect Flight Trajectories for Pollinator Monitoring with Camera Traps},
  author    = {Ratnayake, Malika Nisal and Dorin, Alan},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

---

## Acknowledgements

This work was supported by the Australian Research Council Linkage Projects funding
scheme (project LP210200213), the Australian Blueberry Growers' Association and
Costa Group.

---

## License

- Code: **[to be added]**
- Dataset: CC BY 4.0
