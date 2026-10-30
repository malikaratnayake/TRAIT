# TRAIT — Trajectory Recognition for Automated Insect Tagging

[![Dataset DOI](https://img.shields.io/badge/dataset%20DOI-10.26180%2F32737458-blue)](https://doi.org/10.26180/32737458)
[![Code licence: GPL-3.0](https://img.shields.io/badge/code%20licence-GPL--3.0-blue.svg)](LICENSE)
[![Data licence: CC BY 4.0](https://img.shields.io/badge/data%20licence-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Paper: WACV 2027](https://img.shields.io/badge/paper-WACV%202027-green.svg)](#citation)
[![Release: wacv2027](https://img.shields.io/badge/release-wacv2027-orange.svg)](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027)

TRAIT is a Python toolkit that decides whether a camera-trap track is an **insect** or
**not an insect** from its motion alone. It takes the `(frame, x, y)` centroid track of a
moving object, describes it with **125 interpretable, behaviour-informed motion features**
in six domains, and classifies it with a lightweight model that runs on a CPU. It needs no
images and no GPU.

TRAIT can be used to:

- remove false detections (wind-blown vegetation, shadows, camera shake) from the output of
  an insect tracking or detection system;
- compute interpretable flight and movement features for ecological or behavioural analysis;
- train and compare trajectory classifiers on a labelled dataset with station-disjoint
  splits and cross-farm tests.

The repository includes the code for every step (features, splits, classifiers, ablations,
figures), a pipeline script that runs them end to end, and a tool for comparing a run with
the results of the accompanying paper (see [Citation](#citation)). The dataset of 28,295
annotated tracks is published separately on Figshare.

---

## Contents

- [What is in the toolkit](#what-is-in-the-toolkit)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Using TRAIT on your own tracks](#using-trait-on-your-own-tracks)
- [Dataset](#dataset)
- [Method overview](#method-overview)
- [Reproducing the paper](#reproducing-the-paper)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## What is in the toolkit

| Component | What it does | Script |
|---|---|---|
| Preprocessing | Fills gaps in tracks by linear interpolation and smooths them (Savitzky–Golay) | `src/utils/smoothing.py` |
| Feature extraction | Computes the 125 motion features for every track | `src/generate_features.py` |
| Data splits | Builds station-disjoint cross-validation folds, so no camera station is in both training and test data | `src/create_splits.py` |
| Classifiers | Logistic Regression, Random Forest and XGBoost on the features; 1D-CNN, miniROCKET and ConvTran on the raw trajectories | `src/models/`, `src/train_*.py` |
| Generalisation and ablation | Within- and cross-farm evaluation; domain, subgroup and feature-level ablations | `src/cross_farm.py`, `src/ablation.py` |
| Pipeline | Runs all of the above end to end, with checkpoints so it can be resumed | `run_pipeline.sh` |
| Comparison | Checks a run against the published results | `src/compare_results.py` |

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

Check that every dependency imports:

```bash
python -c "import numpy, scipy, pandas, sklearn, xgboost, matplotlib, torch, aeon; print('OK')"
```

If pip cannot find SciPy 1.15.3, the environment is using Python 3.9 or older. Recreate
it with Python 3.10 or later.

---

## Quick start

1. Download the dataset ([see Dataset](#dataset)) into `data/`.
2. Run the pipeline:

```bash
bash run_pipeline.sh --fps 30 --run-name my_run
```

Results are written to `data/results/my_run/`: the feature table, the model comparison, the
cross-farm and ablation results, and figures. Each step writes a checkpoint when it
finishes, so an interrupted run can be restarted with the same command. Use `--dry-run` to
print the commands without running them, and `-h` to list all options.

---

## Using TRAIT on your own tracks

**Input.** One CSV file per track with the columns `nframe,x,y` (frame number and centroid
in pixels; `x` and `y` may be empty in frames without a detection), and a labels CSV in the
format of `track_classification.csv` (columns `filepath,label`, with `insect` or
`not_insect`).

**Extract the 125 features** for all tracks:

```bash
python src/generate_features.py \
    --tracks-dir path/to/tracks \
    --labels-csv path/to/labels.csv \
    --fps        30 \
    --output-csv features.csv
```

The output has one row per track with its label and the 125 features. Gap filling and
smoothing are done inside the script. The feature names, domains and subgroups are declared
in `src/models/data_loader.py`, and their equations are in the supplementary material of the
paper. The features can then be used with any classifier, or with the training code in
`src/models/`.

The station-disjoint splits and the full pipeline (`src/create_splits.py`,
`run_pipeline.sh`) read the camera station from the track file name (see
[the file naming](docs/REPRODUCING_THE_PAPER.md#file-naming)). To use them on other data, name the files in the same pattern
or adapt `src/utils/metadata.py`. The pipeline can also be pointed at other folders with
`--tracks-dir`, `--labels-csv`, `--features-csv` and `--splits-csv`.

---

## Dataset

The TRAIT dataset has 28,295 manually annotated 2D centroid tracks (3,644 insect, 24,651
not insect) from two commercial blueberry farms in Australia, recorded at 30 fps by camera
traps at 153 stations. It also includes the station-disjoint five-fold split used in the
paper. It is published on Figshare under CC BY 4.0:
[doi:10.26180/32737458](https://doi.org/10.26180/32737458).

Download `tracks.zip`, `track_classification.csv` and `splits_folds.csv`, put the two CSV
files in `data/` and extract `tracks.zip` there:

```
data/
├── tracks/                    # 28,295 CSV files, columns: nframe, x, y (pixels)
├── track_classification.csv   # columns: filepath, label (insect | not_insect)
└── splits_folds.csv           # the five-fold assignment used in the paper
```

The Figshare record also has a `README.txt` that describes the data in full, and
`SHA256SUMS.txt`. A copy of the files, `TRAIT_dataset.zip`, is attached to the
[`wacv2027` release](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027);
the Figshare record is the one to cite. Track file names encode the camera station, for
example `FarmA_L1_S10_250729_093716_100005000023.csv` (farm A, location 1, station 10,
recorded on 2025-07-29 at 09:37:16, track 100005000023); see
[the details](docs/REPRODUCING_THE_PAPER.md#file-naming).

---

## Method overview

1. **Gap filling and smoothing.** Missed detections are filled by linear interpolation and
   the coordinates are smoothed with a Savitzky–Golay filter (11-frame window, order 2).
2. **Feature extraction.** 125 features in six domains (locomotion kinematics, path
   geometry, flight rhythm, navigation, sensorimotor control, behavioural states) and 32
   subgroups. No recording metadata (time of day, station, farm) is used.
3. **Station-disjoint evaluation.** Five folds grouped by camera station, so no station is
   in both training and test data; fold 4 is the held-out test set.
4. **Classification.** Logistic Regression, Random Forest and XGBoost on the features;
   1D-CNN, miniROCKET and ConvTran on the trajectories.

Full details, the feature domains and the repository layout are in
[docs/REPRODUCING_THE_PAPER.md](docs/REPRODUCING_THE_PAPER.md#method-overview).

---

## Reproducing the paper

All results in the paper can be reproduced with

```bash
bash run_pipeline.sh --fps 30 --test-run --run-name paper
```

The published outputs are attached to the
[`wacv2027` release](https://github.com/malikaratnayake/TRAIT/releases/tag/wacv2027) as
`WACV2027_paper_results.zip`, and `src/compare_results.py` checks a run against them.
Options, expected results, the mapping from paper tables and figures to output files, and
the supplementary studies are in
[docs/REPRODUCING_THE_PAPER.md](docs/REPRODUCING_THE_PAPER.md).

---

## Citation

If you use the code or dataset, please cite the paper (GitHub's "Cite this repository"
button uses `CITATION.cff`):

```bibtex
@inproceedings{ratnayake2027trait,
  title     = {Detecting Insect Flight Trajectories for Pollinator Monitoring with Camera Traps},
  author    = {Ratnayake, Malika Nisal and Dorin, Alan},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

and the dataset: Ratnayake, M. N. and Dorin, A. (2026). *TRAIT: Annotated Insect and
Non-Insect Trajectories from Camera Traps*. Monash University. Dataset.
https://doi.org/10.26180/32737458

Questions and bug reports are welcome as [GitHub issues](https://github.com/malikaratnayake/TRAIT/issues).

---

## Acknowledgements

This work was supported by the Australian Research Council Linkage Projects funding
scheme (project LP210200213), the Australian Blueberry Growers' Association and
Costa Group.

---

## License

- Code: GNU General Public License v3.0 (see `LICENSE`)
- Dataset: CC BY 4.0
