# Example tracks

These two tracks are a small sample from the
[TRAIT dataset](https://doi.org/10.26180/32737458): one labelled `insect` and one
labelled `not_insect`. They are included so you can try feature extraction without
downloading the full dataset. They are too small a sample for training or evaluating
a classifier.

Each CSV in `tracks/` contains frame numbers and centroid coordinates in pixels
(`nframe,x,y`). The recordings use 30 frames per second. `labels.csv` lists the
filenames and their original annotations.

From the repository root, run:

```bash
python src/generate_features.py \
    --tracks-dir examples/tracks \
    --labels-csv examples/labels.csv \
    --fps 30 \
    --output-csv data/results/example/track_features.csv
```

The output contains two rows, each with a filename, binary label and 125 feature
columns. An empty feature value means that it could not be estimated for that track.

The sample tracks are shared unchanged under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Source: Ratnayake,
M. N. and Dorin, A. (2026), *TRAIT: Annotated Insect and Non-Insect Trajectories
from Camera Traps*, Monash University, [doi:10.26180/32737458](https://doi.org/10.26180/32737458).
