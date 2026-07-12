# ECG Classification (Normal vs. PVC)

Classify single ECG heartbeats as **Normal** or **PVC** (Premature Ventricular Contraction) using a
Butterworth band-pass filter, Daubechies wavelet features, and a k-Nearest-Neighbours classifier.

## Overview

A Premature Ventricular Contraction is an arrhythmia that produces a visibly distinct beat shape in an
ECG trace. Manually screening 24-hour ECG recordings beat-by-beat is slow and exhausting, so this project
automates the decision for a single beat.

The pipeline is the classical (non-deep-learning) one described in `RequiredDetails.pdf`:

1. **Pre-processing** — 4th-order Butterworth band-pass filter, 0.5–40 Hz, at a 360 Hz sampling rate,
   to strip baseline wander and high-frequency noise.
2. **Feature extraction** — 2-level discrete wavelet transform with the `db4` (Daubechies-4) mother
   wavelet; the level-2 coefficient vector is kept as the feature vector, then min-max scaled to `[0, 1]`.
3. **Classification** — `KNeighborsClassifier` with `k = 5`.

Everything lives in a single notebook, `main.ipynb`, which walks through the pipeline step by step with
Plotly plots of the raw, filtered, and wavelet-transformed signals.

## Data

Beats come from the **MIT-BIH Arrhythmia Database** (360 Hz), pre-extracted into `Data/`:

| File | Contents |
| --- | --- |
| `Normal_Train.txt` / `PVC_Train.txt` | 200 training beats each, 300 samples per beat |
| `Normal_Test.txt` / `PVC_Test.txt` | 200 test beats each, 300 samples per beat |
| `NormalECGSig_*.txt` / `PVCECGSig_*.txt` | Individual beats, for trying single-file inference |

Each line is one beat; amplitudes are `|`-separated.

## Tech stack

Python · NumPy · SciPy (`signal.butter`, `signal.lfilter`) · PyWavelets · pandas · scikit-learn · Plotly

## Getting started

```bash
git clone https://github.com/Omar-Elhakim/ECG-Classification.git
cd ECG-Classification

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook main.ipynb
```

Then run the cells top to bottom. Paths in the notebook are relative to the repository root, so launch
Jupyter from there.

## Usage

To classify a beat stored in its own file, use the helper defined near the end of the notebook — it runs
the same filter → wavelet → normalise pipeline on the file and returns a label:

```python
result = predict_signal_from_file(f"{dirPath}/PVCECGSig_150.txt", knn)
print(f"The signal in this file is classified as: {result}")
# The signal in this file is classified as: PVC
```

## Results

On the 400 held-out beats in `Normal_Test.txt` + `PVC_Test.txt` (disjoint from the training beats), the
5-NN classifier separates the two classes perfectly:

```
              precision    recall  f1-score   support

      Normal       1.00      1.00      1.00       200
         PVC       1.00      1.00      1.00       200

    accuracy                           1.00       400
```

This says less about the classifier than about the features: once a beat is band-pass filtered and reduced
to its level-2 `db4` coefficients, Normal and PVC beats occupy clearly separated regions, so even a simple
distance-based classifier gets them all right.

## Project structure

```
main.ipynb           # the whole pipeline: load → filter → wavelet → normalise → KNN → evaluate
Data/                # MIT-BIH beats (train/test splits + individual sample beats)
RequiredDetails.pdf  # original task specification
requirements.txt
LICENSE
```

## License

Released under the [MIT License](LICENSE).
