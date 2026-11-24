# User Guide

## 1. Requirements
- Python 3.13.9
- Libraries from `requirements.txt` (River, scikit-learn, sentence-transformers, NumPy, pandas, Matplotlib, Jupyter)

## 2. Installation
```bash
git clone https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git
cd Concept-Drift-Analysis-in-Text-Data
pip install -r requirements.txt
```

## 3. Data
No manual dataset download is needed. On the first run, `src/data_loader.py` automatically downloads the raw **20 Newsgroups** dataset via `scikit-learn` (removing headers, footers, and quotes), encodes each document independently into a 384-dimensional vector using the frozen `BAAI/bge-small-en-v1.5` model, and saves the embeddings to a local cache (`data/bge_20ng_canonical.npz`). The cache path does not depend on the folder the scripts are started from.

The embeddings are stored in a fixed order. The cache also records the model, the categories, and the filtering settings, so it is recomputed automatically when any of them changes. To force a new computation, delete the cache file.

The `abrupt` and `gradual` streams (15,518 documents, 16 concept drifts) are generated on the fly from these embeddings. The `seed` parameter (default 42) sets the order of the documents in the stream and the random mixing of the old and new rules inside the gradual transition windows. Different seeds reuse the same cache.

The first run takes a few minutes to compute the embeddings (about 4–5 minutes on a CPU); all subsequent runs load the cache in seconds. The cache is not tracked by git.

## 4. Running the detector comparison
Run the scripts from the repository root or from the `src/` folder:
```bash
cd src
python detectors_abrupt.py
python detectors_gradual.py
```
Each script runs seed 42 and prints Accuracy, Macro-F1, resets, found drifts, mean delay, repeated detections, and outside detections for the baseline (no detector) and for ADWIN, DDM, EDDM, and Page-Hinkley. A script takes a few minutes.

## 5. Notebooks
Start Jupyter and open the notebooks from the `notebooks/` folder (the paths in them are relative to this folder). Run them in this order:

| Notebook | What it does | Output |
|----------|--------------|--------|
| `01_explore_20ng_data.ipynb` | loads the cleaned 20 Newsgroups dataset, BGE embeddings, and checks both streams | printed summary |
| `02_visualize_drift_results.ipynb` | prequential accuracy with DDM detections and true drifts (seed 42) | `images/20ng_abrupt_drift.png`, `images/20ng_gradual_drift.png` |
| `03_baseline_and_detection_quality.ipynb` | baseline vs DDM, delay and missed drifts of all detectors (seed 42), and the same experiments repeated over 5 seeds (mean ± std) | `images/baseline_vs_ddm.png`, `multi_seed_results.csv` |

Once the embedding cache is created, notebooks 01 and 02 take a few minutes. Notebook 03 takes considerably longer (roughly half an hour on a CPU), because the multi-seed experiment trains the tree 50 times (5 seeds × 2 streams × 5 methods). For a quicker run, reduce the `SEEDS` list in the multi-seed cell, for example to `[42, 1]`. The file `multi_seed_results.csv` is written to the repository root and is not tracked by git.

## 6. Expected results
After a correct run with seed 42, DDM should give **85.67%** accuracy with **14** detections on the abrupt stream and **85.06%** accuracy with **14** detections on the gradual stream. The tree without a detector reaches about 68.8–68.9%. Small differences in the last digits can appear on other systems or library versions.

Over the 5 seeds (42, 1, 2, 3, 4), DDM gives **85.51 ± 0.44%** (abrupt) and **85.00 ± 0.40%** (gradual) accuracy. The full tables are in the README.