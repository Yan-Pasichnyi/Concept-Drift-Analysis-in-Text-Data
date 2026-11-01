# User Guide

## 1. Requirements
- Python [3.13.9]
- Libraries from `requirements.txt` (River, NumPy, Matplotlib, Jupyter)

## 2. Installation
```bash
git clone https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git
cd Concept-Drift-Analysis-in-Text-Data
pip install -r requirements.txt
```

## 3. Data
The streams are in `data/adanen_streams/` (`20ng_abrupt.data`, `20ng_gradual.data`) and come from the AdaNEN repository (see References in the README). Each instance has 300 features (document embedding) and a class label. The loader is in `src/data_loader.py`.

## 4. Running the detector comparison
Run the scripts from the `src/` folder (the data paths are relative to it):
```bash
cd src
python detectors_abrupt.py
python detectors_gradual.py
```
Each script prints accuracy, Macro-F1 and the number of detected drifts for ADWIN, DDM, EDDM and Page-Hinkley.

## 5. Notebooks
Start Jupyter and open the notebooks from the `notebooks/` folder (the paths in them are relative to this folder). Run them in this order:

| Notebook | What it does | Output |
|----------|--------------|--------|
| `01_explore_adanen_data.ipynb` | loads both streams and checks size and features | printed summary |
| `02_visualize_drift_results.ipynb` | prequential accuracy with DDM detections and true drifts | `images/20ng_abrupt_drift.png`, `images/20ng_gradual_drift.png` |
| `03_baseline_and_detection_quality.ipynb` | baseline vs DDM, delay and missed drifts of all detectors | `images/baseline_vs_ddm.png` |

All notebooks together take a few minutes to run.

## 6. Expected results
After a correct run, DDM should give 72.59% accuracy with 6 detections on the abrupt stream and 72.20% with 9 detections on the gradual stream. The full tables are in the README.