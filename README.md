# Concept Drift Analysis in Text Data

This repository contains the source code, datasets, and experiments for my Bachelor's thesis: **"Methods for analyzing concept drift in text data" (Metódy pre analýzu concept driftu v textových dátach)**.

## Project Overview

In real-world text streams (e.g., news articles, social media posts), the underlying data distribution and the context of words often change over time due to real-world events. This phenomenon is known as **concept drift**. If machine learning models fail to adapt to these changes, their predictive accuracy degrades significantly. 

The primary goal of my project is to evaluate different drift detection algorithms and adaptation strategies on text streams. I specifically analyze two main types of concept drift dynamics[cite: 5]:
1. **Abrupt Drift:** Sudden changes in the data distribution, representing immediate topic shifts (e.g., breaking news).
2. **Gradual Drift:** Smooth and continuous transitions where old and new concepts overlap before the new concept fully takes over.

## Methodology

### 1. Datasets
Initial experiments utilize vectorized text streams with guaranteed (simulated) concept drifts from the AdaNEN framework[cite: 7]:
- `20ng_abrupt.data`: 20NewsGroup dataset with abrupt topic shifts.
- `20ng_gradual.data`: 20NewsGroup dataset with gradual topic transitions.

### 2. Experimental Setup
I simulate an online machine learning environment where texts arrive sequentially. 
- **Base Model:** `HoeffdingTreeClassifier` (Incremental Decision Tree).
- **Adaptation Strategy:** Active Adaptation (Explicit Detection). I monitor the classification error; when a drift detector signals a change, the base model is completely reset to unlearn obsolete concepts.
- **Drift Detectors:** I compare standard detectors from the `River` library, including ADWIN, DDM, EDDM, and Page-Hinkley.
- **Metrics:** I evaluate the models using Prequential Accuracy and Macro-F1 score (to account for potential class imbalances).

## Initial Results & Findings

I evaluated four explicit drift detectors on both abrupt and gradual drift datasets.

### Experiment 1: Abrupt Drift (`20ng_abrupt.data`)
| Detector | Final Accuracy | Macro-F1 | Drifts Detected |
| :--- | :--- | :--- | :--- |
| **ADWIN** | 71.68% | 71.63% | 7 |
| **DDM** | 72.59% | 72.59% | 6 |
| **EDDM** | 71.03% | 71.02% | 24 |
| **Page-Hinkley** | 69.86% | 69.86% | 9 |

**Conclusion (Abrupt):** **DDM** provided the most stable performance, achieving the highest accuracy and F1 score with a reasonable number of drift points (6). EDDM proved to be overly sensitive, triggering 24 model resets (false alarms), which slightly degraded its learning capacity.

### Experiment 2: Gradual Drift (`20ng_gradual.data`)
| Detector | Final Accuracy | Macro-F1 | Drifts Detected |
| :--- | :--- | :--- | :--- |
| **ADWIN** | 69.91% | 69.84% | 7 |
| **DDM** | 72.20% | 72.19% | 9 |
| **EDDM** | 72.18% | 72.14% | 42 |
| **Page-Hinkley** | 69.57% | 69.55% | 8 |

**Conclusion (Gradual):** 
1. **Performance Drop:** Overall accuracy dropped slightly for most models compared to the abrupt dataset. This is expected, as trees struggle during the gradual transition phase where old and new concepts mix.
2. **EDDM Hyper-sensitivity:** EDDM became highly unstable, detecting **42 drift points**. Since EDDM tracks the distance between errors, the noisy transition period of gradual drift caused frequent false alarms, leading to constant and unnecessary model resets.
3. **Robustness of DDM:** DDM remained my most reliable detector, adapting to the smooth transitions without triggering excessive false alarms and maintaining a top accuracy of 72.20%.

## Getting Started

1. Clone this repository:
   ```bash
   git clone [https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git](https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git)
   cd Concept-Drift-Analysis-in-Text-Data