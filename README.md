# Concept Drift Analysis in Text Data

This repository contains the source code, datasets, and experiments for my Bachelor's thesis: **"Methods for analyzing concept drift in text data" (Metódy pre analýzu concept driftu v textových dátach)**.

## Project Overview

In real-world text streams, such as news articles or social media posts, the underlying data distribution and the context of words often change over time due to real-world events. This phenomenon is known as **concept drift**. If machine learning models fail to adapt to these changes, their predictive accuracy degrades significantly.

The primary goals of this project are:
1. **Detector Evaluation:** Implement and evaluate concept drift detectors on textual data streams with guaranteed (simulated) concept drift. This phase utilizes datasets inspired by the [AdaNEN framework](https://github.com/PouyaGhahramanian/AdaNEN).
2. **Real-World Application:** Apply the calibrated detectors to real-world, timestamped text datasets (categorized by years/months) to identify actual points of concept drift and semantic shift over time.

## Repository Structure

- `data/`: Contains the datasets used for experiments (both simulated streams and real-world chronological data).
- `notebooks/`: Jupyter notebooks for data exploration, visualization, and step-by-step drift detection analysis.
- `src/`: Python source code containing the drift detection algorithms and data processing utilities.

## Getting Started

1. Clone this repository:
   ```bash
   git clone [https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git](https://github.com/Yan-Pasichnyi/Concept-Drift-Analysis-in-Text-Data.git)
   cd Concept-Drift-Analysis-in-Text-DataЁ