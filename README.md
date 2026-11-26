# Concept Drift Analysis in Text Data

This repository contains the code and experiments for my Bachelor's project: **"Methods for analyzing concept drift in text data" (Metódy pre analýzu concept driftu v textových dátach)**.

## Abstract

This project tests how four explicit drift detectors (ADWIN, DDM, EDDM, and Page-Hinkley) behave on evolving text streams when paired with a `HoeffdingTreeClassifier` that resets after each detection. Both abrupt and gradual streams are built from the raw 20 Newsgroups dataset (15,518 documents across 4 super-topics, i.e., top-level categories of the newsgroup names, with 16 scheduled concept switches). Each document is embedded on its own with a frozen `BAAI/bge-small-en-v1.5` model (384 features) so that no corpus statistics leak from future texts. Across 5 random seeds, resetting the tree with DDM improves prequential accuracy by roughly 15 percentage points over a tree with no detector (85.51% vs 70.10% on abrupt drift; 85.00% vs 70.05% on gradual drift). DDM catches 13-14 of the 16 true drifts with zero detections outside the matching windows in every seed, whereas EDDM reacts faster and scores about 0.4-0.5 percentage points higher at the price of ~10 extra resets. Right now the drift is synthetic (labels change while the text pool stays fixed); experiments on real timestamped text data are the next step.

## Project Overview

In streaming text applications such as news feeds or social media, topics and label definitions do not stay fixed. When the mapping between text and target class shifts over time, a static or purely incremental model loses accuracy. This problem is called **concept drift** (see [1] for a broad overview of data change, and [2] for a survey focused on text streams).

Here I compare active drift detection methods on two standard drift dynamics:
1. Abrupt drift, where the target mapping switches instantaneously at a fixed point.
2. Gradual drift, where old and new mappings overlap over a transition window before the new one takes over.

## Methodology

### 1. Dataset and Stream Construction
Both streams start from the raw **20 Newsgroups** corpus with `headers`, `footers`, and `quotes` stripped out so the classifier cannot rely on metadata or reply chains. I keep the 17 newsgroups that fall under 4 super-topics (the top-level categories of the newsgroup names: `comp`, `rec`, `sci`, `talk`), dropping `alt.atheism`, `misc.forsale`, and `soc.religion.christian`. Filtering out texts shorter than 20 characters leaves 15,518 documents.

- Feature extraction: every document is encoded independently into a 384-dimensional vector using the frozen `BAAI/bge-small-en-v1.5` transformer. Because the encoder is never fit or fine-tuned on the stream, future documents cannot leak into earlier vectors. Embeddings are computed once in a fixed order and cached locally; stream order is shuffled afterwards by seed.
- Drift simulation: labels follow the AdaNEN [3] switching schedule, with 16 changes spaced 500 instances apart between indices `4000` and `11500`. At each switch, the binary mapping of the 4 super-topics changes. In the `abrupt` stream the switch is immediate; in `gradual`, old and new rules mix linearly across a transition window whose transition width (`width` in the code, listed as `1/DI` in AdaNEN) ranges from 25 to 250 instances. Since only the label mapping changes while the underlying texts come from the same corpus, this is a synthetic real-drift setup.

### 2. Experimental Setup
Documents arrive one by one in a prequential (test-then-train) loop: predict first, update metrics and detector, then train.

- Base model: `HoeffdingTreeClassifier` from `River` [4], using default parameters.
- Adaptation: active reset. Whenever a detector flags drift on the binary error stream (0 for correct, 1 for error), the tree is discarded and replaced with a fresh instance.
- Detectors: ADWIN, DDM, EDDM, and Page-Hinkley from `River` [4], all with default settings.
- Metrics: overall prequential Accuracy, Macro-F1 (which stays within 0.1 percentage points of accuracy across all runs), and detection quality against the 16 known drift points (Found, Mean Delay, Repeated, Outside).
- Random seeds: detailed plots and single-run tables use seed 42. To check how much stream ordering and gradual-window mixing affect the outcome, every method is re-run across 5 seeds (42, 1, 2, 3, 4).

## Results

All numbers and plots in this section and in Baseline and Detection Quality are for seed 42 unless stated otherwise. Figures 1 and 2 plot rolling prequential accuracy (window = 500) together with DDM reset points and the 16 true drift indices.

### Experiment 1: Abrupt Drift

![Abrupt Drift Performance](images/20ng_abrupt_drift.png)
*Figure 1: Prequential accuracy under abrupt concept drift (seed 42). Red dashed lines show drifts detected by DDM (model resets), orange dotted lines show the true drift positions.*

| Detector | Accuracy | Macro-F1 | Resets |
| :--- | :---: | :---: | :---: |
| **ADWIN** | 82.94% | 82.93% | 9 |
| **DDM** | 85.67% | 85.66% | 14 |
| **EDDM** | 86.61% | 86.60% | 24 |
| **Page-Hinkley** | 82.71% | 82.69% | 9 |

In the stationary prefix (indices 0-4000), rolling accuracy climbs quickly and sits between 85% and 90%. Once the switches start, DDM fires after 14 of the 16 drifts, missing only 9000 and 9500. Each reset causes a brief cold-start drop (down to ~72% around index 8500), but the fresh tree recovers to 82-91% much faster than an unreset tree would. After the last switch at 11500, accuracy stabilizes around 85-92%.

Why the gap between detectors? On abrupt switches, faster detection saves predictions: EDDM reacts earlier than DDM and catches the two switches DDM missed, which likely helps explain its edge of 0.94 percentage points. That edge comes with 24 resets instead of 14, meaning several unnecessary cold starts. ADWIN and Page-Hinkley trigger only 9 resets total, so the tree spends long stretches fitting outdated labels, leaving them roughly 2.7-3.9 percentage points behind DDM and EDDM.

### Experiment 2: Gradual Drift

![Gradual Drift Performance](images/20ng_gradual_drift.png)
*Figure 2: Prequential accuracy under gradual concept drift (seed 42). Red dashed lines show drifts detected by DDM (model resets), orange dotted lines show the true drift positions.*

| Detector | Accuracy | Macro-F1 | Resets |
| :--- | :---: | :---: | :---: |
| **ADWIN** | 82.74% | 82.73% | 9 |
| **DDM** | 85.06% | 85.04% | 14 |
| **EDDM** | 85.20% | 85.19% | 21 |
| **Page-Hinkley** | 82.42% | 82.41% | 9 |

Every detector loses a small amount of accuracy compared to the abrupt stream (about 0.2-1.4 percentage points), because mixed old and new labels inside the transition windows create unavoidable error before the new rule dominates. DDM again misses the switches around 9000 and 9500, where rolling accuracy hits its lowest point (~70%), then recovers past 85-90% once the stream becomes stationary after 11500.

What stands out here is how much EDDM's advantage shrinks: down to 0.14 percentage points over DDM (85.20% vs 85.06%) while still triggering 7 more resets. Because gradual transitions spread errors out instead of producing a single sharp spike, EDDM's aggressive resetting likely pays off less clearly than it did on abrupt drift. ADWIN and Page-Hinkley remain 2.3-2.8 percentage points behind DDM and EDDM, triggering too few resets to keep up with 16 switches.

## Baseline and Detection Quality

To verify that explicit resetting is worth the cold-start cost, I compared DDM against the exact same Hoeffding Tree with no detector. Because the 16 true switch indices are known, I also scored each detector's timing. Every detection is matched to the nearest preceding true drift `t`. A drift counts as found if a detection lands in `[t, t + 400)`, with delay measured to the first such hit. Extra hits for an already found drift are counted as repeated; hits before index 4000 or at least 400 instances after the nearest preceding drift are counted as outside. The 400-instance cutoff stays below the 500-instance gap between switches so windows never touch, though matching is still approximate: a very late detection arriving after `t + 500` looks identical to an early detection of the next drift. Mean delay is averaged only over found drifts, so values from detectors that catch very different numbers of drifts (for example, ADWIN at 5/16 vs DDM at 14/16) are not directly comparable.

![Baseline vs DDM](images/baseline_vs_ddm.png)
*Figure 3: Prequential accuracy of the tree without a detector (gray) and with DDM resets (blue), seed 42. Orange dotted lines show the true drifts. The vertical axis is cut at 40%.*

| Stream | Method | Accuracy | Macro-F1 | Resets |
| :--- | :--- | :---: | :---: | :---: |
| Abrupt | No detector | 68.81% | 68.80% | 0 |
| Abrupt | DDM | 85.67% | 85.66% | 14 |
| Gradual | No detector | 68.93% | 68.91% | 0 |
| Gradual | DDM | 85.06% | 85.04% | 14 |

| Stream | Detector | Resets | Found | Mean delay | Repeated | Outside |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Abrupt | DDM | 14 | 14/16 | 80.4 | 0 | 0 |
| Abrupt | ADWIN | 9 | 5/16 | 101.0 | 0 | 4 |
| Abrupt | EDDM | 24 | 16/16 | 51.1 | 3 | 5 |
| Abrupt | Page-Hinkley | 9 | 7/16 | 172.9 | 0 | 2 |
| Gradual | DDM | 14 | 14/16 | 138.2 | 0 | 0 |
| Gradual | ADWIN | 9 | 7/16 | 90.7 | 0 | 2 |
| Gradual | EDDM | 21 | 13/16 | 74.5 | 3 | 5 |
| Gradual | Page-Hinkley | 9 | 8/16 | 143.6 | 0 | 1 |

Without resets, the tree cannot overwrite its earlier split statistics fast enough, so rolling accuracy collapses below 40% several times inside the drift zone (around 6500, 9000-9500, and 10500). It only climbs back toward 85-90% in the stationary tail after 11500, where no more switches happen. Resetting with DDM keeps rolling accuracy at or above ~70% throughout the entire drift zone, adding +16.86 percentage points on abrupt and +16.13 percentage points on gradual.

Looking at detection quality explains how the four detectors separate into three behaviors:
- DDM waits for a statistically large jump in error rate, which gives zero outside and zero repeated hits here, at the cost of missing two switches.
- EDDM tracks the distance between consecutive errors, so it fires fastest (51.1 abrupt, 74.5 gradual) but also fires extra detections (3 repeated and 5 outside on both streams).
- ADWIN compares the means of two sub-windows inside an adaptive window, while Page-Hinkley accumulates deviations of the observed error from its running mean until a threshold is crossed. Both fire far less often (5-8/16 found, with 1-4 outside detections), which leaves their overall accuracy behind DDM and EDDM.

The clearest difference between abrupt and gradual drift is not accuracy, but delay: DDM jumps from 80.4 to 138.2 instances and EDDM from 51.1 to 74.5, because errors accumulate more slowly while old and new rules are mixed.

## Stability Across Random Seeds

Single-seed numbers can be misleading when stream order changes which documents land right after a switch. I repeated all runs across 5 seeds (42, 1, 2, 3, 4) and report mean ± standard deviation below. Embeddings are reused from the same cache, so only document order and gradual-window mixing change. Per-seed records are written to `multi_seed_results.csv` by notebook 03 (excluded from git).

| Stream | Method | Accuracy | Macro-F1 | Resets | Found (of 16) | Mean delay | Repeated | Outside |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Abrupt | No detector | 70.10 ± 0.88 | 70.08 ± 0.88 | 0 | - | - | - | - |
| Abrupt | ADWIN | 83.39 ± 0.92 | 83.37 ± 0.92 | 9.40 ± 1.14 | 7.80 ± 1.92 | 92.35 ± 11.45 | 0.00 ± 0.00 | 1.60 ± 1.67 |
| Abrupt | DDM | 85.51 ± 0.44 | 85.49 ± 0.45 | 14.20 ± 0.84 | 13.60 ± 0.55 | 77.10 ± 21.62 | 0.60 ± 0.89 | 0.00 ± 0.00 |
| Abrupt | EDDM | 86.04 ± 0.59 | 86.02 ± 0.60 | 24.00 ± 3.81 | 14.80 ± 1.30 | 51.39 ± 11.31 | 4.60 ± 2.07 | 4.60 ± 3.51 |
| Abrupt | Page-Hinkley | 82.49 ± 0.35 | 82.46 ± 0.37 | 8.80 ± 1.10 | 7.40 ± 0.89 | 143.75 ± 30.60 | 0.00 ± 0.00 | 1.40 ± 0.89 |
| Gradual | No detector | 70.05 ± 0.74 | 70.02 ± 0.74 | 0 | - | - | - | - |
| Gradual | ADWIN | 83.26 ± 0.67 | 83.25 ± 0.67 | 10.00 ± 1.22 | 8.20 ± 0.84 | 129.81 ± 30.71 | 0.00 ± 0.00 | 1.80 ± 1.48 |
| Gradual | DDM | 85.00 ± 0.40 | 84.98 ± 0.40 | 13.60 ± 1.52 | 13.20 ± 0.84 | 125.49 ± 18.16 | 0.40 ± 0.89 | 0.00 ± 0.00 |
| Gradual | EDDM | 85.38 ± 0.34 | 85.37 ± 0.34 | 24.00 ± 5.48 | 14.20 ± 1.30 | 76.38 ± 8.05 | 6.20 ± 3.11 | 3.60 ± 4.04 |
| Gradual | Page-Hinkley | 82.41 ± 0.50 | 82.39 ± 0.50 | 9.40 ± 0.55 | 7.60 ± 1.14 | 172.64 ± 20.36 | 0.00 ± 0.00 | 1.80 ± 0.84 |

Running multiple seeds clarifies two things that seed 42 alone obscured:
- Seed 42 slightly overstated the adaptation gain. Its baseline happened to be weaker than average (68.8-68.9% vs ~70.1% mean), so the mean over 5 seeds for DDM's gain is +15.4 percentage points on abrupt and +15.0 percentage points on gradual rather than +16.9 / +16.1 percentage points.
- DDM is not completely free of repeated triggers across seeds (0.4-0.6 on average), though it still never fired outside the matching windows in any of the 5 runs.

Otherwise, the ranking holds steadily across seeds. EDDM is ahead in all 5 abrupt seeds (by 0.32-0.94 percentage points, mean 0.53 percentage points) and in 4 of 5 gradual seeds (by -0.03 to 0.92 percentage points, paired mean 0.39 percentage points). Still, that edge of about 0.4-0.5 percentage points comes with roughly 10 extra resets and much wider variance in reset count (std 3.81-5.48 vs 0.84-1.52 for DDM). Finally, mean delay has a wide spread across seeds (std up to ~30 instances), and because each detector averages over a different number of found drifts, small delay differences between ADWIN, DDM, and Page-Hinkley should be treated as noisy rather than strict rankings.

## Project Structure

Instructions for installation and running the project are in [`doc/user_guide.md`](doc/user_guide.md).

```text
├── data/
│   └── real_data/             # real text data (not used in the experiments yet)
├── doc/
│   └── user_guide.md          # installation, running the scripts and notebooks
├── images/                    # figures used in this README
├── notebooks/
│   ├── 01_explore_20ng_data.ipynb
│   ├── 02_visualize_drift_results.ipynb
│   └── 03_baseline_and_detection_quality.ipynb   # also the multi-seed experiment
├── src/
│   ├── data_loader.py         # raw 20NG loading, BGE embeddings, drift streams
│   ├── evaluation.py          # prequential evaluation, detectors, baselines, detection quality
│   ├── plotting.py            # plots used in the notebooks
│   ├── detectors_abrupt.py    # detector comparison on the abrupt stream
│   └── detectors_gradual.py   # detector comparison on the gradual stream
├── requirements.txt
└── README.md
```

## References

[1] J.-L. Li, C.-F. Hsu, M.-C. Chang, W.-C. Chen. *A Comprehensive Review of Machine Learning Advances on Data Change: A Cross-Field Perspective.* arXiv:2402.12627, 2024. [https://arxiv.org/abs/2402.12627](https://arxiv.org/abs/2402.12627)

[2] C. M. Garcia, R. Abilio, A. L. Koerich, A. de Souza Britto, J. P. Barddal. *Concept Drift Adaptation in Text Stream Mining Settings: A Systematic Review.* ACM Transactions on Intelligent Systems and Technology, 16(2), 2025. [https://doi.org/10.1145/3704922](https://doi.org/10.1145/3704922)

[3] P. Ghahramanian, S. Bakhshi, H. Bonab, F. Can. *A Novel Neural Ensemble Architecture for On-the-fly Classification of Evolving Text Streams.* ACM Transactions on Knowledge Discovery from Data, 18(4), Article 101, 2024. [https://doi.org/10.1145/3639054](https://doi.org/10.1145/3639054). Data streams and code: [https://github.com/PouyaGhahramanian/AdaNEN](https://github.com/PouyaGhahramanian/AdaNEN)

[4] J. Montiel et al. *River: machine learning for streaming data in Python.* Journal of Machine Learning Research, 22, 2021. Documentation: [https://riverml.xyz](https://riverml.xyz)