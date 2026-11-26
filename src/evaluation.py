import bisect
from collections import deque

import numpy as np
from river import tree, drift
from sklearn.metrics import f1_score

from data_loader import stream_generator, TRUE_DRIFTS

# Smaller than the 500 distance between true drifts, so the windows
# [t, t + tolerance) never touch each other.
DEFAULT_TOLERANCE = 400


def get_detectors():
    # return fresh detector instances for each experiment
    return {
        'ADWIN': drift.ADWIN(),
        'DDM': drift.binary.DDM(),
        'EDDM': drift.binary.EDDM(),
        'Page-Hinkley': drift.PageHinkley(),
    }


def new_model():
    return tree.HoeffdingTreeClassifier()


def _summarize(y_true, y_pred, window_size):
    window = deque(maxlen=window_size)
    rolling_sum = 0
    history = []
    eval_true, eval_pred = [], []

    for t, p in zip(y_true, y_pred):
        if p is None:
            history.append(np.nan)
            continue
        ok = int(t == p)
        if len(window) == window.maxlen:
            rolling_sum -= window[0]
        window.append(ok)
        rolling_sum += ok
        history.append(100.0 * rolling_sum / len(window))

        eval_true.append(t)
        eval_pred.append(p)

    n = len(eval_true)
    accuracy = 100.0 * sum(int(t == p) for t, p in zip(eval_true, eval_pred)) / n if n else float('nan')
    macro_f1 = 100.0 * f1_score(eval_true, eval_pred, average='macro', zero_division=0) if n else float('nan')

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'acc_history': history,           # rolling accuracy, NaN where no prediction exists
    }


def evaluate_stream(mode='abrupt', detector=None, seed=42, window_size=500,
                      cache_path=None):
    model = new_model()

    y_true, y_pred_all, drifts = [], [], []

    for i, (x, y) in enumerate(stream_generator(mode=mode, seed=seed, cache_path=cache_path)):
        # test-then-train
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        y_true.append(y)
        y_pred_all.append(y_pred)

        if y_pred is None:
            continue


        if detector is not None:
            error = 0 if y_pred == y else 1
            detector.update(error)

            if detector.drift_detected:
                drifts.append(i)
                # reset classifier after drift
                model = new_model()

    result = _summarize(y_true, y_pred_all, window_size)
    result['drifts'] = drifts
    return result


def evaluate_detection_quality(detected_drifts, true_drifts=TRUE_DRIFTS, tolerance=DEFAULT_TOLERANCE):
    true_sorted = sorted(true_drifts)
    delays = {}
    repeated = 0
    outside = 0

    for d in sorted(detected_drifts):
        k = bisect.bisect_right(true_sorted, d) - 1
        if k < 0 or d - true_sorted[k] >= tolerance:
            outside += 1
            continue

        t = true_sorted[k]
        if t in delays:
            repeated += 1
        else:
            delays[t] = d - t

    mean_delay = sum(delays.values()) / len(delays) if delays else float('nan')

    return {
        'resets': len(detected_drifts),
        'found': len(delays),
        'total_true': len(true_sorted),
        'mean_delay': mean_delay,
        'repeated': repeated,
        'outside': outside,
        'missed': [t for t in true_sorted if t not in delays],
    }