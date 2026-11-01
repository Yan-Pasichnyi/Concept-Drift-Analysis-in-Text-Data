"""Prequential evaluation of drift detectors on AdaNEN streams."""
import os
from river import tree, metrics, drift, utils
from data_loader import load_adanen_stream, stream_generator

# stream paths (work from any folder)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'adanen_streams')
STREAMS = {
    'Abrupt': os.path.join(DATA_DIR, '20ng_abrupt.data'),
    'Gradual': os.path.join(DATA_DIR, '20ng_gradual.data'),
}

# detector classes, create a new one for every run
DETECTORS = {
    'ADWIN': drift.ADWIN,
    'DDM': drift.binary.DDM,
    'EDDM': drift.binary.EDDM,
    'Page-Hinkley': drift.PageHinkley,
}

# true drift positions (AdaNEN description, about every 500 instances)
TRUE_DRIFTS = list(range(4000, 11501, 500))


def evaluate_stream(file_path, detector=None, window_size=500):
    """Prequential evaluation, detector=None is the baseline without detection."""
    stream = load_adanen_stream(file_path)
    model = tree.HoeffdingTreeClassifier()

    rolling = utils.Rolling(metrics.Accuracy, window_size=window_size)
    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()

    acc_history, drift_points = [], []

    for i, (x, y) in enumerate(stream_generator(stream)):
        # test first, then train
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        if y_pred is not None:
            rolling.update(y, y_pred)
            acc.update(y, y_pred)
            f1.update(y, y_pred)
            # rolling accuracy only when the window is full
            acc_history.append(rolling.get() * 100 if i >= window_size else float('nan'))

            if detector is not None:
                detector.update(0 if y_pred == y else 1)
                if detector.drift_detected:
                    drift_points.append(i)
                    model = tree.HoeffdingTreeClassifier()  # reset model
        else:
            acc_history.append(float('nan') if not acc_history else acc_history[-1])

    return {'acc_history': acc_history, 'drifts': drift_points,
            'accuracy': acc.get() * 100, 'macro_f1': f1.get() * 100}


def drift_detection_quality(detected, true_drifts=TRUE_DRIFTS, tolerance=500):
    """Delay, missed, repeated and outside detections for one detector."""
    delays, missed, matched = [], [], set()
    for t in true_drifts:
        # true drift is found if the detector fires in [t, t + tolerance)
        in_window = [d for d in detected if t <= d < t + tolerance]
        if in_window:
            delays.append(in_window[0] - t)
            matched.add(in_window[0])
        else:
            missed.append(t)
    # detections that are not the first one in a window
    extra = [d for d in detected if d not in matched]
    in_any_window = lambda d: any(t <= d < t + tolerance for t in true_drifts)
    repeated = [d for d in extra if in_any_window(d)]
    outside = [d for d in extra if not in_any_window(d)]
    return {'delays': delays, 'missed': missed, 'repeated': repeated, 'outside': outside}


def print_accuracy_table(results):
    """Accuracy, Macro-F1 and resets for every method."""
    print(f"{'Stream':<8} {'Method':<12} {'Accuracy':>9} {'Macro-F1':>9} {'Resets':>7}")
    for stream_name, runs in results.items():
        for method, r in runs.items():
            print(f"{stream_name:<8} {method:<12} {r['accuracy']:>8.2f}% {r['macro_f1']:>8.2f}% {len(r['drifts']):>7}")


def print_quality_table(results):
    """Found drifts, delay, repeated and outside detections for every detector."""
    print(f"{'Stream':<8} {'Detector':<13} {'Accuracy':>9} {'Macro-F1':>9} {'Resets':>7} {'Found':>7} {'Mean delay':>11} {'Repeated':>9} {'Outside':>8}")
    for stream_name, runs in results.items():
        for method, r in runs.items():
            if method == 'No detector':
                continue
            q = drift_detection_quality(r['drifts'])
            mean_delay = sum(q['delays']) / len(q['delays']) if q['delays'] else float('nan')
            found = f"{len(q['delays'])}/{len(TRUE_DRIFTS)}"
            print(f"{stream_name:<8} {method:<13} {r['accuracy']:>8.2f}% {r['macro_f1']:>8.2f}% {len(r['drifts']):>7} "
                  f"{found:>7} {mean_delay:>11.0f} {len(q['repeated']):>9} {len(q['outside']):>8}")

    # missed drifts of DDM
    for stream_name, runs in results.items():
        print(f"\nMissed true drifts for DDM ({stream_name}):", drift_detection_quality(runs['DDM']['drifts'])['missed'])