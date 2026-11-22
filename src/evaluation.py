from river import tree, metrics, drift, utils
from data_loader import stream_generator, TRUE_DRIFTS


def get_detectors():
    # return fresh detector instances for each experiment
    return {
        'ADWIN': drift.ADWIN(),
        'DDM': drift.binary.DDM(),
        'EDDM': drift.binary.EDDM(),
        'Page-Hinkley': drift.PageHinkley(),
    }


def evaluate_stream(mode='abrupt', detector=None, window_size=500, cache_path='../data/bge_20ng_cache.npz'):
    model = tree.HoeffdingTreeClassifier()
    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()
    rolling_acc = utils.Rolling(metrics.Accuracy, window_size=window_size)

    acc_history = []
    drifts = []

    for i, (x, y) in enumerate(stream_generator(mode=mode, cache_path=cache_path)):
        # test-then-train
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        if y_pred is not None:
            acc.update(y, y_pred)
            f1.update(y, y_pred)
            rolling_acc.update(y, y_pred)
            acc_history.append(rolling_acc.get() * 100)

            if detector is not None:
                error = 0 if y_pred == y else 1
                detector.update(error)

                if detector.drift_detected:
                    drifts.append(i)
                    # reset classifier after drift
                    model = tree.HoeffdingTreeClassifier()
        else:
            acc_history.append(0.0)

    return {
        'accuracy': acc.get() * 100,
        'macro_f1': f1.get() * 100,
        'drifts': drifts,
        'acc_history': acc_history,
    }


def evaluate_detection_quality(detected_drifts, true_drifts=TRUE_DRIFTS, tolerance=500):
    found_delays = {}
    repeated = 0
    outside = 0

    for d in detected_drifts:
        matched_drift = None

        # check if detection falls within [true_drift, true_drift + tolerance)
        for t in true_drifts:
            if t <= d < t + tolerance:
                matched_drift = t
                break

        if matched_drift is None:
            outside += 1
        elif matched_drift not in found_delays:
            found_delays[matched_drift] = d - matched_drift
        else:
            repeated += 1

    if len(found_delays) > 0:
        mean_delay = sum(found_delays.values()) / len(found_delays)
    else:
        mean_delay = 0.0

    return {
        'resets': len(detected_drifts),
        'found': len(found_delays),
        'total_true': len(true_drifts),
        'mean_delay': mean_delay,
        'repeated': repeated,
        'outside': outside,
    }