from river import tree, metrics, drift
from data_loader import load_adanen_stream, stream_generator


def evaluate_detector(file_path, detector, detector_name):
    stream = load_adanen_stream(file_path)

    # init model and multiple metrics
    model = tree.HoeffdingTreeClassifier()
    acc = metrics.Accuracy()
    f1 = metrics.MacroF1()

    drifts = []

    for i, (x, y) in enumerate(stream_generator(stream)):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        if y_pred is not None:
            # update both metrics
            acc.update(y, y_pred)
            f1.update(y, y_pred)

            error = 0 if y_pred == y else 1
            detector.update(error)

            if detector.drift_detected:
                drifts.append(i)
                # reset model
                model = tree.HoeffdingTreeClassifier()

    print(f"[{detector_name}] Final Acc: {acc}, F1: {f1}")
    print(f"[{detector_name}] Total drifts found: {len(drifts)}\n")
    return drifts


def run_comparison(file_path):
    print(f"Starting comparison on GRADUAL dataset: {file_path}\n")

    detectors = {
        "ADWIN": drift.ADWIN(),
        "DDM": drift.binary.DDM(),
        "EDDM": drift.binary.EDDM(),
        "Page-Hinkley": drift.PageHinkley()
    }

    for name, det in detectors.items():
        evaluate_detector(file_path, det, name)


if __name__ == "__main__":
    run_comparison('../data/adanen_streams/20ng_gradual.data')