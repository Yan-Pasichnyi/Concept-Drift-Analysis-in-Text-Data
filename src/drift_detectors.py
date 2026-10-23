from river import tree, metrics, drift
from data_loader import load_adanen_stream, stream_generator


def run_drift_detection(file_path):
    stream = load_adanen_stream(file_path)

    # init model, metric, and detector
    model = tree.HoeffdingTreeClassifier()
    metric = metrics.Accuracy()
    detector = drift.ADWIN()

    drifts = []

    for i, (x, y) in enumerate(stream_generator(stream)):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        if y_pred is not None:
            metric.update(y, y_pred)
            error = 0 if y_pred == y else 1
            detector.update(error)

            if detector.drift_detected:
                print(f"Drift at index {i}. Acc: {metric}")
                drifts.append(i)
                # reset model
                model = tree.HoeffdingTreeClassifier()

    print(f"Final accuracy: {metric}")
    print(f"Total drifts: {len(drifts)}")
    return drifts


if __name__ == "__main__":
    run_drift_detection('../data/adanen_streams/20ng_abrupt.data')