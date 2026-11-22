from evaluation import get_detectors, evaluate_stream, evaluate_detection_quality


def main():
    print("--- ABRUPT DRIFT (20NG + BGE + HoeffdingTree) ---")
    print(f"{'Detector':<14} {'Accuracy':>9} {'Macro-F1':>9} {'Resets':>7} {'Found':>7} {'Delay':>7} {'Rep':>5} {'Out':>5}")

    # baseline without detector
    base = evaluate_stream(mode='abrupt', detector=None)
    print(f"{'No detector':<14} {base['accuracy']:>8.2f}% {base['macro_f1']:>8.2f}% {0:>7} {'-':>7} {'-':>7} {'-':>5} {'-':>5}")

    # compare drift detectors
    for name, detector in get_detectors().items():
        res = evaluate_stream(mode='abrupt', detector=detector)
        q = evaluate_detection_quality(res['drifts'])
        found_str = f"{q['found']}/{q['total_true']}"
        print(
            f"{name:<14} {res['accuracy']:>8.2f}% {res['macro_f1']:>8.2f}% "
            f"{q['resets']:>7} {found_str:>7} {q['mean_delay']:>7.1f} {q['repeated']:>5} {q['outside']:>5}"
        )


if __name__ == "__main__":
    main()