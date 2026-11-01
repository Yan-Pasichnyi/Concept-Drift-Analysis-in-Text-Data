"""Plots for the drift detection experiments."""
import matplotlib.pyplot as plt
from evaluation import TRUE_DRIFTS


def plot_drift_result(acc_history, detected, title, color, save_path, detector_name='DDM'):
    """Accuracy curve with true (orange) and detected (red) drifts."""
    plt.figure()
    plt.plot(acc_history, label='Prequential Accuracy (Window=500)', color=color, linewidth=1.5)

    # true drifts
    for t in TRUE_DRIFTS:
        plt.axvline(x=t, color='orange', linestyle=':', alpha=0.7,
                    label='True Drift' if t == TRUE_DRIFTS[0] else "")
    # detected drifts (model resets)
    for d in detected:
        plt.axvline(x=d, color='red', linestyle='--', alpha=0.8,
                    label=f'Detected Drift ({detector_name})' if d == detected[0] else "")

    plt.title(title)
    plt.xlabel('Number of Instances Processed')
    plt.ylabel('Accuracy (%)')
    plt.ylim(40, 100)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Total drifts detected by {detector_name}: {len(detected)}")


def plot_baseline_vs_ddm(results, save_path):
    """One subplot per stream: baseline (gray) vs DDM (blue)."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4.5 * len(results)), sharex=True)
    for ax, (stream_name, runs) in zip(axes, results.items()):
        ax.plot(runs['No detector']['acc_history'], color='gray', linewidth=1.5, label='No detector (baseline)')
        ax.plot(runs['DDM']['acc_history'], color='blue', linewidth=1.5, label='DDM + reset')
        for t in TRUE_DRIFTS:
            ax.axvline(x=t, color='orange', linestyle=':', alpha=0.6,
                       label='True Drift' if t == TRUE_DRIFTS[0] else "")
        ax.set_title(f'{stream_name} drift: baseline vs DDM')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(40, 100)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    axes[-1].set_xlabel('Number of Instances Processed')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()