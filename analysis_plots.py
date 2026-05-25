from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _ensure_results_dir():
    RESULTS_DIR.mkdir(exist_ok=True)


def _write_interpretation(path, title, lines):
    txt_path = path.with_suffix(".txt")
    txt_path.write_text(title + "\n" + "=" * len(title) + "\n\n" + "\n".join(lines), encoding="utf-8")


def _rolling_mean(values, window):
    if len(values) == 0:
        return np.array([])
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def save_training_plots(rewards, actions, output_prefix="training"):
    _ensure_results_dir()
    rewards = np.asarray(rewards, dtype=float)
    actions = np.asarray(actions, dtype=int)

    if len(rewards) == 0:
        return

    cumulative_reward = np.cumsum(rewards)
    rolling_reward = _rolling_mean(rewards, window=1000)

    path = RESULTS_DIR / f"{output_prefix}_reward_over_time.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_reward, label="Cumulative reward", color="#1f77b4")
    plt.title("Training cumulative reward over time")
    plt.xlabel("Training step")
    plt.ylabel("Cumulative reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "Training cumulative reward over time",
        [
            "This graph shows the total reward accumulated during training.",
            "An upward trend means the agent is collecting more positive decisions than penalties.",
            "Large drops usually mean the agent made expensive mistakes, especially letting attacks go to production.",
            f"Final cumulative reward: {cumulative_reward[-1]:.2f}.",
        ],
    )

    path = RESULTS_DIR / f"{output_prefix}_rolling_reward.png"
    plt.figure(figsize=(12, 6))
    x = np.arange(len(rolling_reward)) + 1
    plt.plot(x, rolling_reward, label="Rolling mean reward (1000 steps)", color="#2ca02c")
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.title("Training rolling reward")
    plt.xlabel("Training step")
    plt.ylabel("Average reward")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "Training rolling reward",
        [
            "This graph smooths the reward with a 1000-step moving average.",
            "It is easier to read than raw reward because individual network flows can be noisy.",
            "Values above zero mean the recent policy is generally making profitable security decisions.",
            f"Last rolling average: {rolling_reward[-1]:.4f}.",
        ],
    )

    action_counts = np.bincount(actions, minlength=2)
    path = RESULTS_DIR / f"{output_prefix}_action_distribution.png"
    plt.figure(figsize=(7, 5))
    plt.bar(["PRODUCTION", "HONEYPOT"], action_counts, color=["#4c78a8", "#f58518"])
    plt.title("Training action distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    total = max(int(action_counts.sum()), 1)
    _write_interpretation(
        path,
        "Training action distribution",
        [
            "This graph shows how often the agent selected each action during training.",
            "A heavy PRODUCTION bias may miss attacks; a heavy HONEYPOT bias may create many false positives.",
            f"PRODUCTION: {action_counts[0]} ({action_counts[0] / total:.2%}).",
            f"HONEYPOT: {action_counts[1]} ({action_counts[1] / total:.2%}).",
        ],
    )


def save_evaluation_plots(y_true, y_pred, rewards, output_prefix="evaluation"):
    _ensure_results_dir()
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rewards = np.asarray(rewards, dtype=float)

    if len(y_true) == 0:
        return

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    accuracy = (tn + tp) / len(y_true)

    path = RESULTS_DIR / f"{output_prefix}_confusion_matrix.png"
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion matrix")
    plt.xticks([0, 1], ["PRODUCTION", "HONEYPOT"])
    plt.yticks([0, 1], ["BENIGN", "ATTACK"])
    plt.xlabel("Predicted action")
    plt.ylabel("True class")
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "Confusion matrix",
        [
            "This graph compares the model decisions with the true traffic labels.",
            "Top-left is benign traffic correctly sent to production.",
            "Bottom-right is attack traffic correctly redirected to the honeypot.",
            "Bottom-left is the most dangerous error: an attack sent to production.",
            f"Accuracy: {accuracy:.2%}. TN={tn}, FP={fp}, FN={fn}, TP={tp}.",
        ],
    )

    cumulative_reward = np.cumsum(rewards)
    path = RESULTS_DIR / f"{output_prefix}_reward_over_time.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_reward, color="#1f77b4")
    plt.title("Evaluation cumulative reward over time")
    plt.xlabel("Evaluated flow")
    plt.ylabel("Cumulative reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "Evaluation cumulative reward over time",
        [
            "This graph shows how much reward the trained policy accumulates on the test dataset.",
            "A steady increase means the policy is making mostly correct decisions.",
            "Sudden decreases indicate costly mistakes, especially false negatives.",
            f"Final cumulative reward: {cumulative_reward[-1]:.2f}.",
        ],
    )

    correct = (y_true == y_pred).astype(float)
    rolling_accuracy = _rolling_mean(correct, window=1000)
    path = RESULTS_DIR / f"{output_prefix}_rolling_accuracy.png"
    plt.figure(figsize=(12, 6))
    x = np.arange(len(rolling_accuracy)) + 1
    plt.plot(x, rolling_accuracy, color="#2ca02c")
    plt.ylim(0, 1.02)
    plt.title("Evaluation rolling accuracy")
    plt.xlabel("Evaluated flow")
    plt.ylabel("Accuracy over last 1000 flows")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "Evaluation rolling accuracy",
        [
            "This graph shows local accuracy over a moving window of 1000 flows.",
            "It helps detect parts of the dataset where the model performs worse than its global score.",
            "Drops can indicate a harder attack family, distribution shift, or a sequence of false alarms.",
            f"Last rolling accuracy: {rolling_accuracy[-1]:.2%}.",
        ],
    )

    predicted_counts = np.bincount(y_pred, minlength=2)
    true_counts = np.bincount(y_true, minlength=2)
    path = RESULTS_DIR / f"{output_prefix}_class_distribution.png"
    width = 0.35
    indexes = np.arange(2)
    plt.figure(figsize=(8, 5))
    plt.bar(indexes - width / 2, true_counts, width, label="True labels", color="#54a24b")
    plt.bar(indexes + width / 2, predicted_counts, width, label="Predicted actions", color="#e45756")
    plt.xticks(indexes, ["BENIGN/PRODUCTION", "ATTACK/HONEYPOT"])
    plt.title("True labels vs predicted actions")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    _write_interpretation(
        path,
        "True labels vs predicted actions",
        [
            "This graph compares the real class distribution with the agent action distribution.",
            "If predicted HONEYPOT is much higher than true ATTACK, the model is probably too aggressive.",
            "If predicted HONEYPOT is much lower than true ATTACK, the model may miss attacks.",
            f"True BENIGN={true_counts[0]}, true ATTACK={true_counts[1]}.",
            f"Predicted PRODUCTION={predicted_counts[0]}, predicted HONEYPOT={predicted_counts[1]}.",
        ],
    )
