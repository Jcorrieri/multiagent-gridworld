from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics: list[tuple[float, float]], path: str):
    mean_rewards = [m[0] for m in metrics]
    mean_lengths = [m[1] for m in metrics]
    episode = [m[2] for m in metrics]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(episode, mean_rewards, label="Mean Reward", color='blue')
    axs[0].set_ylabel("Mean Reward")
    axs[0].set_title("Training Progress")
    axs[0].grid(True)

    axs[1].plot(episode, mean_lengths, label="Mean Episode Length", color='green')
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Mean Episode Length")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(Path(path) / "metrics_plot.png")
    print(f"Saved training plot to {path}\\metrics_plot.png")
