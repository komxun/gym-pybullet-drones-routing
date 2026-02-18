"""Comparative study script for all 9 experiment models.

Generates publication-quality plots comparing training and testing
performance across all models. Reads training CSVs and test summary CSV.

Usage:
    python -m training.compare_models
    python -m training.compare_models --models D-5 S-5 X-5
"""

import os
import sys
import csv
import argparse

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed. Plots will be skipped.")

ALL_MODELS = [
    "D-5",  "D-10", "D-15",
    "S-5",  "S-10", "S-15",
    "X-5",  "X-10", "X-15",
]

# Color scheme: D=blue, S=green, X=red; lighter for fewer sensors
MODEL_COLORS = {
    "D-5": "#a6cee3", "D-10": "#1f78b4", "D-15": "#08306b",
    "S-5": "#b2df8a", "S-10": "#33a02c", "S-15": "#006d2c",
    "X-5": "#fb9a99", "X-10": "#e31a1c", "X-15": "#800026",
}

MODEL_LINESTYLES = {
    "D-5": "-", "D-10": "-", "D-15": "-",
    "S-5": "--", "S-10": "--", "S-15": "--",
    "X-5": ":", "X-10": ":", "X-15": ":",
}

OUTPUT_DIR = "experiments/plots"


def load_training_csv(model_name: str):
    """Load training_history.csv for a model. Returns dict of lists."""
    path = os.path.join("experiments", model_name, "results", "training_history.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] Training CSV not found: {path}")
        return None

    data = {"episode": [], "reward": [], "steps": [], "outcome": [],
            "intrusion_steps": [], "seconds": []}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["reward"].append(float(row["reward"]))
            data["steps"].append(int(row["steps"]))
            data["outcome"].append(row.get("outcome", ""))
            data["intrusion_steps"].append(int(row.get("intrusion_steps", 0)))
            data["seconds"].append(float(row["seconds"]) if row.get("seconds") else 0)
    return data


def load_test_summary():
    """Load the combined test_summary.csv."""
    path = os.path.join("experiments", "test_summary.csv")
    if not os.path.exists(path):
        print(f"[WARN] Test summary not found: {path}")
        return None

    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k in ["num_sensors", "n_episodes"]:
                if k in row:
                    row[k] = int(row[k])
            for k in ["mean_reward", "std_reward", "mean_steps", "std_steps",
                       "success_rate", "collision_rate", "timeout_rate",
                       "intrusion_rate", "mean_intrusion_steps"]:
                if k in row:
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def rolling_mean(data, window=20):
    """Compute rolling mean with given window size."""
    arr = np.array(data, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_rewards(models, training_data, window=20):
    """Plot training reward curves for all models."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in models:
        d = training_data.get(name)
        if d is None:
            continue
        smoothed = rolling_mean(d["reward"], window)
        episodes = np.arange(window, window + len(smoothed))
        ax.plot(episodes, smoothed,
                color=MODEL_COLORS.get(name, "gray"),
                linestyle=MODEL_LINESTYLES.get(name, "-"),
                label=name, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (rolling {window}-ep mean)")
    ax.set_title("Training Reward Curves")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_rewards.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_outcomes(models, training_data, window=50):
    """Plot running success/collision/timeout rates during training."""
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ["Success Rate", "Collision Rate", "Timeout Rate"]
    outcome_keys = ["reached", "collision", "timeout"]

    for ax, title, key in zip(axes, titles, outcome_keys):
        for name in models:
            d = training_data.get(name)
            if d is None:
                continue
            binary = [1.0 if o == key else 0.0 for o in d["outcome"]]
            if len(binary) < window:
                continue
            smoothed = rolling_mean(binary, window)
            episodes = np.arange(window, window + len(smoothed))
            ax.plot(episodes, smoothed * 100,
                    color=MODEL_COLORS.get(name, "gray"),
                    linestyle=MODEL_LINESTYLES.get(name, "-"),
                    label=name, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"Training {title} (rolling {window}-ep)")
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_outcome_rates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_intrusions(models, training_data, window=20):
    """Plot intrusion steps per episode during training."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in models:
        d = training_data.get(name)
        if d is None:
            continue
        smoothed = rolling_mean(d["intrusion_steps"], window)
        episodes = np.arange(window, window + len(smoothed))
        ax.plot(episodes, smoothed,
                color=MODEL_COLORS.get(name, "gray"),
                linestyle=MODEL_LINESTYLES.get(name, "-"),
                label=name, linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Intrusion Steps (rolling {window}-ep mean)")
    ax.set_title("Training Intrusion Steps")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_intrusions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_test_bar_charts(test_summary, models):
    """Plot grouped bar charts for test metrics."""
    if not HAS_MPL or test_summary is None:
        return

    # Filter to requested models and preserve order
    summary_map = {r["model"]: r for r in test_summary}
    filtered = [summary_map[m] for m in models if m in summary_map]
    if not filtered:
        print("  [SKIP] No test data for requested models.")
        return

    names = [r["model"] for r in filtered]
    x = np.arange(len(names))
    width = 0.6
    colors = [MODEL_COLORS.get(n, "gray") for n in names]

    # --- Success / Collision / Intrusion rates ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics = [
        ("success_rate", "Success Rate (%)", "green"),
        ("collision_rate", "Collision Rate (%)", "red"),
        ("intrusion_rate", "Intrusion Rate (%)", "orange"),
    ]
    for ax, (key, title, _) in zip(axes, metrics):
        vals = [r[key] * 100 for r in filtered]
        bars = ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "test_rates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Mean reward bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [r["mean_reward"] for r in filtered]
    errs = [r["std_reward"] for r in filtered]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Reward")
    ax.set_title("Test Mean Reward (with std)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "test_rewards.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Mean steps bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [r["mean_steps"] for r in filtered]
    errs = [r["std_steps"] for r in filtered]
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Steps")
    ax.set_title("Test Mean Episode Steps (with std)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "test_steps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_sensor_comparison(test_summary, models):
    """Line plots: metric vs num_sensors, grouped by obs_choice."""
    if not HAS_MPL or test_summary is None:
        return

    summary_map = {r["model"]: r for r in test_summary}
    obs_types = {"ray": "D", "sensor": "S", "sector": "X"}
    obs_colors = {"ray": "#1f78b4", "sensor": "#33a02c", "sector": "#e31a1c"}
    sensor_counts = [5, 10, 15]

    metrics = [
        ("success_rate", "Success Rate", 100),
        ("collision_rate", "Collision Rate", 100),
        ("intrusion_rate", "Intrusion Rate", 100),
        ("mean_reward", "Mean Reward", 1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (key, title, scale) in zip(axes, metrics):
        for obs, prefix in obs_types.items():
            vals = []
            for ns in sensor_counts:
                name = f"{prefix}-{ns}"
                if name in summary_map:
                    vals.append(summary_map[name][key] * scale)
                else:
                    vals.append(float("nan"))
            ax.plot(sensor_counts, vals, "o-",
                    color=obs_colors[obs], label=f"{obs} ({prefix})",
                    linewidth=2, markersize=8)
        ax.set_xlabel("Number of Sensors")
        ax.set_ylabel(f"{title}" + (" (%)" if scale == 100 else ""))
        ax.set_title(title)
        ax.set_xticks(sensor_counts)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sensor_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Comparative study of experiment models")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to include. Default: all. Choices: {ALL_MODELS}"
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling window for training curves")
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("COMPARATIVE STUDY - DRL Drone Collision Avoidance")
    print("=" * 70)

    # --- Load training data ---
    print("\nLoading training histories...")
    training_data = {}
    for name in models:
        d = load_training_csv(name)
        if d is not None:
            training_data[name] = d
            print(f"  {name}: {len(d['episode'])} episodes")

    # --- Load test summary ---
    print("\nLoading test summary...")
    test_summary = load_test_summary()
    if test_summary:
        print(f"  {len(test_summary)} models found")

    # --- Generate plots ---
    if not HAS_MPL:
        print("\n[ERROR] matplotlib required for plots. Install: pip install matplotlib")
        return

    print("\nGenerating training plots...")
    if training_data:
        plot_training_rewards(models, training_data, window=args.window)
        plot_training_outcomes(models, training_data, window=min(50, args.window * 2))
        plot_training_intrusions(models, training_data, window=args.window)
    else:
        print("  [SKIP] No training data found.")

    print("\nGenerating test plots...")
    if test_summary:
        plot_test_bar_charts(test_summary, models)
        plot_sensor_comparison(test_summary, models)
    else:
        print("  [SKIP] No test summary found.")

    # --- Print summary table ---
    if test_summary:
        print(f"\n{'='*80}")
        print("TEST RESULTS SUMMARY")
        print(f"{'='*80}")
        header = (f"{'Model':<8} {'Obs':<8} {'#Sens':>5} "
                  f"{'Success':>8} {'Collis':>8} {'Intrus':>8} "
                  f"{'Reward':>10} {'Steps':>8}")
        print(header)
        print("-" * 80)
        summary_map = {r["model"]: r for r in test_summary}
        for name in models:
            if name not in summary_map:
                continue
            s = summary_map[name]
            print(
                f"{s['model']:<8} {s['obs_choice']:<8} {s['num_sensors']:>5} "
                f"{s['success_rate']*100:>7.1f}% {s['collision_rate']*100:>7.1f}% "
                f"{s['intrusion_rate']*100:>7.1f}% "
                f"{s['mean_reward']:>+9.2f} {s['mean_steps']:>7.1f}"
            )

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
