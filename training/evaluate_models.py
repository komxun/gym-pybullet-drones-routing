"""Batch evaluation script for all 9 experiment models.

Loads each trained model, runs N evaluation episodes, and exports
per-model test metrics (collision rate, intrusion rate, success rate,
mean reward, mean steps) to a single CSV for comparative analysis.

Usage:
    python -m training.evaluate_models
    python -m training.evaluate_models --models D-5 S-10 --episodes 50
"""

import os
import sys
import csv
import time
import argparse
from collections import Counter

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from training.configs.config import load_config, _deep_merge, _dict_to_namespace, Config
from training.networks.fc_dueling_q import FCDuelingQ
from training.envs.env_factory import make_env

ALL_MODELS = [
    "D-5",  "D-10", "D-15",
    "S-5",  "S-10", "S-15",
    "X-5",  "X-10", "X-15",
]

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs", "experiments")


def select_action(model, state, device="cpu"):
    """Greedy action selection."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_t)
    return q_values.argmax(dim=-1).item()


def load_experiment_config(model_name: str) -> Config:
    """Load default config merged with experiment overrides."""
    default_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    experiment_path = os.path.join(CONFIGS_DIR, f"{model_name}.yaml")

    cfg = load_config(default_path)
    with open(experiment_path, "r") as f:
        overrides = yaml.safe_load(f)
    merged = _deep_merge(cfg.to_dict(), overrides)
    ns = _dict_to_namespace(merged)
    return Config(**vars(ns))


def evaluate_model(model_name: str, n_episodes: int = 50, gui: bool = False):
    """Evaluate a single trained model and return per-episode metrics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    cfg = load_experiment_config(model_name)

    # Find best model
    model_path = os.path.join(cfg.paths.model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    # Create environment
    env = make_env(cfg, gui=gui)
    nS = env.observation_space.shape[0]
    nA = env.action_space.n

    # Load model
    device = "cpu"
    model = FCDuelingQ(
        input_dim=nS,
        output_dim=nA,
        hidden_dims=tuple(cfg.network.hidden_dims),
        device=device,
    )
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device), weights_only=True)
    )
    model.eval()
    print(f"  Model loaded: {nS} obs -> {nA} actions")
    print(f"  obs_choice={cfg.env.obs_choice}, num_sensors={cfg.sensor.num_sensors}")

    # Run evaluation episodes
    episodes_data = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        step = 0
        intrusion_count = 0

        while True:
            action = select_action(model, state, device)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            if info.get("intrusion", False):
                intrusion_count += 1

            if terminated or truncated:
                # Determine outcome
                if info.get("reached", False):
                    outcome = "reached"
                elif info.get("collision", False):
                    outcome = "collision"
                else:
                    outcome = "timeout"
                break

        episodes_data.append({
            "episode": ep + 1,
            "reward": ep_reward,
            "steps": step,
            "outcome": outcome,
            "intrusion_steps": intrusion_count,
            "d2dest": info.get("d2dest", float("nan")),
        })

        print(
            f"  ep {ep+1:3d}/{n_episodes}: "
            f"r={ep_reward:+8.2f}, steps={step:4d}, {outcome}, "
            f"intrusions={intrusion_count}"
        )

    env.close()

    # Aggregate metrics
    rewards = [e["reward"] for e in episodes_data]
    steps = [e["steps"] for e in episodes_data]
    outcomes = [e["outcome"] for e in episodes_data]
    intrusions = [e["intrusion_steps"] for e in episodes_data]
    counts = Counter(outcomes)

    summary = {
        "model": model_name,
        "obs_choice": cfg.env.obs_choice,
        "num_sensors": cfg.sensor.num_sensors,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_steps": float(np.mean(steps)),
        "std_steps": float(np.std(steps)),
        "success_rate": counts.get("reached", 0) / n_episodes,
        "collision_rate": counts.get("collision", 0) / n_episodes,
        "timeout_rate": counts.get("timeout", 0) / n_episodes,
        "intrusion_rate": sum(1 for i in intrusions if i > 0) / n_episodes,
        "mean_intrusion_steps": float(np.mean(intrusions)),
    }

    print(f"\n  Summary for {model_name}:")
    print(f"    Success:   {summary['success_rate']*100:.1f}%")
    print(f"    Collision: {summary['collision_rate']*100:.1f}%")
    print(f"    Timeout:   {summary['timeout_rate']*100:.1f}%")
    print(f"    Intrusion: {summary['intrusion_rate']*100:.1f}%")
    print(f"    Reward:    {summary['mean_reward']:.2f} +/- {summary['std_reward']:.2f}")

    # Save per-episode CSV for this model
    ep_csv_path = os.path.join(cfg.paths.results_dir, "test_episodes.csv")
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    with open(ep_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episodes_data[0].keys())
        writer.writeheader()
        writer.writerows(episodes_data)
    print(f"  Per-episode CSV -> {ep_csv_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate all 9 experiment models")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Specific models to evaluate. Default: all. Choices: {ALL_MODELS}"
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per model")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS
    for m in models:
        if m not in ALL_MODELS:
            print(f"[ERROR] Unknown model: {m}. Valid: {ALL_MODELS}")
            sys.exit(1)

    print("=" * 70)
    print("BATCH EVALUATION - DRL Drone Collision Avoidance")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Episodes per model: {args.episodes}")

    summaries = []
    for model_name in models:
        summary = evaluate_model(model_name, n_episodes=args.episodes, gui=args.gui)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        print("\n[WARN] No models were evaluated.")
        return

    # Export combined summary CSV
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "test_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\n[INFO] Combined test summary -> {summary_path}")

    # Print final table
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    header = f"{'Model':<8} {'Obs':<8} {'#Sens':>5} {'Success':>8} {'Collis':>8} {'Intrus':>8} {'Reward':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['model']:<8} {s['obs_choice']:<8} {s['num_sensors']:>5} "
            f"{s['success_rate']*100:>7.1f}% {s['collision_rate']*100:>7.1f}% "
            f"{s['intrusion_rate']*100:>7.1f}% {s['mean_reward']:>+9.2f}"
        )


if __name__ == "__main__":
    main()
