import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.18.2025_22.06.14\PPO\PPO_autorouting-mas-aviary-v0_f6699_00000_0_2025-05-18_22-06-27"
# trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.19.2025_08.54.14\PPO\PPO_autorouting-mas-aviary-v0_7d1a9_00000_0_2025-05-19_08-54-27"
# trial_dir=r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.28.2025_16.19.31\PPO\PPO_autorouting-mas-aviary-v0_2e39e_00000_0_2025-05-28_16-19-42"
trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.28.2025_16.45.32\PPO\PPO_autorouting-mas-aviary-v0_d0e65_00000_0_2025-05-28_16-45-44"
# result_path = os.path.join(trial_dir, "result.json")
# with open(result_path, "r") as f:
#     result = json.load(f)
# print(json.dumps(result, indent=2))


# Use raw string (prefix with r) to handle backslashes in Windows path

csv_path = os.path.join(trial_dir, "progress.csv")

# Load the CSV
df = pd.read_csv(csv_path)

# Optional: smooth reward
df["reward_smooth"] = df["episode_reward_mean"].rolling(window=10).mean()

def plot_metric(x, y, title, ylabel, labels=None):
    plt.figure(figsize=(10, 6))
    if isinstance(y, list):
        for yi, label in zip(y, labels):
            plt.plot(df[x], df[yi], label=label)
    else:
        plt.plot(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.title(title)
    if labels:
        plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# 1. Overall Reward Stats
plot_metric("training_iteration", 
            ["episode_reward_mean", "reward_smooth", "episode_reward_min", "episode_reward_max"], 
            "Episode Reward (Mean/Min/Max)", 
            "Reward", 
            ["Mean", "Smoothed", "Min", "Max"])

# 2. Episode Length
plot_metric("training_iteration", "episode_len_mean", "Mean Episode Length", "Episode Length")

# 3. Per-policy Rewards
# plot_metric("training_iteration", 
#             ["policy_reward_mean/pol0", "policy_reward_mean/pol1"], 
#             "Per-Policy Mean Rewards", 
#             "Reward", 
#             ["Policy 0", "Policy 1"])

# 4. Policy Losses (pol0)
plot_metric("training_iteration", 
            ["info/learner/pol0/learner_stats/total_loss", 
             "info/learner/pol0/learner_stats/policy_loss", 
             "info/learner/pol0/learner_stats/vf_loss"], 
            "Policy 0 Losses", 
            "Loss", 
            ["Total", "Policy", "Value Function"])

# 5. Policy Losses (pol1)
plot_metric("training_iteration", 
            ["info/learner/pol1/learner_stats/total_loss", 
             "info/learner/pol1/learner_stats/policy_loss", 
             "info/learner/pol1/learner_stats/vf_loss"], 
            "Policy 1 Losses", 
            "Loss", 
            ["Total", "Policy", "Value Function"])

# 6. KL Divergence
plot_metric("training_iteration", 
            ["info/learner/pol0/learner_stats/kl", 
             "info/learner/pol1/learner_stats/kl"], 
            "KL Divergence", 
            "KL", 
            ["Pol0", "Pol1"])

# 7. Entropy
plot_metric("training_iteration", 
            ["info/learner/pol0/learner_stats/entropy", 
             "info/learner/pol1/learner_stats/entropy"], 
            "Policy Entropy", 
            "Entropy", 
            ["Pol0", "Pol1"])

# 8. Value Function Explained Variance
plot_metric("training_iteration", 
            ["info/learner/pol0/learner_stats/vf_explained_var", 
             "info/learner/pol1/learner_stats/vf_explained_var"], 
            "Value Function Explained Variance", 
            "Explained Variance", 
            ["Pol0", "Pol1"])

# 9. Sampler and Environment Wait Times
plot_metric("training_iteration", 
            ["sampler_perf/mean_inference_ms", 
             "sampler_perf/mean_env_wait_ms", 
             "sampler_perf/mean_action_processing_ms"], 
            "Sampler Performance", 
            "Time (ms)", 
            ["Inference", "Env Wait", "Action Processing"])

