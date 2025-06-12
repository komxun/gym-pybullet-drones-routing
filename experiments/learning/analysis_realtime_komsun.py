import pandas as pd
import matplotlib.pyplot as plt
import os
import time

realtime_plot = False
# trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.30.2025_14.04.33\PPO\PPO_autorouting-mas-aviary-v0_a8ea9_00000_0_2025-05-30_14-04-46"
# trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-06.02.2025_09.00.59\SAC\SAC_autorouting-mas-aviary-v0_bf55a_00000_0_2025-06-02_09-01-11"
trial_dir = r"C:\Users\s400263\Documents\gym-pybullet-drones-routing\experiments\learning\results\save-autorouting-mas-aviary-v0-2-cc-kin-autorouting-05.28.2025_16.45.32\PPO\PPO_autorouting-mas-aviary-v0_d0e65_00000_0_2025-05-28_16-45-44"
csv_path = os.path.join(trial_dir, "progress.csv")

# Initialize interactive mode and figure once
plt.ion()
fig, axs = plt.subplots(4, 2, figsize=(15, 15))
axs = axs.flatten()

def safe_plot(ax, df, x, ys, labels, title, ylabel):
    ax.clear()
    for y, label in zip(ys, labels):
        if y in df.columns:
            ax.plot(df[x], df[y], label=label)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def plot_all_metrics(df):
    df["reward_smooth"] = df["episode_reward_mean"].rolling(window=10).mean()

    safe_plot(axs[0], df, "training_iteration",
              ["episode_reward_mean", "reward_smooth", "episode_reward_min", "episode_reward_max"],
              ["Mean", "Smoothed", "Min", "Max"],
              "Episode Reward (Mean/Min/Max)", "Reward")

    safe_plot(axs[1], df, "training_iteration", ["episode_len_mean"], ["Mean"], "Mean Episode Length", "Length")

    safe_plot(axs[2], df, "training_iteration",
              ["info/learner/pol0/learner_stats/total_loss",
               "info/learner/pol0/learner_stats/policy_loss",
               "info/learner/pol0/learner_stats/vf_loss"],
              ["Total", "Policy", "Value Function"], "Policy 0 Losses", "Loss")

    safe_plot(axs[3], df, "training_iteration",
              ["info/learner/pol1/learner_stats/total_loss",
               "info/learner/pol1/learner_stats/policy_loss",
               "info/learner/pol1/learner_stats/vf_loss"],
              ["Total", "Policy", "Value Function"], "Policy 1 Losses", "Loss")

    safe_plot(axs[4], df, "training_iteration",
              ["info/learner/pol0/learner_stats/kl",
               "info/learner/pol1/learner_stats/kl"],
              ["Pol0", "Pol1"], "KL Divergence", "KL")

    safe_plot(axs[5], df, "training_iteration",
              ["info/learner/pol0/learner_stats/entropy",
               "info/learner/pol1/learner_stats/entropy"],
              ["Pol0", "Pol1"], "Policy Entropy", "Entropy")

    safe_plot(axs[6], df, "training_iteration",
              ["info/learner/pol0/learner_stats/vf_explained_var",
               "info/learner/pol1/learner_stats/vf_explained_var"],
              ["Pol0", "Pol1"], "VF Explained Variance", "Explained Variance")

    safe_plot(axs[7], df, "training_iteration",
              ["sampler_perf/mean_inference_ms",
               "sampler_perf/mean_env_wait_ms",
               "sampler_perf/mean_action_processing_ms"],
              ["Inference", "Env Wait", "Action Processing"], "Sampler Perf", "Time (ms)")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()




# Main loop to refresh every second

while True:
    try:
        df = pd.read_csv(csv_path)
        plot_all_metrics(df)
    except Exception as e:
        print("Error reading or plotting:", e)
    time.sleep(5)


