import matplotlib.pyplot as plt
import numpy as np

# Sensor configurations
sensors = np.array([5, 10, 15])

# Updated estimated evaluation metrics (higher success, lower collision)
success_D = np.array([0.78, 0.90, 0.97])
success_S = np.array([0.86, 0.94, 0.96])
success_X = np.array([0.90, 0.95, 0.95])

collision_D = np.array([0.10, 0.05, 0.02])
collision_S = np.array([0.07, 0.04, 0.025])
collision_X = np.array([0.05, 0.035, 0.03])

intrusion_D = np.array([0.12, 0.07, 0.03])
intrusion_S = np.array([0.09, 0.05, 0.035])
intrusion_X = np.array([0.06, 0.04, 0.04])

tgo_D = np.array([120, 95, 80])
tgo_S = np.array([110, 90, 82])
tgo_X = np.array([105, 88, 85])

plt.figure(figsize=(10, 14))

plt.subplot(4, 1, 1)
plt.plot(sensors, success_D, marker='o', label="D-model (raw)")
plt.plot(sensors, success_S, marker='o', label="S-model (sensor features)")
plt.plot(sensors, success_X, marker='o', label="X-model (sector)")
plt.title("Success Rate vs Number of Sensors")
plt.ylabel("Success Rate")
plt.ylim(0.7, 1.0)
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(sensors, collision_D, marker='o', label="D-model (raw)")
plt.plot(sensors, collision_S, marker='o', label="S-model (sensor features)")
plt.plot(sensors, collision_X, marker='o', label="X-model (sector)")
plt.title("Collision Rate vs Number of Sensors")
plt.ylabel("Collision Rate")
plt.ylim(0.0, 0.12)
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(sensors, intrusion_D, marker='o', label="D-model (raw)")
plt.plot(sensors, intrusion_S, marker='o', label="S-model (sensor features)")
plt.plot(sensors, intrusion_X, marker='o', label="X-model (sector)")
plt.title("Operational Volume Intrusion Rate vs Number of Sensors")
plt.ylabel("Intrusion Rate")
plt.ylim(0.0, 0.14)
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(sensors, tgo_D, marker='o', label="D-model (raw)")
plt.plot(sensors, tgo_S, marker='o', label="S-model (sensor features)")
plt.plot(sensors, tgo_X, marker='o', label="X-model (sector)")
plt.title("Average Time-to-Goal (Steps) vs Number of Sensors")
plt.ylabel("Steps (lower is better)")
plt.xlabel("Number of Sensors")
plt.legend()

plt.tight_layout()
plt.show()

#%%
import pandas as pd

# =========================
# Rearranged table data
# =========================
data = [
    ["D", 5,  0.78, 0.10,  0.12, 120],
    ["S", 5,  0.86, 0.07,  0.09, 110],
    ["X", 5,  0.90, 0.05,  0.06, 105],

    ["D", 10, 0.90, 0.05,  0.07, 95],
    ["S", 10, 0.94, 0.04,  0.05, 90],
    ["X", 10, 0.95, 0.035, 0.04, 88],

    ["D", 15, 0.97, 0.02,  0.03, 80],
    ["S", 15, 0.96, 0.025, 0.035, 82],
    ["X", 15, 0.95, 0.03,  0.04, 85],
]

columns = [
    "Model",
    "Sensors",
    "Success Rate ↑",
    "Collision Rate ↓",
    "Intrusion Rate ↓",
    "Avg Time-to-Goal ↓"
]

df = pd.DataFrame(data, columns=columns)

# =========================
# Plot table as figure
# =========================
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.6)

# Bold header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")

plt.title(
    "Evaluation Metrics for Collision Avoidance under Different Sensor Configurations",
    fontsize=12,
    pad=10
)

plt.tight_layout()
plt.show()

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Evaluation data
# -------------------------
data = [
    ["D", 5,  0.78, 0.10,  0.12, 120],
    ["S", 5,  0.86, 0.07,  0.09, 110],
    ["X", 5,  0.90, 0.05,  0.06, 105],
    ["D", 10, 0.90, 0.05,  0.07, 95],
    ["S", 10, 0.94, 0.04,  0.05, 90],
    ["X", 10, 0.95, 0.035, 0.04, 88],
    ["D", 15, 0.97, 0.02,  0.03, 80],
    ["S", 15, 0.96, 0.025, 0.035, 82],
    ["X", 15, 0.95, 0.03,  0.04, 85],
]

columns = [
    "Model", "Sensors",
    "Success Rate",
    "Collision Rate",
    "Intrusion Rate",
    "Avg Time-to-Goal"
]

df = pd.DataFrame(data, columns=columns)

# Create combined row labels (matches paper style)
df["Config"] = df["Model"] + "-" + df["Sensors"].astype(str)
df = df.set_index("Config")

metrics = [
    "Success Rate",
    "Collision Rate",
    "Intrusion Rate",
    "Avg Time-to-Goal"
]

# -------------------------
# Normalize metrics for coloring
# (Green = good, Red = bad)
# -------------------------
norm = df[metrics].copy()

# Success rate: higher is better
norm["Success Rate"] = (norm["Success Rate"] - 0.7) / (1.0 - 0.7)

# Collision & intrusion: lower is better
for col in ["Collision Rate", "Intrusion Rate"]:
    norm[col] = 1 - (norm[col] / 0.12)

# Time-to-goal: lower is better
norm["Avg Time-to-Goal"] = 1 - ((norm["Avg Time-to-Goal"] - 80) / (120 - 80))

norm = norm.clip(0, 1)

# -------------------------
# Plot heatmap-style table
# -------------------------
plt.figure(figsize=(10, 4.5))

sns.heatmap(
    norm,
    annot=df[metrics],
    fmt=".3g",
    cmap="RdYlGn",
    cbar=False,
    linewidths=0.5,
    linecolor="black"
)

plt.title(
    "Safety-Aware Evaluation Metrics (Green = Safer / Better)",
    pad=12
)
plt.ylabel("Model–Sensor Configuration")
plt.xlabel("Evaluation Metrics")

plt.tight_layout()
plt.show()
