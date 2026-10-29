import numpy as np
import matplotlib.pyplot as plt

# Rows: clients 0–4. Columns: Normal, Attack.
counts = np.array([
    [8657, 5808],
    [11025, 104],
    [159, 313],
    [1944, 12770],
    [749, 72755],
])

percent = counts / counts.sum(axis=1, keepdims=True) * 100
clients = np.arange(len(counts))
colors = ["#277DA8", "#E27649"]

fig, ax = plt.subplots(figsize=(9, 6))

for j, label in enumerate(["Normal", "Attack"]):
    ax.barh(
        clients,
        percent[:, j],
        left=percent[:, :j].sum(axis=1),
        height=0.6,
        color=colors[j],
        label=label,
    )

for i in clients:
    # Labels inside sufficiently wide segments.
    for j in range(2):
        if percent[i, j] >= 8:
            center = percent[i, :j].sum() + percent[i, j] / 2
            ax.text(
                center, i, f"{percent[i, j]:.1f}%",
                ha="center", va="center",
                color="white", fontsize=13, fontweight="bold",
            )

    # Callouts for very small segments.
    if percent[i, 0] < 8:
        ax.annotate(
            f"Normal {percent[i, 0]:.1f}%",
            xy=(percent[i, 0] / 2, i),
            xytext=(16, i + 0.38),
            color=colors[0], fontsize=11,
            arrowprops=dict(arrowstyle="-", color=colors[0]),
        )

    if percent[i, 1] < 8:
        ax.annotate(
            f"Attack {percent[i, 1]:.1f}%",
            xy=(100 - percent[i, 1] / 2, i),
            xytext=(72, i + 0.38),
            color=colors[1], fontsize=11,
            arrowprops=dict(arrowstyle="-", color=colors[1]),
        )

ax.set(
    yticks=clients,
    yticklabels=[f"Client {i}" for i in clients],
    xlim=(0, 100),
    xticks=np.arange(0, 101, 20),
)
ax.invert_yaxis()
ax.set_title("Class balance — within each client", fontsize=18)
ax.set_xlabel("Share of each client (%)", fontsize=14)
ax.tick_params(labelsize=12)
ax.grid(axis="x", alpha=0.18)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("client_class_balance.png", dpi=300, bbox_inches="tight")
fig.savefig("client_class_balance.svg", bbox_inches="tight")
plt.show()