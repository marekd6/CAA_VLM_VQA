import json
import matplotlib.pyplot as plt

with open("gelungene_runs/caa_count_train_opt_100_100_combined.json") as f:
        data = json.load(f)

metrics = [
    "shift_true_norm",
    "shift_target_norm",
    "shift_true",
    "shift_target",
]

pair = (-2, 2)

fig, axes = plt.subplots(
    2, 2,
    # figsize=(10, 8),
    sharex=True,
    sharey='row',
    # constrained_layout=True,
)

for ax, metric in zip(axes.flat, metrics):

    for s in pair:
        key = str(float(s))

        values = data[key]["count"][metric]

        x = sorted(map(int, values.keys()))
        y = [values[str(v)] for v in x]

        ax.plot(x, y, marker="o", label=key)

    ax.set_title(metric)
    ax.grid(alpha=0.3)
    ax.margins(x=0)

axes[0, 0].legend(title="Series")

# fig.supxlabel("Filename parameter")
# fig.supylabel("Value")
plt.subplots_adjust(left=0.1, right=0.4, top=0.4, bottom=0.1)
fig.suptitle("Pair: -2 vs 2")
fig.tight_layout()
plt.show()
