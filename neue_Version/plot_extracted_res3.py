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
    sharex=True,
    sharey='row',
    # layout='constrained'
    layout='tight'
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

# axes[1, 1].legend(title="multiplier")
# axes[1, 0].set_xlabel('layer')
# axes[1, 1].set_xlabel('layer')

plt.legend(title="multiplier")
fig.supxlabel('layer')

fig.suptitle("Steering into false random counting, vect=100, test=100, closed choice opt")
plt.show()
# plt.savefig('cnt_100_100_opt.png')
