import json
import matplotlib.pyplot as plt


def plot_metric_pairs(data, category, metric, fn):
    pairs = [(-2, 2), (-1, 1)]

    fig, axes = plt.subplots(
        # 1, len(pairs),
        len(pairs), 1,
        figsize=(5 * len(pairs), 4),
        sharey=True
    )

    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):

        for s in pair:
            key = str(float(s))      # "-2.0", "1.0", ...

            if key not in data:
                continue

            values = data[key][category][metric]

            x = sorted(map(int, values.keys()))
            y = [values[str(v)] for v in x]

            ax.plot(
                x,
                y,
                marker="o",
                label=key
            )

        ax.set_title(" vs ".join(map(str, pair)))
        ax.set_xlabel("Filename parameter")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(metric)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{fn}_{category}_{metric}.png')
    plt.close()


if __name__ == '__main__':

    with open("gelungene_runs/caa_count_train_opt_100_100_combined.json") as f:
        data = json.load(f)

    plot_metric_pairs(data, 'count', 'shift_true', 'caa_count_train_opt_100_100')
    plot_metric_pairs(data, 'count', 'shift_target', 'caa_count_train_opt_100_100')
    plot_metric_pairs(data, 'count', 'shift_true_norm', 'caa_count_train_opt_100_100')
    plot_metric_pairs(data, 'count', 'shift_target_norm', 'caa_count_train_opt_100_100')
    