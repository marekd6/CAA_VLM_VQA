import json
import matplotlib.pyplot as plt


metrics = [
    "shift_true_norm",
    "shift_target_norm",
    "shift_true",
    "shift_target",
]


def draw_plot(data, mult, tit, fn):
    with open(data) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(
        2, 2,
        sharex=True,
        sharey='row',
        layout='constrained'
    )

    for ax, metric in zip(axes.flat, metrics):
        for s in mult:
            key = str(float(s))

            values = data[key]["count"][metric]

            x = sorted(map(int, values.keys()))
            y = [values[str(v)] for v in x]

            ax.plot(x, y, marker="o", label=key)

        ax.set_title(metric)
        ax.grid(alpha=0.3)

    # fig.legend(mult, loc='outside right upper', title="multiplier")
    axes[1, 1].legend(mult, title="multiplier")
    fig.supxlabel('layer')

    fig.suptitle(tit)
    # fig.show()
    fig.savefig(fn)


if __name__ == '__main__':
    draw_plot('gelungene_runs/caa_count_train_opt_100_100_combined.json', (-2, 2), 
              "Steering into false random counting, vect=100, test=100, closed choice opt", 'cnt_100_100_opt.png')
    draw_plot('gelungene_runs/caa_count_train_100_combined.json', (-2, 2),
              "Steering into false random counting, vect=100, test=100", 'cnt_100.png')
