import json
import matplotlib.pyplot as plt


METR1 = [
    "shift_true_norm",
    "shift_target_norm",
    "shift_true",
    "shift_target",
]

METR2 = [
    "rel_true_norm",
    "rel_target_norm",
    "rel_true",
    "rel_target",
]

METR3 = [
    "rel_true_norm2",
    "rel_target_norm2",
    "rel_true",
    "rel_target",
]


def draw_plot(data, mult, tit, fn, metrics=METR1):
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
    ax = axes[1, 1]
    ax.legend(mult, title="multiplier")
    fig.supxlabel('layer')

    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    fig.suptitle(tit)
    # plt.show()
    plt.savefig(fn)


if __name__ == '__main__':
    # draw_plot('gelungene_runs/caa_count_train_opt_100_100_combined.json', (-2, 2), 
    #           "Steering into false random counting, vect=100, test=100, closed choice opt", 'cnt_100_100_opt.png')
    # draw_plot('gelungene_runs/caa_count_train_100_combined.json', (-2, 2),
    #           "Steering into false random counting, vect=100, test=100", 'cnt_100.png')
    draw_plot('caa_count_train_opt_10_100_combined2.json', (-2, 2),
              "Steering into false random counting, vect=100, test=10", 'cnt_10_100_v2.png')
    draw_plot('caa_count_train_opt_10_100_combined2.json', (-2, 2),
              "Steering into false random counting, vect=100, test=10", 'cnt_10_100_v2_2.png', METR2)
    draw_plot('caa_count_train_opt_10_100_combined2.json', (-2, 2),
              "Steering into false random counting, vect=100, test=10", 'cnt_10_100_v2_3.png', METR3)
