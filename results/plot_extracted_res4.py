import json
import matplotlib.pyplot as plt
import os


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
    "shift_true_filtered",
    "shift_target_filtered",
    "rel_true_filtered",
    "rel_target_filtered",
]


def draw_plot(data, mult, tit, fn, task="count", metrics=METR1, sys=False):
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

            values = data[key][task][metric]

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

    tit = tit.split('_')
    sys_opt_prt = 'SYSP_OPT' if sys else 'OPT'
    fig.suptitle(f"Steering into false random {task}ing, {sys_opt_prt if len(tit) > 2 else ''}, vect={tit[-1]}, test={tit[-2]}")
    plt.savefig(fn)


def process_dir_res(dir, sys=False, MX=[METR1, METR2, METR3]):
    files = [f for f in os.listdir(dir) if f.endswith('.json')]
    print(files)
    for f in files:
        print(f)
        name = f.replace('.json', '').split('_')
        print(name)
        src = os.path.join(dir, f)
        for i, met in enumerate(MX):
            for t in ["count"]:
                for mult in [(-2, 2)]:
                    title = '_'.join(name[3:len(name)-1])
                    fn = f'{title}_met{i}.png'
                    print(title, fn)
                    draw_plot(src, mult, title, os.path.join(dir, fn), t, met, sys)


if __name__ == '__main__':
    process_dir_res('results/V2/just', sys=True)
    process_dir_res('results/V1/just', MX=[METR1])
