import json
import matplotlib.pyplot as plt
import os
# import seaborn as sns
# import pandas as pd


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

METR4 = [
    "avg_prob_true",
    "avg_prob_target",
    "avg_prob_true_norm",
    "avg_prob_target_norm",
]

METR_REL_TRG = [
    "rel_target",
]

METR_REL_TRG4 = [
    "rel_target",
    "rel_target",
    "avg_prob_target",
    "avg_prob_target",
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
    ax = axes[-1, -1]
    ax.legend(mult, title="multiplier")
    fig.supxlabel('layer')

    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    tit = tit.split('_')
    sys_opt_prt = 'SYSP_OPT' if sys else 'OPT'
    fig.suptitle(f"Steering into false random {task}ing, {sys_opt_prt if len(tit) > 2 else ''}, vect={tit[-1]}, test={tit[-2]}")
    # plt.savefig(fn)
    plt.show()


# def draw_plot1(data, mult, tit, fn, task="count", metric4=METR_REL_TRG, sys=False):
#     with open(data) as f:
#         data = json.load(f)

#     df = pd.DataFrame(data)
    
#     sns.catplot(
#         data=df,
#         x=''
#     )

#     for ax, metric in zip(axes.flat, metrics):
#         for s in mult:
#             key = str(float(s))

#             values = data[key][task][metric]

#             x = sorted(map(int, values.keys()))
#             y = [values[str(v)] for v in x]

#             ax.plot(x, y, marker="o", label=key)

#         ax.set_title(metric)
#         ax.grid(alpha=0.3)

#     # fig.legend(mult, loc='outside right upper', title="multiplier")
#     ax = axes[-1, -1]
#     ax.legend(mult, title="multiplier")
#     fig.supxlabel('layer')

#     yabs_max = abs(max(ax.get_ylim(), key=abs))
#     ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

#     tit = tit.split('_')
#     sys_opt_prt = 'SYSP_OPT' if sys else 'OPT'
#     fig.suptitle(f"Steering into false random {task}ing, {sys_opt_prt if len(tit) > 2 else ''}, vect={tit[-1]}, test={tit[-2]}")
#     # plt.savefig(fn)
#     plt.show()


def plot_metric(src, category, metric, series_filter=None, t=''):
    with open(src) as f:
        data = json.load(f)
    plt.figure(figsize=(8, 5))

    items = data.items()

    if series_filter is not None:
        items = (
            (k, v)
            for k, v in items
            if k in series_filter # multiplier
        )

    yp = []
    yn = []
    for series_name, series_data in sorted( # mult, data
            items,
            key=lambda x: float(x[0])): # data under mult

        values = series_data[category][metric]

        x = sorted(map(int, values.keys())) # layer
        y = [values[str(v)]*100 for v in x] # prob
        # if len(yp) == 0:
        #     yp = y.copy()
        # else:
        #     yn = y.copy()
        if len(yn) == 0:
            yn = y.copy()
        else:
            yp = y.copy()

        # print('x', x)
        # print('y', y)

        plt.plot(
            x,
            y,
            marker="o",
            label=series_name
        )

    # print(yp)
    # print(yn)
    print(max(yp), min(yn))
    print(yp.index(max(yp)), yn.index(min(yn)))
    ydiff = [p - n for p, n in zip(yp, yn)]
    # print(ydiff)
    print(max(ydiff), 'at', ydiff.index(max(ydiff)))
    for i in range(len(ydiff)):
        print(i, ydiff[i])


    ydiff = yp
    # print(len(ydiff[7:30]))
    # print()
    # print(sum(ydiff[7:30]) / len(ydiff[7:30])) # 12,34
    # print(sum(ydiff[0:7]) / len(ydiff[0:7])) # 0,96
    # print(sum(ydiff[30:len(ydiff)]) / len(ydiff[30:len(ydiff)])) # 6,94
    # print((sum(ydiff[0:7]) + sum(ydiff[30:len(ydiff)])) / (len(ydiff[0:7]) + len(ydiff[30:len(ydiff)]))) # 3,14
    
    # print(len(ydiff[5:29])) # 24
    # print(len(ydiff[0:5])) # 5
    # print(len(ydiff[29:34])) # 5
    # print()
    # print(sum(ydiff[5:29]) / len(ydiff[5:29])) # 12,34
    # print(sum(ydiff[0:5]) / len(ydiff[0:5])) # 0,96
    # print(sum(ydiff[29:34]) / len(ydiff[29:34])) # 6,94
    # print((sum(ydiff[0:5]) + sum(ydiff[29:34])) / (len(ydiff[0:5]) + len(ydiff[29:34]))) # 3,14
    
    print(len(ydiff[7:27])) # 24
    print(len(ydiff[0:7])) # 5
    print(len(ydiff[27:34])) # 5
    print('yp')
    print(sum(ydiff[7:27]) / len(ydiff[7:27])) # 12,34
    print(sum(ydiff[0:7]) / len(ydiff[0:7])) # 0,96
    print(sum(ydiff[27:34]) / len(ydiff[27:34])) # 6,94
    print((sum(ydiff[0:7]) + sum(ydiff[27:34])) / (len(ydiff[0:7]) + len(ydiff[27:34]))) # 3,14


    print('yn')
    ydiff = yn
    print(sum(ydiff[7:27]) / len(ydiff[7:27])) # 12,34
    print(sum(ydiff[0:7]) / len(ydiff[0:7])) # 0,96
    print(sum(ydiff[27:34]) / len(ydiff[27:34])) # 6,94
    print((sum(ydiff[0:7]) + sum(ydiff[27:34])) / (len(ydiff[0:7]) + len(ydiff[27:34]))) # 3,14


    metric = 'Δp(answer matching behaviour) %'
    plt.xlabel("Layer")
    plt.ylabel('p(true answ norm) %')
    plt.title(t)
    # plt.title('Per-layer CAA with 10-sample generated vectors')
    plt.legend(title="Multiplier")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    b, tt = plt.ylim()
    # yyy = max(-b, tt)
    # plt.ylim((-yyy, yyy))
    yyy = abs(max(b, tt))
    # plt.ylim((-yyy, yyy))
    # plt.show()
    plt.savefig(os.path.join(os.path.dirname(src), t+'.png'))
    plt.close()


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
                    # draw_plot(src, mult, title, os.path.join(dir, fn), t, met, sys)
                    plot_metric(src, t, 'rel_target', ['-2.0', '2.0'], 'Steering with vector of 100 samples')


def process_dir_res2(dir, sys=False, MX=[METR1, METR2, METR3]):
    files = [f for f in os.listdir(dir) if f.endswith('.json')]
    print(files)
    for f in files:
        name = f.replace('.json', '').split('_')
        print(f, name)
        src = os.path.join(dir, f)
        # for i, met in enumerate(MX):
        for yyy in [9]:
            for t in ["count"]:
                for mult in [(-2, 2)]:
                    # plot_metric(src, t, 'rel_target_norm', ['-2.0', '2.0'], f.replace('.json', '')+'_rel_target_norm') # ten
                    # title = '_'.join(name[3:len(name)-1])
                    # fn = f'{title}_met{i}.png'
                    # print(title, fn)
                    # draw_plot(src, mult, title, os.path.join(dir, fn), t, met, sys)
                    # plot_metric(src, t, 'p_target_norm', ['-2.0', '2.0'], f.replace('.json', '')+'p_target_norm')
                    plot_metric(src, t, 'p_true_norm', ['-2.0', '2.0'], f.replace('.json', '')+'p_true_norm')
                    # break
                    # plot_metric(src, t, 'rel_target', ['-2.0', '2.0'], f.replace('.json', '')+'rel_target')
                    # plot_metric(src, t, 'rel_target_norm', ['-2.0', '2.0'], f.replace('.json', '')+'_rel_target_norm') # ten
                    # # plot_metric(src, t, 'rel_target_filtered', ['-2.0', '2.0'], f.replace('.json', ''))
                    # plot_metric(src, t, 'avg_prob_target_norm', ['-2.0', '2.0'], f.replace('.json', '')+'p_target_norm')
                    # plot_metric(src, t, 'shift_target', ['-2.0', '2.0'], f.replace('.json', '')+'shift_target')
                    # plot_metric(src, t, 'shift_target_norm', ['-2.0', '2.0'], f.replace('.json', '')+'shift_target_norm')


if __name__ == '__main__':
    # process_dir_res('results/V2/just', sys=True)
    # process_dir_res('results/V1/just', MX=[METR1])
    # process_dir_res('results/V2/just', MX=[METR4], sys=True)
    # process_dir_res('results/V2/just', MX=[METR_REL_TRG4], sys=True)
    # process_dir_res('results/V1/just', MX=[METR_REL_TRG], sys=False)

    # process_dir_res2('results/to_plot/paper')
    process_dir_res2('results/to_plot/2')
