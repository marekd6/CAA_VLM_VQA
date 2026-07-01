import json
import matplotlib.pyplot as plt


def plot_metric(data, category, metric, series_filter=None):
    plt.figure(figsize=(8, 5))

    items = data.items()

    if series_filter is not None:
        items = (
            (k, v)
            for k, v in items
            if k in series_filter
        )

    for series_name, series_data in sorted(
            items,
            key=lambda x: float(x[0])):

        values = series_data[category][metric]

        x = sorted(map(int, values.keys()))
        y = [values[str(v)] for v in x]

        plt.plot(
            x,
            y,
            marker="o",
            label=series_name
        )

    plt.xlabel("Parameter")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


with open("gelungene_runs/caa_count_train_opt_100_100_combined.json") as f:
    data = json.load(f)

plot_metric(data, "count", "shift_true", series_filter=["1.5", "-1.5"])
plot_metric(data, "count", "shift_true_norm", series_filter=["1.5", "-1.5"])
plot_metric(data, "count", "shift_true_filtered", series_filter=["1.5", "-1.5"])

plot_metric(data, "count", "shift_target", series_filter=["1.5", "-1.5"])
plot_metric(data, "count", "shift_target_norm", series_filter=["1.5", "-1.5"])
plot_metric(data, "count", "shift_target_filtered", series_filter=["1.5", "-1.5"])
