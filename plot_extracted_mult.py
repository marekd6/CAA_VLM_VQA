import json
import matplotlib.pyplot as plt

# Path to your JSON file
json_file = "results/to_plot/paper/sysopt_100_100.json"
json_file = "results/to_plot/2/sysopt_100_100_trtrg.json"

# Keys specifying the nested value to extract
key1 = "count"
key2 = "avg_prob_target_norm"
key2 = "p_target_norm"
# key2 = "p_true_norm"
key3 = "14"

# Load JSON
with open(json_file, "r") as f:
    data = json.load(f)

x = []
y = []

# Sort first-level keys numerically
for first_key in sorted(data.keys(), key=float):
    try:
        value = data[first_key][key1][key2][key3] * 100
        x.append(float(first_key))
        y.append(value)
    except KeyError:
        print(f"Skipping {first_key}: missing key")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker="o")
plt.grid(True, alpha=0.3)
plt.xlabel("Steering vector multiplier")
plt.ylabel('p(answer matching behaviour) %')
plt.title('CAA with 100-sample generated vector on layer 14')
plt.grid(True)

plt.tight_layout()
plt.show()
