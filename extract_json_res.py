import json
from pathlib import Path
from collections import defaultdict


INPUT_DIR = Path("/data/5drwal")
MERGED_SUB_DIR = 'merged'
OUTPUT_SUFFIX = "_combined.json"


def merge_into(target, source, parameter):
    """
    Recursively merge source into target.

    Leaf values become:
        {parameter: value}
    """
    for key, value in source.items():
        if isinstance(value, dict):
            target.setdefault(key, {})
            merge_into(target[key], value, parameter)
        else:
            target.setdefault(key, {})
            target[key][parameter] = value


def group_files(files):
    """
    Group files by all filename components except the last one.

    Example:
        caa_count_train_100_10.json
        -> group='caa_count_train_100'
           parameter='10'
    """
    groups = defaultdict(list)

    for path in files:
        parts = path.stem.split("_")

        if len(parts) < 2:
            continue

        parameter = parts[-1]
        group_name = "_".join(parts[:-1])

        groups[group_name].append((parameter, path))

    return groups


def main():
    files = list(INPUT_DIR.glob("*.json"))
    groups = group_files(files)

    for group_name, entries in groups.items():

        # Sort numerically when possible
        entries.sort(key=lambda x: float(x[0]))

        merged = {}

        for parameter, path in entries:
            print(f"Reading {path.name}")

            with open(path, "r") as f:
                data = json.load(f)

            merge_into(merged, data, parameter)

        output_file = INPUT_DIR / MERGED_SUB_DIR / f"{group_name}{OUTPUT_SUFFIX}"

        with open(output_file, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
