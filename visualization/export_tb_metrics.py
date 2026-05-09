import argparse
import csv
import json
import os
import shutil
from collections import defaultdict


def load_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to read event files. Install it in the same environment:\n"
            "  pip install tensorboard"
        ) from exc
    return event_accumulator


def has_event_file(path):
    if not os.path.isdir(path):
        return False
    return any(name.startswith("events.out.tfevents") for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)))


def find_event_dir(tb_dir):
    if not has_event_file(tb_dir):
        raise SystemExit(f"No TensorBoard event files found directly under {tb_dir}")
    return tb_dir


def read_scalars(run_dir):
    event_accumulator = load_event_accumulator()
    accumulator = event_accumulator.EventAccumulator(run_dir)
    accumulator.Reload()

    scalars = defaultdict(dict)
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            scalars[event.step][tag] = event.value
    return dict(sorted(scalars.items())), accumulator.Tags().get("scalars", [])


def write_metrics_csv(metrics_by_step, tags, csv_path):
    fieldnames = ["step", *tags]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step, values in metrics_by_step.items():
            row = {"step": step}
            row.update(values)
            writer.writerow(row)


def write_metrics_jsonl(metrics_by_step, jsonl_path):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for step, values in metrics_by_step.items():
            row = {"step": step, **values}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def copy_hparams(run_dir, output_dir):
    output_dir = os.path.abspath(output_dir)
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = [name for name in dirs if os.path.abspath(os.path.join(root, name)) != output_dir]
        if "hparams.yml" in files:
            src = os.path.abspath(os.path.join(root, "hparams.yml"))
            dst = os.path.join(output_dir, "hparams.yml")
            if src != dst:
                shutil.copy2(src, dst)
            return


def plot_metrics(metrics_by_step, tags, output_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to write metrics.png. Install it in the same environment:\n"
            "  pip install matplotlib"
        ) from exc

    preferred_tags = [tag for tag in ("loss", "eval_loss", "train/loss", "eval/loss") if tag in tags]
    plot_tags = preferred_tags or tags

    if not plot_tags:
        raise SystemExit("No scalar tags found, cannot write metrics.png")

    plt.figure(figsize=(10, 6))
    for tag in plot_tags:
        points = [(step, values[tag]) for step, values in metrics_by_step.items() if tag in values]
        if not points:
            continue
        steps, values = zip(*points)
        plt.plot(steps, values, label=tag, linewidth=1.5)

    plt.xlabel("global step")
    plt.ylabel("value")
    plt.title("TensorBoard Scalars")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def export_run(run_dir, output_dir):
    output_dir = output_dir or os.path.join(run_dir, "vis")

    os.makedirs(output_dir, exist_ok=True)

    metrics_by_step, tags = read_scalars(run_dir)
    write_metrics_csv(metrics_by_step, tags, os.path.join(output_dir, "metrics.csv"))
    write_metrics_jsonl(metrics_by_step, os.path.join(output_dir, "metrics.jsonl"))
    copy_hparams(run_dir, output_dir)
    plot_path = plot_metrics(metrics_by_step, tags, os.path.join(output_dir, "metrics.png"))

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "num_steps": len(metrics_by_step),
        "tags": tags,
        "plot_path": plot_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Export scalar metrics from TensorBoard event files.")
    parser.add_argument("--tb-dir", required=True, help="TensorBoard run directory that contains one event file set")
    parser.add_argument("--output-dir", default=None, help="Directory for exported files. Default: <tb-dir>/vis")
    args = parser.parse_args()

    run_dir = find_event_dir(args.tb_dir)
    result = export_run(run_dir, args.output_dir)
    print(
        f"exported {result['run_dir']} -> {result['output_dir']} "
        f"({result['num_steps']} steps, tags: {', '.join(result['tags']) or 'none'}, "
        f"plot: {result['plot_path']})"
    )


if __name__ == "__main__":
    main()
