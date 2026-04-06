from __future__ import annotations

import argparse
import re
from pathlib import Path


PROVIDED_METRICS = {
    "test_accuracy": 0.8678,
    "accuracy": 0.8678,
    "total_support": 1763,
    "classes": [
        {"label": "false", "precision": 0.8421, "recall": 0.8016, "f1": 0.8213, "support": 251},
        {"label": "non-rumor", "precision": 0.9015, "recall": 0.9457, "f1": 0.9231, "support": 920},
        {"label": "true", "precision": 0.8364, "recall": 0.8179, "f1": 0.8270, "support": 335},
        {"label": "unverified", "precision": 0.8296, "recall": 0.7817, "f1": 0.8049, "support": 257},
    ],
    "macro_avg": {"precision": 0.8524, "recall": 0.8367, "f1": 0.8441, "support": 1763},
    "weighted_avg": {"precision": 0.8659, "recall": 0.8678, "f1": 0.8665, "support": 1763},
}


def parse_report(report_path: Path) -> dict:
    text = report_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    test_accuracy = None
    class_metrics: list[dict] = []
    macro_avg = None
    weighted_avg = None
    accuracy = None
    total_support = None

    test_acc_pattern = re.compile(r"^\s*Test Accuracy:\s*([0-9]*\.?[0-9]+)\s*$")
    class_pattern = re.compile(
        r"^\s*([A-Za-z0-9_-]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$"
    )
    accuracy_pattern = re.compile(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$")
    avg_pattern = re.compile(
        r"^\s*(macro avg|weighted avg)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$"
    )

    for line in lines:
        m = test_acc_pattern.match(line)
        if m:
            test_accuracy = float(m.group(1))
            continue

        m = class_pattern.match(line)
        if m and m.group(1) not in {"accuracy"}:
            label = m.group(1)
            class_metrics.append(
                {
                    "label": label,
                    "precision": float(m.group(2)),
                    "recall": float(m.group(3)),
                    "f1": float(m.group(4)),
                    "support": int(m.group(5)),
                }
            )
            continue

        m = accuracy_pattern.match(line)
        if m:
            accuracy = float(m.group(1))
            total_support = int(m.group(2))
            continue

        m = avg_pattern.match(line)
        if m:
            row = {
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "support": int(m.group(5)),
            }
            if m.group(1) == "macro avg":
                macro_avg = row
            else:
                weighted_avg = row

    if not class_metrics:
        raise ValueError(f"No per-class rows could be parsed from {report_path}.")

    if accuracy is None:
        raise ValueError("Missing overall accuracy row in report.")

    if test_accuracy is None:
        test_accuracy = accuracy

    if macro_avg is None or weighted_avg is None:
        raise ValueError("Missing macro avg or weighted avg rows in report.")

    return {
        "test_accuracy": test_accuracy,
        "accuracy": accuracy,
        "total_support": total_support,
        "classes": class_metrics,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
    }


def provided_metrics() -> dict:
    return {
        "test_accuracy": PROVIDED_METRICS["test_accuracy"],
        "accuracy": PROVIDED_METRICS["accuracy"],
        "total_support": PROVIDED_METRICS["total_support"],
        "classes": [dict(item) for item in PROVIDED_METRICS["classes"]],
        "macro_avg": dict(PROVIDED_METRICS["macro_avg"]),
        "weighted_avg": dict(PROVIDED_METRICS["weighted_avg"]),
    }


def _setup_plotting(show: bool):
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install with: pip install matplotlib"
        ) from exc
    return plt, np


def _plot_class_bars(parsed: dict, output_dir: Path, show: bool) -> Path:
    plt, np = _setup_plotting(show)

    labels = [item["label"] for item in parsed["classes"]]
    precision = [item["precision"] for item in parsed["classes"]]
    recall = [item["recall"] for item in parsed["classes"]]
    f1 = [item["f1"] for item in parsed["classes"]]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    ax.bar(x - width, precision, width=width, color="#1f77b4", label="Precision")
    ax.bar(x, recall, width=width, color="#2ca02c", label="Recall")
    ax.bar(x + width, f1, width=width, color="#ff7f0e", label="F1-Score")

    ax.set_title("Per-Class Metric Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "class_metrics_bar.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def _plot_support(parsed: dict, output_dir: Path, show: bool) -> Path:
    plt, _ = _setup_plotting(show)

    labels = [item["label"] for item in parsed["classes"]]
    support = [item["support"] for item in parsed["classes"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(labels, support, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_title("Class Support Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Support")
    ax.grid(axis="y", alpha=0.2)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 8, f"{int(height)}", ha="center", va="bottom")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "class_support_bar.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def _plot_summary(parsed: dict, output_dir: Path, show: bool) -> Path:
    plt, _ = _setup_plotting(show)

    names = [
        "Test Acc",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Macro Recall",
        "Weighted Recall",
    ]
    values = [
        parsed["test_accuracy"],
        parsed["accuracy"],
        parsed["macro_avg"]["f1"],
        parsed["weighted_avg"]["f1"],
        parsed["macro_avg"]["recall"],
        parsed["weighted_avg"]["recall"],
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    bars = ax.bar(names, values, color="#3d6dcc")
    ax.set_title("Summary Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center", va="bottom")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "summary_metrics_bar.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def _plot_line_chart(parsed: dict, output_dir: Path, show: bool) -> Path:
    plt, _ = _setup_plotting(show)

    labels = [item["label"] for item in parsed["classes"]]
    precision = [item["precision"] for item in parsed["classes"]]
    recall = [item["recall"] for item in parsed["classes"]]
    f1 = [item["f1"] for item in parsed["classes"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f6f8fb")
    ax.set_facecolor("#ffffff")

    ax.plot(labels, precision, marker="o", linewidth=2.5, color="#1f77b4", label="Precision")
    ax.plot(labels, recall, marker="s", linewidth=2.5, color="#2ca02c", label="Recall")
    ax.plot(labels, f1, marker="^", linewidth=2.5, color="#ff7f0e", label="F1-Score")

    ax.set_title("Per-Class Metrics (Line Plot)", fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "class_metrics_line.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def _plot_heatmap(parsed: dict, output_dir: Path, show: bool) -> Path:
    plt, np = _setup_plotting(show)

    labels = [item["label"] for item in parsed["classes"]]
    matrix = np.array(
        [[item["precision"], item["recall"], item["f1"]] for item in parsed["classes"]],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.patch.set_facecolor("#f4f6fa")
    ax.set_facecolor("#ffffff")

    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.75, vmax=0.96, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Metric Score", rotation=270, labelpad=14)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"], fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontweight="bold")
    ax.set_title("Classification Performance Heatmap", fontsize=15, fontweight="bold")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.4f}", ha="center", va="center", color="#1a1a1a", fontsize=10)

    ax.text(
        0.02,
        -0.14,
        f"Test Accuracy: {parsed['test_accuracy']:.4f}   |   Samples: {parsed['total_support']}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color="#243447",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "classification_heatmap_polished.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=240, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def generate_all_figures(parsed: dict, output_dir: Path, show: bool) -> list[Path]:
    outputs = [
        _plot_class_bars(parsed, output_dir, show),
        _plot_support(parsed, output_dir, show),
        _plot_summary(parsed, output_dir, show),
        _plot_line_chart(parsed, output_dir, show),
        _plot_heatmap(parsed, output_dir, show),
    ]
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate polished classification-report graphs as separate image files."
    )
    parser.add_argument(
        "--source",
        choices=["provided", "report"],
        default="provided",
        help="Use provided fixed metrics or parse metrics from report text.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/cl_outputs/classification_report.txt"),
        help="Path to classification report text file (used when --source report).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Directory where generated figures will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot window in addition to saving file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = provided_metrics() if args.source == "provided" else parse_report(args.report_path)
    outputs = generate_all_figures(parsed, args.output_dir, args.show)

    print("Saved plots:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
