#!/usr/bin/env python3
"""Compare PrismNet run results under a results/ directory.

This script scans metrics files produced by PrismNet:
  results/<run>/<protein>/out/evals/<identity>.metrics

It aggregates ACC/AUC/PRC per protein for each run and generates:
- A grouped bar chart (ACC/AUC/PRC) comparing runs per protein
- A CSV summary table

It also summarizes per-sample probability outputs (.probs):
- eval probs: out/evals/<identity>.probs ("prob\tlabel")
- infer probs: out/infer/<identity>*.probs ("prob")

Example:
  python compare_results.py \
    --results_dir results \
    --runs batch_20260514T101138 data2vec_20260513T161955 mlm_20260514T060352
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricRow:
    dataset_name: str
    acc: float
    auc: float
    prc: float
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass(frozen=True)
class EvalProbsStats:
    probs_path: str
    n: int
    n_pos: int
    n_neg: int
    mean_all: float
    mean_pos: float
    mean_neg: float
    q05_all: float
    q50_all: float
    q95_all: float
    q05_pos: float
    q50_pos: float
    q95_pos: float
    q05_neg: float
    q50_neg: float
    q95_neg: float
    acc_at_0p5: float
    auc_from_probs: float
    ap_from_probs: float


@dataclass(frozen=True)
class InferProbsStats:
    probs_path: str
    n: int
    mean: float
    std: float
    min: float
    q05: float
    q50: float
    q95: float
    max: float


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return num / den


def _quantile_sorted(xs_sorted: Sequence[float], qv: float) -> float:
    """Linear-interpolated quantile for a pre-sorted sequence."""
    n = len(xs_sorted)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(xs_sorted[0])
    pos = (n - 1) * qv
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs_sorted[lo])
    w = pos - lo
    return float(xs_sorted[lo] * (1 - w) + xs_sorted[hi] * w)


def parse_metrics_file(path: Path) -> MetricRow:
    line = path.read_text().strip().splitlines()[0].strip()
    parts = line.split()  # handles tab/space
    if len(parts) < 8:
        raise ValueError(f"Unexpected metrics format in {path}: {line}")

    return MetricRow(
        dataset_name=parts[0],
        acc=_safe_float(parts[1]),
        auc=_safe_float(parts[2]),
        prc=_safe_float(parts[3]),
        tp=_safe_int(parts[4]),
        tn=_safe_int(parts[5]),
        fp=_safe_int(parts[6]),
        fn=_safe_int(parts[7]),
    )


def find_run_metrics(results_dir: Path, run_name: str) -> Dict[str, MetricRow]:
    """Return mapping: protein -> metrics for the run."""
    run_dir = results_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    protein_to_metrics: Dict[str, MetricRow] = {}

    # Expected: <run>/<protein>/out/evals/*.metrics
    for metrics_path in run_dir.glob("*/out/evals/*.metrics"):
        protein = metrics_path.parent.parent.parent.name  # .../<protein>/out/evals
        try:
            protein_to_metrics[protein] = parse_metrics_file(metrics_path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse {metrics_path}: {e}") from e

    return protein_to_metrics


def parse_eval_probs_file(path: Path) -> Tuple[List[float], List[int]]:
    probs: List[float] = []
    labels: List[int] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        p = _safe_float(parts[0])
        y = _safe_int(parts[1])
        if isinstance(p, float) and math.isnan(p):
            continue
        probs.append(float(p))
        labels.append(1 if y != 0 else 0)
    return probs, labels


def parse_infer_probs_file(path: Path) -> List[float]:
    probs: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = _safe_float(line.split()[0])
        if isinstance(p, float) and math.isnan(p):
            continue
        probs.append(float(p))
    return probs


def summarize_eval_probs(
    probs_path: Path,
    metrics_row: Optional[MetricRow],
) -> EvalProbsStats:
    probs, labels = parse_eval_probs_file(probs_path)

    n = len(probs)
    if n == 0:
        return EvalProbsStats(
            probs_path=str(probs_path),
            n=0,
            n_pos=0,
            n_neg=0,
            mean_all=float("nan"),
            mean_pos=float("nan"),
            mean_neg=float("nan"),
            q05_all=float("nan"),
            q50_all=float("nan"),
            q95_all=float("nan"),
            q05_pos=float("nan"),
            q50_pos=float("nan"),
            q95_pos=float("nan"),
            q05_neg=float("nan"),
            q50_neg=float("nan"),
            q95_neg=float("nan"),
            acc_at_0p5=float("nan"),
            auc_from_probs=float("nan"),
            ap_from_probs=float("nan"),
        )

    n_pos = sum(labels)
    n_neg = n - n_pos

    probs_sorted = sorted(probs)
    probs_pos_sorted = sorted([p for p, y in zip(probs, labels) if y == 1])
    probs_neg_sorted = sorted([p for p, y in zip(probs, labels) if y == 0])

    mean_all = sum(probs) / n
    mean_pos = _safe_div(sum(p for p, y in zip(probs, labels) if y == 1), n_pos)
    mean_neg = _safe_div(sum(p for p, y in zip(probs, labels) if y == 0), n_neg)

    q05_all = _quantile_sorted(probs_sorted, 0.05)
    q50_all = _quantile_sorted(probs_sorted, 0.50)
    q95_all = _quantile_sorted(probs_sorted, 0.95)
    q05_pos = _quantile_sorted(probs_pos_sorted, 0.05)
    q50_pos = _quantile_sorted(probs_pos_sorted, 0.50)
    q95_pos = _quantile_sorted(probs_pos_sorted, 0.95)
    q05_neg = _quantile_sorted(probs_neg_sorted, 0.05)
    q50_neg = _quantile_sorted(probs_neg_sorted, 0.50)
    q95_neg = _quantile_sorted(probs_neg_sorted, 0.95)

    acc_at_0p5 = _safe_div(
        sum((1 if (p >= 0.5) else 0) == y for p, y in zip(probs, labels)),
        n,
    )

    auc_from_probs = float("nan")
    ap_from_probs = float("nan")
    # Compute only if both classes are present.
    if n_pos > 0 and n_neg > 0:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            auc_from_probs = float(roc_auc_score(labels, probs))
            ap_from_probs = float(average_precision_score(labels, probs))
        except Exception:
            # Keep NaN if sklearn isn't available.
            pass

    return EvalProbsStats(
        probs_path=str(probs_path),
        n=n,
        n_pos=n_pos,
        n_neg=n_neg,
        mean_all=float(mean_all),
        mean_pos=float(mean_pos),
        mean_neg=float(mean_neg),
        q05_all=float(q05_all),
        q50_all=float(q50_all),
        q95_all=float(q95_all),
        q05_pos=float(q05_pos),
        q50_pos=float(q50_pos),
        q95_pos=float(q95_pos),
        q05_neg=float(q05_neg),
        q50_neg=float(q50_neg),
        q95_neg=float(q95_neg),
        acc_at_0p5=float(acc_at_0p5),
        auc_from_probs=float(auc_from_probs),
        ap_from_probs=float(ap_from_probs),
    )


def summarize_infer_probs(probs_path: Path) -> InferProbsStats:
    probs = parse_infer_probs_file(probs_path)
    n = len(probs)
    if n == 0:
        return InferProbsStats(
            probs_path=str(probs_path),
            n=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            q05=float("nan"),
            q50=float("nan"),
            q95=float("nan"),
            max=float("nan"),
        )

    probs_sorted = sorted(probs)
    mean = sum(probs_sorted) / n
    var = sum((p - mean) ** 2 for p in probs_sorted) / max(n - 1, 1)
    std = math.sqrt(var)

    return InferProbsStats(
        probs_path=str(probs_path),
        n=n,
        mean=float(mean),
        std=float(std),
        min=float(probs_sorted[0]),
        q05=float(_quantile_sorted(probs_sorted, 0.05)),
        q50=float(_quantile_sorted(probs_sorted, 0.50)),
        q95=float(_quantile_sorted(probs_sorted, 0.95)),
        max=float(probs_sorted[-1]),
    )


def find_run_eval_probs_stats(
    results_dir: Path,
    run_name: str,
    metrics_by_protein: Dict[str, MetricRow],
) -> Dict[str, EvalProbsStats]:
    """Return mapping: protein -> stats for eval probs (.probs with labels)."""
    run_dir = results_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    protein_to_stats: Dict[str, EvalProbsStats] = {}
    for probs_path in run_dir.glob("*/out/evals/*.probs"):
        # Only treat this as eval probs if a matching .metrics exists.
        if not probs_path.with_suffix(".metrics").exists():
            continue
        protein = probs_path.parent.parent.parent.name
        protein_to_stats[protein] = summarize_eval_probs(
            probs_path, metrics_row=metrics_by_protein.get(protein)
        )

    return protein_to_stats


def find_run_eval_probs_distributions(
    results_dir: Path,
    run_name: str,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """Return mapping: protein -> (pos_probs, neg_probs) for eval probs."""
    run_dir = results_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    protein_to_dist: Dict[str, Tuple[List[float], List[float]]] = {}
    for probs_path in run_dir.glob("*/out/evals/*.probs"):
        if not probs_path.with_suffix(".metrics").exists():
            continue
        protein = probs_path.parent.parent.parent.name
        probs, labels = parse_eval_probs_file(probs_path)
        pos_probs = [p for p, y in zip(probs, labels) if y == 1]
        neg_probs = [p for p, y in zip(probs, labels) if y == 0]
        protein_to_dist[protein] = (pos_probs, neg_probs)
    return protein_to_dist


def find_run_infer_probs_stats(results_dir: Path, run_name: str) -> List[Tuple[str, InferProbsStats]]:
    """Return list of (protein, stats) for infer probs (.probs without labels)."""
    run_dir = results_dir / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    items: List[Tuple[str, InferProbsStats]] = []
    for probs_path in run_dir.glob("*/out/infer/*.probs"):
        protein = probs_path.parent.parent.parent.name
        items.append((protein, summarize_infer_probs(probs_path)))
    return items


def union_proteins(per_run: Dict[str, Dict[str, MetricRow]]) -> List[str]:
    proteins = set()
    for run_metrics in per_run.values():
        proteins |= set(run_metrics.keys())
    return sorted(proteins)


def _nanmean(values: Sequence[float]) -> float:
    xs = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def write_csv(
    out_csv: Path,
    runs: Sequence[str],
    proteins: Sequence[str],
    per_run: Dict[str, Dict[str, MetricRow]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "protein",
                "run",
                "dataset_name",
                "acc",
                "auc",
                "prc",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )
        for protein in proteins:
            for run in runs:
                row = per_run.get(run, {}).get(protein)
                if row is None:
                    w.writerow([protein, run, "", "", "", "", "", "", "", ""])
                else:
                    w.writerow(
                        [
                            protein,
                            run,
                            row.dataset_name,
                            f"{row.acc:.6f}",
                            f"{row.auc:.6f}",
                            f"{row.prc:.6f}",
                            row.tp,
                            row.tn,
                            row.fp,
                            row.fn,
                        ]
                    )


def write_eval_probs_stats_csv(
    out_csv: Path,
    runs: Sequence[str],
    proteins: Sequence[str],
    metrics_per_run: Dict[str, Dict[str, MetricRow]],
    eval_probs_per_run: Dict[str, Dict[str, EvalProbsStats]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "protein",
                "run",
                "eval_probs_path",
                "n",
                "n_pos",
                "n_neg",
                "mean_all",
                "mean_pos",
                "mean_neg",
                "q05_all",
                "q50_all",
                "q95_all",
                "q05_pos",
                "q50_pos",
                "q95_pos",
                "q05_neg",
                "q50_neg",
                "q95_neg",
                "acc_at_0.5",
                "auc_reported",
                "prc_reported",
                "auc_from_probs",
                "ap_from_probs",
            ]
        )
        for protein in proteins:
            for run in runs:
                met = metrics_per_run.get(run, {}).get(protein)
                st = eval_probs_per_run.get(run, {}).get(protein)
                if st is None:
                    w.writerow(
                        [
                            protein,
                            run,
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )
                    continue
                w.writerow(
                    [
                        protein,
                        run,
                        st.probs_path,
                        st.n,
                        st.n_pos,
                        st.n_neg,
                        f"{st.mean_all:.6f}",
                        f"{st.mean_pos:.6f}",
                        f"{st.mean_neg:.6f}",
                        f"{st.q05_all:.6f}" if not math.isnan(st.q05_all) else "",
                        f"{st.q50_all:.6f}" if not math.isnan(st.q50_all) else "",
                        f"{st.q95_all:.6f}" if not math.isnan(st.q95_all) else "",
                        f"{st.q05_pos:.6f}" if not math.isnan(st.q05_pos) else "",
                        f"{st.q50_pos:.6f}" if not math.isnan(st.q50_pos) else "",
                        f"{st.q95_pos:.6f}" if not math.isnan(st.q95_pos) else "",
                        f"{st.q05_neg:.6f}" if not math.isnan(st.q05_neg) else "",
                        f"{st.q50_neg:.6f}" if not math.isnan(st.q50_neg) else "",
                        f"{st.q95_neg:.6f}" if not math.isnan(st.q95_neg) else "",
                        f"{st.acc_at_0p5:.6f}",
                        "" if met is None else f"{met.auc:.6f}",
                        "" if met is None else f"{met.prc:.6f}",
                        f"{st.auc_from_probs:.6f}" if not math.isnan(st.auc_from_probs) else "",
                        f"{st.ap_from_probs:.6f}" if not math.isnan(st.ap_from_probs) else "",
                    ]
                )


def write_infer_probs_stats_csv(
    out_csv: Path,
    runs: Sequence[str],
    infer_probs_per_run: Dict[str, List[Tuple[str, InferProbsStats]]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "protein",
                "run",
                "infer_probs_path",
                "n",
                "mean",
                "std",
                "min",
                "q05",
                "q50",
                "q95",
                "max",
            ]
        )
        for run in runs:
            for protein, st in infer_probs_per_run.get(run, []):
                w.writerow(
                    [
                        protein,
                        run,
                        st.probs_path,
                        st.n,
                        f"{st.mean:.6f}" if not math.isnan(st.mean) else "",
                        f"{st.std:.6f}" if not math.isnan(st.std) else "",
                        f"{st.min:.6f}" if not math.isnan(st.min) else "",
                        f"{st.q05:.6f}" if not math.isnan(st.q05) else "",
                        f"{st.q50:.6f}" if not math.isnan(st.q50) else "",
                        f"{st.q95:.6f}" if not math.isnan(st.q95) else "",
                        f"{st.max:.6f}" if not math.isnan(st.max) else "",
                    ]
                )


def plot_eval_prob_means(
    out_png: Path,
    title: str,
    runs: Sequence[str],
    proteins: Sequence[str],
    eval_probs_per_run: Dict[str, Dict[str, EvalProbsStats]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("mean(prob|pos)", "mean_pos"),
        ("mean(prob|neg)", "mean_neg"),
    ]

    n_proteins = len(proteins)
    n_runs = len(runs)
    x = np.arange(n_proteins)
    bar_width = min(0.8 / max(n_runs, 1), 0.25)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_width

    fig, axes = plt.subplots(len(panels), 1, figsize=(max(12, n_proteins * 0.7), 6), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (ylabel, key) in zip(axes, panels):
        for i, run in enumerate(runs):
            ys: List[float] = []
            for protein in proteins:
                st = eval_probs_per_run.get(run, {}).get(protein)
                ys.append(float("nan") if st is None else float(getattr(st, key)))
            ax.bar(x + offsets[i], ys, width=bar_width, label=run)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    axes[0].set_title(title)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(proteins, rotation=45, ha="right")
    axes[0].legend(loc="upper right", ncol=1, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _slugify(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _stable_seed(text: str) -> int:
    # Use 32-bit seed derived from md5 for reproducibility.
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _scatter_with_jitter(ax, x_center: float, ys: Sequence[float], *, jitter: float, seed: int, **kwargs) -> None:
    # Deterministic jitter per group for reproducible images.
    import numpy as np

    rng = np.random.default_rng(seed)
    xs = x_center + rng.uniform(-jitter, jitter, size=len(ys))
    ax.scatter(xs, ys, **kwargs)


def _downsample_points(ys: Sequence[float], max_points: int, *, seed: int) -> List[float]:
    """Downsample points for scatter overlay.

    Uses deterministic random sampling (seeded) to keep density impression while
    reducing clutter.
    """
    if max_points <= 0:
        return []
    ys_list = list(ys)
    n = len(ys_list)
    if n <= max_points:
        return ys_list

    import numpy as np

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return [ys_list[int(i)] for i in idx]


def plot_eval_prob_boxplots_scatter_per_run(
    out_dir: Path,
    title_prefix: str,
    runs: Sequence[str],
    proteins: Sequence[str],
    dists_per_run: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    scatter_max_points: int,
) -> List[Path]:
    """Boxplots per run, split by pos/neg, with scatter overlay.

    Writes one PNG per run into out_dir and returns written paths.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    n_proteins = len(proteins)
    fig_w = max(12, n_proteins * 0.65)
    fig_h = 7

    positions = list(range(1, n_proteins + 1))
    jitter = 0.18

    for run in runs:
        dist = dists_per_run.get(run, {})
        pos_data: List[List[float]] = []
        neg_data: List[List[float]] = []
        for protein in proteins:
            pos_probs, neg_probs = dist.get(protein, ([], []))
            # Matplotlib boxplot can't handle empty datasets; use a NaN placeholder.
            pos_data.append(pos_probs if len(pos_probs) > 0 else [float("nan")])
            neg_data.append(neg_probs if len(neg_probs) > 0 else [float("nan")])

        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, sharey=True)
        ax_pos, ax_neg = axes

        ax_pos.boxplot(pos_data, positions=positions, showfliers=False)
        ax_neg.boxplot(neg_data, positions=positions, showfliers=False)

        # Scatter overlays
        for i, protein in enumerate(proteins):
            pos_probs, neg_probs = dist.get(protein, ([], []))
            if pos_probs:
                pos_plot = _downsample_points(
                    pos_probs,
                    scatter_max_points,
                    seed=_stable_seed(f"{run}::{protein}::pos::downsample"),
                )
                _scatter_with_jitter(
                    ax_pos,
                    positions[i],
                    pos_plot,
                    jitter=jitter,
                    seed=_stable_seed(f"{run}::{protein}::pos"),
                    s=6,
                    alpha=0.25,
                    linewidths=0,
                    rasterized=True,
                )
            if neg_probs:
                neg_plot = _downsample_points(
                    neg_probs,
                    scatter_max_points,
                    seed=_stable_seed(f"{run}::{protein}::neg::downsample"),
                )
                _scatter_with_jitter(
                    ax_neg,
                    positions[i],
                    neg_plot,
                    jitter=jitter,
                    seed=_stable_seed(f"{run}::{protein}::neg"),
                    s=6,
                    alpha=0.25,
                    linewidths=0,
                    rasterized=True,
                )

        ax_pos.set_title(f"{run}  pos (label=1)")
        ax_neg.set_title(f"{run}  neg (label=0)")
        ax_neg.set_xlabel("protein")
        ax_pos.set_ylabel("prob")
        ax_neg.set_ylabel("prob")

        for ax in (ax_pos, ax_neg):
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        ax_neg.set_xticks(positions)
        ax_neg.set_xticklabels(proteins, rotation=45, ha="right")

        fig.suptitle(f"{title_prefix} ({run})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_png = out_dir / f"compare_eval_prob_boxplots_scatter__{_slugify(run)}.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        written.append(out_png)

    return written


def plot_eval_prob_histograms(
    out_png: Path,
    title: str,
    runs: Sequence[str],
    dists_per_run: Dict[str, Dict[str, Tuple[List[float], List[float]]]],
    bins: int = 50,
) -> None:
    """Histograms aggregated across proteins, split by pos/neg, for each run."""
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    n_runs = len(runs)
    fig, axes = plt.subplots(n_runs, 1, figsize=(12, max(3.0, 2.8 * n_runs)), sharex=True, sharey=False)
    if n_runs == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        dist = dists_per_run.get(run, {})
        pos_all: List[float] = []
        neg_all: List[float] = []
        for pos_probs, neg_probs in dist.values():
            pos_all.extend(pos_probs)
            neg_all.extend(neg_probs)

        ax.hist(neg_all, bins=bins, range=(0.0, 1.0), alpha=0.45, label=f"neg (n={len(neg_all)})")
        ax.hist(pos_all, bins=bins, range=(0.0, 1.0), alpha=0.45, label=f"pos (n={len(pos_all)})")
        ax.set_title(run)
        ax.set_ylabel("count")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("prob")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_grouped_bars(
    out_png: Path,
    title: str,
    runs: Sequence[str],
    proteins: Sequence[str],
    per_run: Dict[str, Dict[str, MetricRow]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("AUC", "auc"),
        ("ACC", "acc"),
        ("PRC", "prc"),
    ]

    n_proteins = len(proteins)
    n_runs = len(runs)

    x = np.arange(n_proteins)
    bar_width = min(0.8 / max(n_runs, 1), 0.25)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_width

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(12, n_proteins * 0.7), 9), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (label, key) in zip(axes, metrics):
        for i, run in enumerate(runs):
            ys: List[float] = []
            for protein in proteins:
                row = per_run.get(run, {}).get(protein)
                ys.append(float("nan") if row is None else float(getattr(row, key)))
            ax.bar(x + offsets[i], ys, width=bar_width, label=run)

        ax.set_ylabel(label)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Add per-run mean in the corner
        means = []
        for run in runs:
            ys = [float(getattr(per_run[run][p], key)) for p in proteins if p in per_run.get(run, {})]
            means.append(_nanmean(ys))
        mean_text = "  ".join(f"{run}: {m:.3f}" for run, m in zip(runs, means))
        ax.text(0.01, 0.98, f"mean  {mean_text}", transform=ax.transAxes, va="top", ha="left", fontsize=9)

    axes[0].set_title(title)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(proteins, rotation=45, ha="right")
    axes[0].legend(loc="upper right", ncol=1, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PrismNet results across multiple runs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing run subdirectories (default: results)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Run directory names under results_dir. If omitted, uses all directories under results_dir.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots/CSVs (default: <results_dir>/comparisons/<timestamp>/)",
    )
    parser.add_argument(
        "--scatter_max_points",
        type=int,
        default=200,
        help="Max points per protein per class (pos/neg) to draw in scatter overlays (default: 200). Set 0 to disable scatter.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    runs: List[str]
    if args.runs is None:
        runs = sorted([p.name for p in results_dir.iterdir() if p.is_dir()])
    else:
        runs = list(args.runs)

    if len(runs) < 2:
        raise ValueError("Please provide at least 2 runs to compare (via --runs)")

    per_run: Dict[str, Dict[str, MetricRow]] = {}
    for run in runs:
        per_run[run] = find_run_metrics(results_dir, run)

    proteins = union_proteins(per_run)
    if not proteins:
        raise RuntimeError(
            "No metrics files found. Expected something like results/<run>/<protein>/out/evals/*.metrics"
        )

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (results_dir / "comparisons" / ts)

    out_png = out_dir / "compare_acc_auc_prc.png"
    out_csv = out_dir / "compare_acc_auc_prc.csv"
    out_eval_probs_csv = out_dir / "compare_eval_probs_stats.csv"
    out_infer_probs_csv = out_dir / "compare_infer_probs_stats.csv"
    out_eval_probs_png = out_dir / "compare_eval_prob_means.png"
    out_eval_hist_png = out_dir / "compare_eval_prob_histograms.png"

    title = f"PrismNet results comparison ({', '.join(runs)})"

    # Metrics (.metrics)
    write_csv(out_csv, runs=runs, proteins=proteins, per_run=per_run)
    plot_grouped_bars(out_png, title=title, runs=runs, proteins=proteins, per_run=per_run)

    # Probs summaries (.probs)
    eval_probs_per_run: Dict[str, Dict[str, EvalProbsStats]] = {}
    infer_probs_per_run: Dict[str, List[Tuple[str, InferProbsStats]]] = {}
    eval_dists_per_run: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
    for run in runs:
        eval_probs_per_run[run] = find_run_eval_probs_stats(results_dir, run, per_run.get(run, {}))
        infer_probs_per_run[run] = find_run_infer_probs_stats(results_dir, run)
        eval_dists_per_run[run] = find_run_eval_probs_distributions(results_dir, run)

    write_eval_probs_stats_csv(
        out_eval_probs_csv,
        runs=runs,
        proteins=proteins,
        metrics_per_run=per_run,
        eval_probs_per_run=eval_probs_per_run,
    )
    write_infer_probs_stats_csv(out_infer_probs_csv, runs=runs, infer_probs_per_run=infer_probs_per_run)

    # Optional visualization for eval probs (mean prob by class)
    if any(eval_probs_per_run[run] for run in runs):
        plot_eval_prob_means(
            out_eval_probs_png,
            title=f"PrismNet eval prob means ({', '.join(runs)})",
            runs=runs,
            proteins=proteins,
            eval_probs_per_run=eval_probs_per_run,
        )
        written_box_scatter = plot_eval_prob_boxplots_scatter_per_run(
            out_dir,
            title_prefix="PrismNet eval prob boxplots + scatter",
            runs=runs,
            proteins=proteins,
            dists_per_run=eval_dists_per_run,
            scatter_max_points=args.scatter_max_points,
        )
        plot_eval_prob_histograms(
            out_eval_hist_png,
            title=f"PrismNet eval prob histograms ({', '.join(runs)})",
            runs=runs,
            dists_per_run=eval_dists_per_run,
        )

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_eval_probs_csv}")
    print(f"Wrote: {out_infer_probs_csv}")
    if out_eval_probs_png.exists():
        print(f"Wrote: {out_eval_probs_png}")
    if out_eval_hist_png.exists():
        print(f"Wrote: {out_eval_hist_png}")
    for p in written_box_scatter if 'written_box_scatter' in locals() else []:
        if p.exists():
            print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
