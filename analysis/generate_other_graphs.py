"""
Generate "other graphs" for reviewer-facing diagnostics.

These figures are not required for core reproduction, but are useful for
strengthening the paper with distributional and robustness views.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


LOSS_ORDER = ["gt_score", "sharpe", "sortino", "simple"]
LOSS_LABELS = {
    "gt_score": "GT-Score",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "simple": "Simple",
}
LOSS_COLORS = {
    "gt_score": "#1f77b4",  # blue
    "sharpe": "#ff7f0e",    # orange
    "sortino": "#2ca02c",   # green
    "simple": "#d62728",    # red
}


def _load_json_records(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * np.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_walkforward_gen_ratio_ecdf(wf: pd.DataFrame, out_dir: Path) -> Path:
    df = wf.copy()
    df["gen_ratio"] = df["val_return"] / df["train_return"].replace({0.0: np.nan})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gen_ratio", "loss_function"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for loss in LOSS_ORDER:
        sub = df[df["loss_function"] == loss]["gen_ratio"].to_numpy()
        if sub.size == 0:
            continue
        x = np.sort(sub)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, label=LOSS_LABELS.get(loss, loss), color=LOSS_COLORS.get(loss))

    ax.set_title("Walk-Forward Generalization Ratio ECDF (val_return / train_return)")
    ax.set_xlabel("Generalization ratio")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    out_path = out_dir / "wf_generalization_ratio_ecdf.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_walkforward_gen_ratio_violin(wf: pd.DataFrame, out_dir: Path) -> Path:
    df = wf.copy()
    df["gen_ratio"] = df["val_return"] / df["train_return"].replace({0.0: np.nan})
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["gen_ratio", "loss_function"])

    data = []
    labels = []
    colors = []
    for loss in LOSS_ORDER:
        sub = df[df["loss_function"] == loss]["gen_ratio"].to_numpy()
        if sub.size == 0:
            continue
        data.append(sub)
        labels.append(LOSS_LABELS.get(loss, loss))
        colors.append(LOSS_COLORS.get(loss, "#333333"))

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Walk-Forward Generalization Ratio Distribution")
    ax.set_ylabel("val_return / train_return")
    ax.grid(True, axis="y", alpha=0.25)

    out_path = out_dir / "wf_generalization_ratio_violin.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_walkforward_win_rate(wf: pd.DataFrame, out_dir: Path) -> Path:
    df = wf[["asset", "strategy", "split_num", "loss_function", "val_return"]].copy()

    gt = df[df["loss_function"] == "gt_score"].rename(columns={"val_return": "gt_val"})
    results = []

    for baseline in ["sharpe", "sortino", "simple"]:
        bl = df[df["loss_function"] == baseline].rename(columns={"val_return": "bl_val"})
        merged = gt.merge(
            bl,
            on=["asset", "strategy", "split_num"],
            how="inner",
            suffixes=("_gt", "_bl"),
        )
        if merged.empty:
            continue

        wins = int((merged["gt_val"] > merged["bl_val"]).sum())
        losses = int((merged["gt_val"] < merged["bl_val"]).sum())
        ties = int((merged["gt_val"] == merged["bl_val"]).sum())
        n = wins + losses

        win_rate = (wins / n) if n > 0 else 0.0
        lo, hi = _wilson_ci(wins, n)
        results.append(
            {
                "baseline": baseline,
                "win_rate": win_rate,
                "ci_low": lo,
                "ci_high": hi,
                "n": n,
                "ties": ties,
            }
        )

    if not results:
        raise RuntimeError("No paired walk-forward results found to compute win rates.")

    res = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(res))
    y = res["win_rate"].to_numpy()
    yerr = np.vstack([y - res["ci_low"].to_numpy(), res["ci_high"].to_numpy() - y])

    ax.bar(
        x,
        y,
        color=[LOSS_COLORS.get(b, "#666666") for b in res["baseline"]],
        alpha=0.8,
        width=0.6,
    )
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=1)

    ax.axhline(0.5, color="black", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LOSS_LABELS.get(b, b) for b in res["baseline"]])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Win rate (GT-Score > baseline on val return)")
    ax.set_title("Walk-Forward Win Rate vs Baselines (95% Wilson CI)")
    ax.grid(True, axis="y", alpha=0.25)

    # Annotate sample sizes
    for i, row in res.iterrows():
        ax.text(
            i,
            min(0.98, row["win_rate"] + 0.06),
            f"n={int(row['n'])}\nties={int(row['ties'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out_path = out_dir / "wf_win_rate_val_return.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_walkforward_trade_frequency(wf: pd.DataFrame, out_dir: Path) -> Path:
    df = wf.copy()
    df["val_start"] = pd.to_datetime(df["val_start"])
    df["val_end"] = pd.to_datetime(df["val_end"])
    val_days = (df["val_end"] - df["val_start"]).dt.days.clip(lower=1)
    df["val_trades_per_year"] = df["val_trades"] / (val_days / 365.25)

    data = []
    labels = []
    colors = []
    for loss in LOSS_ORDER:
        sub = df[df["loss_function"] == loss]["val_trades_per_year"].dropna().to_numpy()
        if sub.size == 0:
            continue
        data.append(sub)
        labels.append(LOSS_LABELS.get(loss, loss))
        colors.append(LOSS_COLORS.get(loss, "#333333"))

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title("Walk-Forward Trade Frequency (Validation Window)")
    ax.set_ylabel("Trades per year")
    ax.grid(True, axis="y", alpha=0.25)

    out_path = out_dir / "wf_trade_frequency_val_trades_per_year.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_monte_carlo_cost_sensitivity(mc: pd.DataFrame, out_dir: Path) -> Path:
    df = mc.copy()

    cost_bps = np.array([0, 1, 5, 10], dtype=float)
    cost_per_side = cost_bps / 10000.0

    fig, ax = plt.subplots(figsize=(8, 5))

    for loss in LOSS_ORDER:
        sub = df[df["loss_function"] == loss].copy()
        if sub.empty:
            continue

        means = []
        for c in cost_per_side:
            # Approximate additional per-side cost: entry + exit => 2 * trades * c
            net = sub["test_return"] - (2.0 * sub["test_trades"] * c)
            means.append(float(net.mean()))

        ax.plot(
            cost_bps,
            means,
            marker="o",
            lw=2,
            label=LOSS_LABELS.get(loss, loss),
            color=LOSS_COLORS.get(loss),
        )

    ax.set_title("Monte Carlo: Transaction Cost Sensitivity (Test Return)")
    ax.set_xlabel("Additional cost per side (bps)")
    ax.set_ylabel("Mean net test return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)

    out_path = out_dir / "mc_transaction_cost_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_monte_carlo_qq_paired_differences(mc: pd.DataFrame, out_dir: Path) -> Path:
    df = mc[["asset", "strategy", "seed", "loss_function", "test_return"]].copy()
    gt = df[df["loss_function"] == "gt_score"].rename(columns={"test_return": "gt"})

    baselines = ["sharpe", "sortino", "simple"]
    fig, axes = plt.subplots(1, len(baselines), figsize=(12, 4))
    if len(baselines) == 1:
        axes = [axes]

    for ax, baseline in zip(axes, baselines):
        bl = df[df["loss_function"] == baseline].rename(columns={"test_return": "bl"})
        merged = gt.merge(bl, on=["asset", "strategy", "seed"], how="inner")
        diffs = (merged["gt"] - merged["bl"]).dropna().to_numpy()

        (osm, osr), (slope, intercept, r) = stats.probplot(diffs, dist="norm")
        ax.scatter(osm, osr, s=8, alpha=0.5, color=LOSS_COLORS.get(baseline, "#444444"))
        xline = np.array([osm.min(), osm.max()])
        ax.plot(xline, slope * xline + intercept, color="black", lw=1)

        ax.set_title(f"GT - {LOSS_LABELS.get(baseline, baseline)}\nQQ (r={r:.3f})")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Sample quantiles" if ax is axes[0] else "")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Paired Test-Return Differences vs Normal (Monte Carlo)")
    out_path = out_dir / "mc_qqplot_paired_test_return_differences.png"
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate additional graphs for reviewer review.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("output") / "other_graphs"),
        help="Output directory for generated images.",
    )
    parser.add_argument(
        "--walkforward",
        type=str,
        default=str(Path("output") / "results" / "walkforward_balanced.json"),
        help="Walk-forward results JSON path.",
    )
    parser.add_argument(
        "--monte-carlo",
        type=str,
        default=str(Path("output") / "results" / "monte_carlo_balanced.json"),
        help="Monte Carlo results JSON path.",
    )
    args = parser.parse_args()

    out_dir = _ensure_out_dir(Path(args.out_dir))
    wf_path = Path(args.walkforward)
    mc_path = Path(args.monte_carlo)

    if not wf_path.exists():
        raise FileNotFoundError(f"Walk-forward results not found: {wf_path}")
    if not mc_path.exists():
        raise FileNotFoundError(f"Monte Carlo results not found: {mc_path}")

    wf = _load_json_records(wf_path)
    mc = _load_json_records(mc_path)

    outputs = [
        plot_walkforward_gen_ratio_violin(wf, out_dir),
        plot_walkforward_gen_ratio_ecdf(wf, out_dir),
        plot_walkforward_win_rate(wf, out_dir),
        plot_walkforward_trade_frequency(wf, out_dir),
        plot_monte_carlo_cost_sensitivity(mc, out_dir),
        plot_monte_carlo_qq_paired_differences(mc, out_dir),
    ]

    # Emit a tiny manifest for quick browsing.
    manifest = out_dir / "README.md"
    with manifest.open("w") as f:
        f.write("# Other Graphs\n\n")
        f.write("Generated by `reproducible_code/analysis/generate_other_graphs.py`.\n\n")
        for p in outputs:
            f.write(f"- `{p.name}`\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

