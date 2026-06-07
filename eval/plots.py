"""Plots (§9) — Stream 3 (O7). The figures that are the demo's evidence.

Five matplotlib figures, always over multiple seeds/episodes with variance
(Law 4 — never a single anecdote):

  1. learning curve       — team frontier vs training episode (Full C2)
  2. bandit arm-value     — each arm's running value over episodes (S1 + S3)
  3. transfer bar chart   — 3 conditions × {train, held-out}, ±1 std
  4. ablation bar chart   — Full C2 vs memory/coaching/gate-off on held-out
  5. invalid-rate         — team invalid-action rate over training episodes

``matplotlib`` is imported lazily (Agg backend) so importing ``eval`` — and
running ``run.py`` / ``pytest`` — never requires it (Green-main law).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .harness import (
    TrainResult,
    TransferResult,
    run_ablations,
    run_learning_curve,
    run_transfer,
)
from .records import HELDOUT, TRAIN, EpisodeRecord, summarize

# Default provenance stamped on every figure so a figure on its own is never
# mistaken for a real-environment result (the headline is offline until Stream 2's
# LLMWorker is evaluated via RealRunner).
SIM_SOURCE = "source: offline calibrated outcome model (eval/outcome_model.py) — NOT the real LLM-worker env"
REAL_SOURCE = "source: real env + LLM workers (eval.harness.RealRunner)"

_CONDITION_ORDER = ["static", "comms", "full_c2"]
_CONDITION_PRETTY = {"static": "Static baseline", "comms": "Comms (no Orca)", "full_c2": "Full C2"}
_SITUATION_PRETTY = {
    "S1_role_assignment": "S1 · role assignment",
    "S2_nether_entry": "S2 · nether entry",
    "S3_fortress_search": "S3 · fortress search",
    "S4_end_approach": "S4 · end approach",
}


def situation_label(sit: str) -> str:
    return _SITUATION_PRETTY.get(sit, sit)


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt

    return plt


def _rolling(xs: list[float], k: int = 5) -> list[float]:
    out: list[float] = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        window = xs[lo : i + 1]
        out.append(sum(window) / len(window))
    return out


def _stamp_source(fig, source: str) -> None:
    """Stamp the data-source provenance as a small footnote on every figure."""
    fig.text(0.005, 0.005, source, ha="left", va="bottom", fontsize=6.5, style="italic", color="dimgray")


def _source_label(runner) -> str:
    from .harness import RealRunner

    return REAL_SOURCE if isinstance(runner, RealRunner) else SIM_SOURCE


# --------------------------------------------------------------------------- #
def plot_learning_curve(tr: TrainResult, path: str | Path, *, source: str = SIM_SOURCE) -> Path:
    plt = _plt()
    fr = [r.frontier_value for r in tr.learning]
    eps = list(range(len(fr)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(eps, fr, ".", alpha=0.35, color="tab:blue", label="episode frontier")
    ax.plot(eps, _rolling(fr), "-", color="tab:blue", lw=2.2, label="rolling mean (5)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("team frontier (objective DAG)")
    ax.set_title("Learning curve — Full C2 on train seeds {A, T2, T3}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _stamp_source(fig, source)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_bandit_values(
    tr: TrainResult,
    path: str | Path,
    situations: Optional[list[str]] = None,
    *,
    source: str = SIM_SOURCE,
) -> Path:
    plt = _plt()
    from orca.situations import S1, S3

    situations = situations or [S1, S3]
    snaps = tr.value_snapshots
    eps = list(range(len(snaps)))
    fig, axes = plt.subplots(1, len(situations), figsize=(6.2 * len(situations), 4.2), squeeze=False)
    for ax, sit in zip(axes[0], situations):
        arms = list(snaps[-1].get(sit, {}).keys()) if snaps else []
        for arm in arms:
            series = [snap.get(sit, {}).get(arm, 0.0) for snap in snaps]
            ax.plot(eps, series, "-", lw=1.8, label=arm)
        if arms:
            best = max(arms, key=lambda a: snaps[-1][sit][a])
            ax.annotate(
                f"learned: {best}",
                xy=(eps[-1], snaps[-1][sit][best]),
                xytext=(0.05, 0.92),
                textcoords="axes fraction",
                fontsize=9,
                fontweight="bold",
                color="black",
            )
        ax.set_title(f"Bandit arm values — {situation_label(sit)}")
        ax.set_xlabel("training episode")
        ax.set_ylabel("estimated value")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _stamp_source(fig, source)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_transfer(records: list[EpisodeRecord], path: str | Path, *, source: str = SIM_SOURCE) -> Path:
    plt = _plt()
    import numpy as np

    stats = summarize(records, field_name="frontier_value")
    conds = [c for c in _CONDITION_ORDER if any(r.condition == c for r in records)]
    x = np.arange(len(conds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, split in enumerate((TRAIN, HELDOUT)):
        means = [stats.get((c, split)).mean if (c, split) in stats else 0.0 for c in conds]
        errs = [stats.get((c, split)).std if (c, split) in stats else 0.0 for c in conds]
        ax.bar(
            x + (i - 0.5) * w,
            means,
            w,
            yerr=errs,
            capsize=4,
            label="held-out {B,C}" if split == HELDOUT else "train {A,T2,T3}",
            color="tab:green" if split == HELDOUT else "tab:gray",
            alpha=0.9 if split == HELDOUT else 0.7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_CONDITION_PRETTY.get(c, c) for c in conds])
    ax.set_ylabel("mean team frontier (±1 std)")
    ax.set_title("Transfer (calibrated outcome model): Full C2 ≥ baselines on held-out")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _stamp_source(fig, source)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_ablation(records: list[EpisodeRecord], path: str | Path, *, source: str = SIM_SOURCE) -> Path:
    plt = _plt()
    stats = summarize(records, field_name="frontier_value")
    order = ["full_c2", "no_memory", "no_coaching", "no_gate"]
    pretty = {
        "full_c2": "Full C2",
        "no_memory": "− memory",
        "no_coaching": "− coaching",
        "no_gate": "− accept-gate",
    }
    conds = [c for c in order if (c, HELDOUT) in stats]
    means = [stats[(c, HELDOUT)].mean for c in conds]
    errs = [stats[(c, HELDOUT)].std for c in conds]
    colors = ["tab:green"] + ["tab:orange"] * (len(conds) - 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([pretty.get(c, c) for c in conds], means, yerr=errs, capsize=4, color=colors, alpha=0.9)
    ax.set_ylabel("mean held-out frontier (±1 std)")
    ax.set_title("Ablations — each knob removed from Full C2 (held-out {B,C})")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _stamp_source(fig, source)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_invalid_rate(tr: TrainResult, path: str | Path, *, source: str = SIM_SOURCE) -> Path:
    plt = _plt()
    inv = [r.invalid_rate for r in tr.learning]
    eps = list(range(len(inv)))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(eps, inv, ".", alpha=0.35, color="tab:red", label="episode invalid-rate")
    ax.plot(eps, _rolling(inv), "-", color="tab:red", lw=2.2, label="rolling mean (5)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("team invalid-action rate")
    ax.set_title("Invalid-rate over training — coaching clears the miner bottleneck")
    ax.set_ylim(0, max(0.4, max(inv) * 1.15 if inv else 0.4))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _stamp_source(fig, source)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


# --------------------------------------------------------------------------- #
def make_all_plots(
    out_dir: str | Path = "figures",
    *,
    runner=None,
    n_train: int = 40,
    eval_reps: int = 8,
) -> dict[str, Path]:
    """Run the experiments and write all five figures; return their paths (§9)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = _source_label(runner)

    lc = run_learning_curve(n_train=n_train, runner=runner)
    transfer: TransferResult = run_transfer(n_train=n_train, eval_reps=eval_reps, runner=runner)
    ablation = run_ablations(n_train=n_train, eval_reps=eval_reps, runner=runner)

    paths = {
        "learning_curve": plot_learning_curve(lc, out / "learning_curve.png", source=source),
        "bandit_values": plot_bandit_values(lc, out / "bandit_values.png", source=source),
        "transfer": plot_transfer(transfer.records, out / "transfer.png", source=source),
        "ablation": plot_ablation(ablation, out / "ablation.png", source=source),
        "invalid_rate": plot_invalid_rate(lc, out / "invalid_rate.png", source=source),
    }
    return paths


__all__ = [
    "plot_learning_curve",
    "plot_bandit_values",
    "plot_transfer",
    "plot_ablation",
    "plot_invalid_rate",
    "make_all_plots",
]
