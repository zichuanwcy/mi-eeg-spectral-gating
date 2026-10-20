"""Render historical quantitative panels from their frozen CSV inputs.

Use scripts/plot_results.py to select input/output paths. These panels
predate the final author-revised manuscript artwork.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tifffile


ROOT = Path.cwd()
BASE = ROOT / "results" / "historical_figures"
SOURCE = ROOT / "data" / "derived" / "figure_source"
OUT = BASE / "main"
QA = BASE / "qa"

OPENBMI = "#365F7D"
PHYSIONET = "#9A652F"
F0 = "#4F6E68"
TRADITIONAL = "#80596D"
EA = "#496B70"
SE = "#675A78"
INK = "#25282B"
MID = "#64696E"
LIGHT = "#D2D5D8"
PALE = "#F4F5F6"
WHITE = "#FFFFFF"

SIZES_MM = {
    "Fig02_gate_gain": (183, 95),
    "Fig03_optionA_stability_scatter": (183, 82),
    "Fig03_optionB_stability_counts": (183, 82),
    "Fig04_bci_diagnostics": (183, 105),
    "Fig05_se_transfer": (183, 78),
    "Fig06_fixed_strategy_scatter": (183, 82),
    "Fig07_fixed_strategy_difference": (183, 82),
}


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "axes.titlesize": 7.6,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.75,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(SOURCE / name)


def clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.7, width=0.7, color=INK)


def panel(ax, letter: str, x: float = -0.08, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.2, fontweight="bold", color=INK, clip_on=False)


def save(fig, figure_id: str) -> None:
    width_mm, height_mm = SIZES_MM[figure_id]
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / figure_id
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=600)
    plt.close(fig)

    with Image.open(png) as im:
        rgb = im.convert("RGB")
        ImageOps.grayscale(rgb).save(
            OUT / f"{figure_id}_grayscale.png", dpi=(600, 600)
        )
        tifffile.imwrite(
            stem.with_suffix(".tiff"),
            np.asarray(rgb),
            photometric="rgb",
            compression="deflate",
            resolution=(600, 600),
        )

    expected = (
        round(width_mm / 25.4 * 600),
        round(height_mm / 25.4 * 600),
    )
    with Image.open(png) as im:
        if abs(im.width - expected[0]) > 2 or abs(im.height - expected[1]) > 2:
            raise AssertionError(f"{figure_id}: {im.size} != {expected}")


def locked_stats(figure: str) -> pd.DataFrame:
    data = read("Locked_Statistics_Stage7.csv")
    return data[data["figure"] == figure].set_index("dataset")


def rank_lollipop(ax, values: np.ndarray, color: str, *, xlabel: str,
                  ylabel: str, title: str, n: int, marker: str) -> None:
    values = np.sort(np.asarray(values, dtype=float))
    x = np.arange(1, len(values) + 1)
    ax.vlines(x, 0, values, color=color, alpha=0.24, lw=0.55)
    ax.scatter(x, values, s=13, marker=marker, facecolor=WHITE,
               edgecolor=color, linewidth=0.75, zorder=3)
    ax.axhline(0, color=INK, ls="--", lw=0.8)
    ax.set_xlim(0, len(values) + 1)
    lim = max(5.0, math.ceil(max(abs(values.min()), abs(values.max())) / 5) * 5)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([1, len(values)])
    ax.set_xticklabels(["1", str(len(values))])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", color=color, fontweight="bold", pad=4)
    neg = int((values < 0).sum())
    zero = int(np.isclose(values, 0).sum())
    pos = int((values > 0).sum())
    ax.text(0.98, 0.03, f"n={n}; +/0/− = {pos}/{zero}/{neg}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.3, color=MID)
    clean(ax)


def render_fig02() -> None:
    subjects = read("Fig02_subject_gate_gain.csv")
    stats = locked_stats("Fig02")
    fig = plt.figure(figsize=(183 / 25.4, 95 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.82], wspace=0.38)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.91)

    openbmi = subjects.query("dataset == 'openbmi'")
    physio = subjects.query("dataset == 'physionet'")
    rank_lollipop(
        ax_a, openbmi["mean_gate_gain_pp"].to_numpy(), OPENBMI,
        xlabel="Subjects ranked by gate gain",
        ylabel="Gate gain over fixed baseline (pp)",
        title="OpenBMI · fixed baseline: Traditional", n=54, marker="o",
    )
    rank_lollipop(
        ax_b, physio["mean_gate_gain_pp"].to_numpy(), PHYSIONET,
        xlabel="Subjects ranked by gate gain", ylabel="",
        title="PhysioNet · fixed baseline: F0", n=105, marker="s",
    )
    ax_b.set_yticklabels([])
    panel(ax_a, "a")
    panel(ax_b, "b")

    y = np.array([1, 0])
    datasets = ["OpenBMI", "PhysioNet"]
    colors = [OPENBMI, PHYSIONET]
    labels = ["OpenBMI\nvs Traditional", "PhysioNet\nvs F0"]
    for yi, dataset, color in zip(y, datasets, colors):
        row = stats.loc[dataset]
        mean = row["mean_difference_pp"]
        lo = row["bca95_low_pp"]
        hi = row["bca95_high_pp"]
        ax_c.plot([lo, hi], [yi, yi], color=color, lw=1.8)
        ax_c.plot(mean, yi, "D", ms=5.2, mfc=WHITE, mec=color, mew=1.2)
        ax_c.text(hi + 0.20, yi,
                  f"{mean:+.3f} [{lo:.3f}, {hi:.3f}]",
                  ha="left", va="center", fontsize=5.4, color=color)
    ax_c.axvline(0, color=INK, ls="--", lw=0.8)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(labels)
    ax_c.set_xlim(-1.0, 6.5)
    ax_c.set_ylim(-0.7, 1.7)
    ax_c.set_xlabel("Mean gate gain (pp), 95% BCa CI")
    ax_c.set_title("Locked mean effects", loc="left", fontweight="bold", pad=4)
    ax_c.text(0.02, 0.02, "Stage 7 locked intervals",
              transform=ax_c.transAxes, fontsize=5.2, color=MID,
              ha="left", va="bottom")
    clean(ax_c)
    panel(ax_c, "c", x=-0.15)
    save(fig, "Fig02_gate_gain")


def symmetric_offsets(n: int, width: float = 3.8) -> np.ndarray:
    if n <= 1:
        return np.zeros(n)
    seq = [0]
    for idx in range(1, n):
        mag = (idx + 1) // 2
        seq.append(mag if idx % 2 else -mag)
    return np.asarray(seq, dtype=float) * width / max(abs(x) for x in seq)


def split_panel(ax, splits: pd.DataFrame) -> None:
    for dataset, color, marker, label in [
        ("openbmi", OPENBMI, "o", "OpenBMI"),
        ("physionet", PHYSIONET, "s", "PhysioNet"),
    ]:
        g = splits[splits.dataset == dataset].sort_values("split_index")
        x = g["split_index"].to_numpy()
        y = g["mean_gate_gain_pp"].to_numpy()
        ax.plot(x, y, color=color, marker=marker, ms=4.8,
                mfc=WHITE, mec=color, mew=1.0, label=label)
        for row in g.itertuples(index=False):
            ax.text(row.split_index, row.mean_gate_gain_pp + 0.15,
                    f"{int(row.improved_vs_global_best)}/{int(row.worsened_vs_global_best)}",
                    ha="center", va="bottom", fontsize=4.9, color=color)
        ax.text(x[-1] + 0.12, y[-1], label, color=color,
                fontsize=5.7, ha="left", va="center", fontweight="bold")
    ax.axhline(0, color=INK, ls="--", lw=0.8)
    ax.set_xlim(0.7, 5.75)
    ax.set_ylim(-0.25, 4.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Predefined split index")
    ax.set_ylabel("Mean gate gain (pp)")
    ax.text(0.02, 0.02,
            "Labels are improved/worsened subject counts.\nThe same subjects recur in all five splits.",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.1, color=MID)
    clean(ax)
    panel(ax, "a")


def stability_scatter_panel(ax, subjects: pd.DataFrame) -> None:
    for dataset, color, marker, label in [
        ("openbmi", OPENBMI, "o", "OpenBMI"),
        ("physionet", PHYSIONET, "s", "PhysioNet"),
    ]:
        group = subjects[subjects.dataset == dataset]
        for freq, bin_group in group.groupby("positive_gain_splits_percent"):
            ordered = bin_group.sort_values(["mean_gate_gain_pp", "subject"])
            offsets = symmetric_offsets(len(ordered))
            ax.scatter(
                np.full(len(ordered), freq) + offsets,
                ordered["mean_gate_gain_pp"].to_numpy(),
                s=14, marker=marker, facecolor=WHITE, edgecolor=color,
                linewidth=0.65, alpha=0.90,
                label=label if freq == group["positive_gain_splits_percent"].min() else None,
            )
    for x in [0, 20, 40, 60, 80, 100]:
        ax.axvline(x, color=LIGHT, lw=0.45, zorder=0)
    ax.axhline(0, color=INK, ls="--", lw=0.8)
    ax.set_xlim(-7, 107)
    ax.set_ylim(-28, 38)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Splits with positive gain (%)")
    ax.set_ylabel("Five-split mean gate gain (pp)")
    ax.legend(loc="upper left", ncol=2)
    ax.text(0.02, 0.04,
            "0% is observed data, not missingness:\n"
            "OpenBMI 24 = 9 unchanged + 15 negative;\n"
            "PhysioNet 86 = 74 unchanged + 12 negative.",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.0, color=MID)
    ax.text(0.98, 0.04, "No fitted trend or correlation test",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.0, color=MID)
    clean(ax)
    panel(ax, "b")


def stability_count_panel(ax, subjects: pd.DataFrame) -> None:
    freqs = np.array([0, 20, 40, 60, 80, 100], dtype=float)
    width = 7.0
    for dataset, color, shift, label in [
        ("openbmi", OPENBMI, -4.0, "OpenBMI"),
        ("physionet", PHYSIONET, 4.0, "PhysioNet"),
    ]:
        g = subjects[subjects.dataset == dataset]
        counts = np.array([(g.positive_gain_splits_percent == f).sum() for f in freqs])
        bars = ax.bar(freqs + shift, counts, width=width, facecolor=WHITE,
                      edgecolor=color, lw=1.0, label=label)
        for bar, count in zip(bars, counts):
            if count:
                ax.text(bar.get_x() + bar.get_width()/2, count + 0.9, str(int(count)),
                        ha="center", va="bottom", fontsize=5.0, color=color)
        zero = g[g.positive_gain_splits_percent == 0]
        unchanged = int(np.isclose(zero.mean_gate_gain_pp, 0).sum())
        negative = int((zero.mean_gate_gain_pp < 0).sum())
        ax.bar(0 + shift, unchanged, width=width, color=LIGHT,
               edgecolor=color, lw=0.8)
        ax.bar(0 + shift, negative, bottom=unchanged, width=width,
               color=color, alpha=0.75, edgecolor=color, lw=0.8)
    ax.set_xlim(-12, 112)
    ax.set_ylim(0, 94)
    ax.set_xticks(freqs)
    ax.set_xlabel("Splits with positive gain (%)")
    ax.set_ylabel("Number of subjects")
    ax.legend(loc="upper right", ncol=2)
    ax.text(0.02, 0.96,
            "At 0%: light segment = unchanged in all splits;\n"
            "dark segment = negative five-split mean.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.0, color=MID)
    clean(ax)
    panel(ax, "b")


def render_fig03(option: str) -> None:
    subjects = read("Fig02_subject_gate_gain.csv")
    splits = read("Fig02_split_gate_gain.csv")
    figure_id = (
        "Fig03_optionA_stability_scatter"
        if option == "scatter"
        else "Fig03_optionB_stability_counts"
    )
    fig = plt.figure(figsize=(183 / 25.4, 82 / 25.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.91)
    split_panel(ax_a, splits)
    if option == "scatter":
        stability_scatter_panel(ax_b, subjects)
    else:
        stability_count_panel(ax_b, subjects)
    save(fig, figure_id)


def render_fig04() -> None:
    spectral = read("Fig03_fixed_spectral_summary.csv")
    occlusion = read("Fig03_frequency_occlusion_summary.csv")
    ea = read("Fig03_ea_summary.csv")

    fig = plt.figure(figsize=(183 / 25.4, 105 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.43)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.16, top=0.92)

    order = ["F0", "0.5-4", "4-8", "8-13", "13-30", "30-40",
             "4-40", "0.5-13", "0.5-30", "40-100"]
    spectral = spectral.set_index("strategy_or_band_hz").loc[order].reset_index()
    labels = ["F0", "δ 0.5–4", "θ 4–8", "α 8–13", "β 13–30",
              "γ 30–40", "Traditional 4–40", "0.5–13", "0.5–30", "40–100"]
    y = np.arange(len(spectral))[::-1]
    for yi, row, label in zip(y, spectral.itertuples(index=False), labels):
        color = F0 if row.role == "wideband" else TRADITIONAL if row.role == "traditional" else MID
        marker = "o" if row.role == "wideband" else "s" if row.role == "traditional" else "^"
        ax_a.hlines(yi, 35, row.accuracy_percent, color=LIGHT, lw=0.8)
        ax_a.plot(row.accuracy_percent, yi, marker=marker, ms=4.8,
                  mfc=WHITE, mec=color, mew=1.0)
        ax_a.text(row.accuracy_percent + 0.45, yi, f"{row.accuracy_percent:.2f}",
                  fontsize=5.1, color=color, va="center")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlim(35, 74)
    ax_a.set_xlabel("LOSO mean accuracy (%)")
    ax_a.text(0.98, 0.02, "Summary values; no subject points or CI",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=4.9, color=MID)
    clean(ax_a)
    panel(ax_a, "a", x=-0.16)

    band_order = ["delta 0.5-4", "theta 4-8", "alpha 8-13",
                  "beta 13-30", "gamma 30-40", "high 40-100"]
    occlusion = occlusion.set_index("occluded_band_hz").loc[band_order].reset_index()
    band_labels = ["δ 0.5–4", "θ 4–8", "α 8–13",
                   "β 13–30", "γ 30–40", "High 40–100"]
    yb = np.arange(len(occlusion))[::-1]
    for yi, row, label in zip(yb, occlusion.itertuples(index=False), band_labels):
        color = F0 if label.startswith("α") else MID
        ax_b.errorbar(row.mean_accuracy_drop_pp, yi,
                      xerr=row.sd_accuracy_drop_pp,
                      fmt="o", ms=4.5, mfc=WHITE, mec=color, mew=1.0,
                      ecolor=color, elinewidth=1.0, capsize=2.3)
        ax_b.text(row.mean_accuracy_drop_pp + row.sd_accuracy_drop_pp + 0.6,
                  yi, f"{row.mean_accuracy_drop_pp:.2f}±{row.sd_accuracy_drop_pp:.2f}",
                  fontsize=4.8, color=color, va="center")
    ax_b.axvline(0, color=INK, ls="--", lw=0.8)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels(band_labels)
    ax_b.set_xlim(-3, 37)
    ax_b.set_xlabel("Accuracy decrease after occlusion (pp)")
    ax_b.text(0.98, 0.02, "Segments are existing SD, not CI",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=4.9, color=MID)
    clean(ax_b)
    panel(ax_b, "b", x=-0.18)

    configs = ea["configuration"].tolist()
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    for idx, row in enumerate(ea.itertuples(index=False)):
        ax_c.plot([0, 1], [row.no_ea_accuracy_percent, row.ea_accuracy_percent],
                  color=EA, lw=1.2, ls=line_styles[idx],
                  marker=markers[idx], ms=4.0, mfc=WHITE, mec=EA)
        ax_c.text(-0.06, row.no_ea_accuracy_percent, configs[idx],
                  ha="right", va="center", fontsize=4.9, color=MID)
        ax_c.text(1.05, row.ea_accuracy_percent,
                  f"+{row.ea_minus_no_ea_pp:.2f}",
                  ha="left", va="center", fontsize=4.9, color=EA)
    ax_c.set_xlim(-0.32, 1.30)
    ax_c.set_ylim(61.8, 69.6)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["No EA", "EA"])
    ax_c.set_ylabel("LOSO mean accuracy (%)")
    ax_c.text(0.02, 0.02,
              "Lines join matched configurations;\nnot uncertainty intervals.",
              transform=ax_c.transAxes, ha="left", va="bottom",
              fontsize=4.9, color=MID)
    clean(ax_c)
    panel(ax_c, "c", x=-0.18)
    save(fig, "Fig04_bci_diagnostics")


def render_fig05() -> None:
    bci = read("Fig03_se_bci_summary.csv")
    physio = read("Fig03_se_physionet_subjects.csv")
    openbmi = read("Fig03_se_openbmi_folds.csv")

    fig = plt.figure(figsize=(183 / 25.4, 78 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.25, 1.05], wspace=0.38)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.86)

    vals = bci["accuracy_percent"].to_numpy()
    ax_a.plot([0, 1], vals, color=SE, lw=1.3, marker="o", ms=5,
              mfc=WHITE, mec=SE)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["F0", "SE"])
    ax_a.set_xlim(-0.4, 1.4)
    ax_a.set_ylim(68.8, 72.0)
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_title("BCI IV-2a · summary level", loc="left",
                   fontsize=6.8, fontweight="bold", color=SE)
    ax_a.text(0.5, 71.70, "+0.81 pp; p≈.30",
              ha="center", va="top", fontsize=5.2, color=SE)
    ax_a.text(0.5, 68.93, "small positive observation",
              ha="center", va="bottom", fontsize=4.8, color=MID)
    clean(ax_a)
    panel(ax_a, "a", x=-0.16)

    deltas = physio["delta_se_minus_f0_pp"].to_numpy()
    offsets = np.asarray([
        ((sum((idx + 1) * ord(char) for idx, char in enumerate(subject)) % 97) / 96 - 0.5) * 0.34
        for subject in physio["subject"].astype(str)
    ])
    ax_b.scatter(deltas, offsets, s=13, marker="o",
                 facecolor=WHITE, edgecolor=SE, linewidth=0.65)
    mean = float(deltas.mean())
    ax_b.axvline(0, color=INK, ls="--", lw=0.8)
    ax_b.axvline(mean, color=SE, lw=1.5)
    ax_b.plot(mean, 0.25, "D", ms=4.8, mfc=WHITE, mec=SE, mew=1.0)
    ax_b.set_xlim(-18, 12)
    ax_b.set_ylim(-0.28, 0.36)
    ax_b.set_yticks([])
    ax_b.set_xlabel("SE − F0 accuracy (pp)")
    ax_b.set_title("PhysioNet · subject level (n=105)", loc="left",
                   fontsize=6.8, fontweight="bold", color=SE)
    ax_b.text(0.02, 0.94, f"mean {mean:+.3f} pp",
              transform=ax_b.transAxes, ha="left", va="top",
              fontsize=5.2, color=SE)
    clean(ax_b)
    panel(ax_b, "b", x=-0.12)

    seed_styles = {
        42: (OPENBMI, "-", "Seed 42"),
        43: ("#71889A", "--", "Seed 43"),
    }
    for seed, group in openbmi.groupby("seed", sort=True):
        color, ls, label = seed_styles[int(seed)]
        for idx, row in enumerate(group.itertuples(index=False)):
            ax_c.plot([0, 1], [row.f0_accuracy_percent, row.se_accuracy_percent],
                      color=color, ls=ls, lw=1.0, marker="o", ms=3.4,
                      mfc=WHITE, mec=color,
                      label=label if idx == 0 else None)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["F0", "SE"])
    ax_c.set_xlim(-0.32, 1.32)
    ax_c.set_ylim(70, 83)
    ax_c.set_ylabel("Fold accuracy (%)")
    ax_c.set_title("OpenBMI · partial completed logs", loc="left",
                   fontsize=6.8, fontweight="bold", color=OPENBMI)
    ax_c.legend(loc="lower right", ncol=2, fontsize=5.2)
    ax_c.text(0.5, 82.4, "2 seeds / 6 folds; fold-level only\n6/6 negative; mean −3.615 pp",
              ha="center", va="top", fontsize=5.0, color=OPENBMI)
    clean(ax_c)
    panel(ax_c, "c", x=-0.12)
    save(fig, "Fig05_se_transfer")


def render_fig06() -> None:
    data = read("Fig04_fixed_strategy_subjects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.90, wspace=0.25)

    for ax, dataset, color, marker, choice, letter in [
        (axes[0], "OpenBMI", OPENBMI, "o", "Traditional", "a"),
        (axes[1], "PhysioNet", PHYSIONET, "s", "F0", "b"),
    ]:
        g = data[data.dataset == dataset]
        ax.scatter(g.traditional_accuracy_percent, g.f0_accuracy_percent,
                   s=17, marker=marker, facecolor=WHITE, edgecolor=color,
                   linewidth=0.7)
        ax.plot([20, 100], [20, 100], color=INK, ls="--", lw=0.8)
        ax.set_xlim(20, 100)
        ax.set_ylim(20, 100)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Traditional accuracy (%)")
        ax.set_ylabel("F0 accuracy (%)" if letter == "a" else "")
        ax.set_title(f"{dataset} · global fixed choice: {choice}",
                     loc="left", color=color, fontweight="bold", pad=4)
        ax.text(0.96, 0.88, "F0 higher", transform=ax.transAxes,
                ha="right", color=F0, fontsize=5.3)
        ax.text(0.04, 0.08, "Traditional higher", transform=ax.transAxes,
                ha="left", color=TRADITIONAL, fontsize=5.3)
        ax.text(0.96, 0.04, f"n={len(g)}", transform=ax.transAxes,
                ha="right", va="bottom", color=MID, fontsize=5.2)
        clean(ax)
        panel(ax, letter)
    save(fig, "Fig06_fixed_strategy_scatter")


def render_fig07() -> None:
    data = read("Fig04_fixed_strategy_subjects.csv")
    stats = locked_stats("Fig04")
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 82 / 25.4))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.90, wspace=0.26)

    for ax, dataset, color, marker, letter in [
        (axes[0], "OpenBMI", OPENBMI, "o", "a"),
        (axes[1], "PhysioNet", PHYSIONET, "s", "b"),
    ]:
        g = data[data.dataset == dataset].sort_values(
            ["f0_minus_traditional_pp", "subject"]
        )
        vals = g.f0_minus_traditional_pp.to_numpy()
        x = np.arange(1, len(vals) + 1)
        ax.vlines(x, 0, vals, color=color, alpha=0.24, lw=0.55)
        ax.scatter(x, vals, s=13, marker=marker, facecolor=WHITE,
                   edgecolor=color, linewidth=0.7)
        ax.axhline(0, color=INK, ls="--", lw=0.8)
        mean_x = len(vals) + max(7, len(vals) * 0.10)
        row = stats.loc[dataset]
        mean, lo, hi = (
            row.mean_difference_pp,
            row.bca95_low_pp,
            row.bca95_high_pp,
        )
        ax.axvline(len(vals) + 2.5, color=LIGHT, lw=0.7)
        ax.errorbar([mean_x], [mean],
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="D", ms=5.0, mfc=WHITE, mec=color, mew=1.1,
                    ecolor=color, elinewidth=1.5, capsize=2.8)
        ax.set_xlim(0, mean_x + 4)
        ax.set_ylim(-50, 50)
        ax.set_xticks([1, len(vals), mean_x])
        ax.set_xticklabels(["1", str(len(vals)), "Mean"])
        ax.set_xlabel("Subjects ranked by F0 − Traditional")
        ax.set_ylabel("F0 − Traditional accuracy (pp)" if letter == "a" else "")
        ax.set_title(dataset, loc="left", color=color,
                     fontweight="bold", pad=4)
        ax.text(0.02, 0.97,
                f"mean {mean:+.3f} pp\n95% BCa CI [{lo:.3f}, {hi:.3f}]",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=5.3, color=color)
        clean(ax)
        panel(ax, letter)
    save(fig, "Fig07_fixed_strategy_difference")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    render_fig02()
    render_fig03("scatter")
    render_fig03("counts")
    render_fig04()
    render_fig05()
    render_fig06()
    render_fig07()
    print(json.dumps({"generated": list(SIZES_MM)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
