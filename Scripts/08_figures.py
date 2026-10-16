#!/usr/bin/env python3
"""
Stage 08 Figures: publication/poster-ready visualizations for baseline results.

Recommended inputs:
- Stage 08A summary metrics
- Stage 08.6 summary metrics
- Stage 08.6 out-of-fold predictions
- Stage 08.6 fold metrics
- Stage 08.6 coefficient summary

The script creates:
1. Baseline performance ladder
2. Feature-set performance with bootstrap confidence intervals
3. ROC curves from pooled out-of-fold predictions
4. Precision-recall curves from pooled out-of-fold predictions
5. Full-vs-non-TIM robustness comparison
6. Fold-level performance stability
7. Coefficient stability for the best interpretable combined model

No new modeling is performed.

Example:
    python3 scripts/08_figures.py \
      --stage08a-summary results/stage08A/stage08A_summary_metrics.csv \
      --stage086-summary results/stage08_6/stage08_6_summary_metrics.csv \
      --stage086-oof results/stage08_6/stage08_6_oof_predictions.csv \
      --stage086-fold-metrics results/stage08_6/stage08_6_fold_metrics.csv \
      --stage086-coefficients results/stage08_6/stage08_6_coefficient_summary.csv \
      --output-dir figures/stage08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


PRIMARY_FOLD = "homology_cv_fold"

DISPLAY_NAMES = {
    "majority": "Majority",
    "length_only": "Length only",
    "contact_order_only": "Contact order only",
    "sequence_final": "Sequence",
    "structure_minimal_final": "Structure: minimal",
    "structure_core_final": "Structure: core",
    "sequence_plus_structure_minimal": "Sequence + minimal structure",
    "sequence_plus_structure_core": "Sequence + core structure",
}

LADDER_ORDER = [
    "majority",
    "length_only",
    "contact_order_only",
    "sequence_final",
    "structure_minimal_final",
    "sequence_plus_structure_minimal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready figures from Stage 08 results."
    )
    parser.add_argument("--stage08a-summary", required=True)
    parser.add_argument("--stage086-summary", required=True)
    parser.add_argument("--stage086-oof", default=None)
    parser.add_argument("--stage086-fold-metrics", default=None)
    parser.add_argument("--stage086-coefficients", default=None)
    parser.add_argument("--output-dir", default="figures/stage08")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--top-coefficients", type=int, default=12)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def require_file(path_text: str, label: str) -> Path:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def filter_primary_full(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["subset"] == "full")
        & (df["fold_column"] == PRIMARY_FOLD)
    ].copy()


def plot_baseline_ladder(
    stage08a: pd.DataFrame,
    stage086: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    a = filter_primary_full(stage08a).rename(columns={"model": "model_key"})
    b = filter_primary_full(stage086).rename(columns={"feature_set": "model_key"})

    combined = pd.concat(
        [
            a[["model_key", "mcc", "mcc_ci_low", "mcc_ci_high"]],
            b[["model_key", "mcc", "mcc_ci_low", "mcc_ci_high"]],
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates("model_key", keep="last")
    combined = combined[combined["model_key"].isin(LADDER_ORDER)].copy()
    combined["order"] = combined["model_key"].map(
        {name: i for i, name in enumerate(LADDER_ORDER)}
    )
    combined = combined.sort_values("order")
    combined["display_name"] = combined["model_key"].map(DISPLAY_NAMES)

    lower = combined["mcc"] - combined["mcc_ci_low"]
    upper = combined["mcc_ci_high"] - combined["mcc"]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(combined["display_name"], combined["mcc"])
    axis.errorbar(
        np.arange(len(combined)),
        combined["mcc"],
        yerr=np.vstack([lower, upper]),
        fmt="none",
        capsize=4,
    )
    axis.axhline(0, linewidth=1)
    axis.set_ylabel("Matthews correlation coefficient")
    axis.set_title("Progression from simple baselines to integrated features")
    axis.tick_params(axis="x", rotation=30)
    save_figure(fig, output_dir, "figure_01_baseline_performance_ladder", dpi)


def plot_feature_set_metrics(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    data = filter_primary_full(summary)
    preferred = [
        "sequence_final",
        "structure_minimal_final",
        "structure_core_final",
        "sequence_plus_structure_minimal",
        "sequence_plus_structure_core",
    ]
    data = data[data["feature_set"].isin(preferred)].copy()
    data["order"] = data["feature_set"].map(
        {name: i for i, name in enumerate(preferred)}
    )
    data = data.sort_values("order")
    data["display_name"] = data["feature_set"].map(DISPLAY_NAMES)

    metrics = [
        ("mcc", "MCC"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for axis, (metric, title) in zip(axes, metrics):
        lower = data[metric] - data[f"{metric}_ci_low"]
        upper = data[f"{metric}_ci_high"] - data[metric]
        positions = np.arange(len(data))

        axis.bar(positions, data[metric])
        axis.errorbar(
            positions,
            data[metric],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            capsize=4,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(data["display_name"], rotation=45, ha="right")
        axis.set_title(title)
        axis.set_ylabel(title)
        if metric != "mcc":
            axis.set_ylim(0, 1)

    fig.suptitle(
        "Grouped out-of-fold performance of frozen Feature Set v2",
        y=1.02,
    )
    save_figure(fig, output_dir, "figure_02_feature_set_performance", dpi)


def plot_oof_curves(
    oof: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    data = oof[
        (oof["subset"] == "full")
        & (oof["fold_column"] == PRIMARY_FOLD)
    ].copy()

    preferred = [
        "sequence_final",
        "structure_minimal_final",
        "structure_core_final",
        "sequence_plus_structure_minimal",
        "sequence_plus_structure_core",
    ]

    fig, axis = plt.subplots(figsize=(7, 6))
    for feature_set in preferred:
        group = data[data["feature_set"] == feature_set]
        if group.empty:
            continue
        y = group["binary_label"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y, probability)
        axis.plot(fpr, tpr, label=DISPLAY_NAMES[feature_set])

    axis.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("ROC curves from pooled out-of-fold predictions")
    axis.legend(fontsize=8)
    save_figure(fig, output_dir, "figure_03_oof_roc_curves", dpi)

    fig, axis = plt.subplots(figsize=(7, 6))
    prevalence = float(data["binary_label"].mean())
    for feature_set in preferred:
        group = data[data["feature_set"] == feature_set]
        if group.empty:
            continue
        y = group["binary_label"].to_numpy(dtype=int)
        probability = group["predicted_probability"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, probability)
        axis.plot(recall, precision, label=DISPLAY_NAMES[feature_set])

    axis.axhline(prevalence, linestyle="--", label="Positive prevalence")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves from out-of-fold predictions")
    axis.legend(fontsize=8)
    save_figure(fig, output_dir, "figure_04_oof_precision_recall_curves", dpi)


def plot_non_tim_robustness(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    preferred = [
        "sequence_final",
        "structure_minimal_final",
        "sequence_plus_structure_minimal",
    ]
    data = summary[
        (summary["fold_column"] == PRIMARY_FOLD)
        & (summary["subset"].isin(["full", "non_tim"]))
        & (summary["feature_set"].isin(preferred))
    ].copy()

    data["display_name"] = data["feature_set"].map(DISPLAY_NAMES)
    pivot = data.pivot(
        index="display_name",
        columns="subset",
        values="roc_auc",
    ).reindex([DISPLAY_NAMES[x] for x in preferred])

    fig, axis = plt.subplots(figsize=(9, 5.5))
    pivot.plot(kind="bar", ax=axis)
    axis.set_ylabel("ROC-AUC")
    axis.set_ylim(0, 1)
    axis.set_xlabel("")
    axis.set_title("Performance persists after removal of TIM-barrel proteins")
    axis.tick_params(axis="x", rotation=25)
    axis.legend(title="Dataset")
    save_figure(fig, output_dir, "figure_05_non_tim_robustness", dpi)


def plot_fold_stability(
    fold_metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    preferred = [
        "sequence_final",
        "structure_minimal_final",
        "sequence_plus_structure_minimal",
    ]
    data = fold_metrics[
        (fold_metrics["subset"] == "full")
        & (fold_metrics["fold_column"] == PRIMARY_FOLD)
        & (fold_metrics["feature_set"].isin(preferred))
    ].copy()
    if data.empty:
        return

    data["display_name"] = data["feature_set"].map(DISPLAY_NAMES)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for name, group in data.groupby("display_name"):
        group = group.sort_values("fold")
        axis.plot(
            group["fold"],
            group["mcc"],
            marker="o",
            label=name,
        )

    axis.axhline(0, linewidth=1)
    axis.set_xlabel("Homology-grouped outer fold")
    axis.set_ylabel("Fold-level MCC")
    axis.set_title("Performance stability across grouped folds")
    axis.set_xticks(sorted(data["fold"].unique()))
    axis.legend(fontsize=8)
    save_figure(fig, output_dir, "figure_06_fold_stability", dpi)


def plot_coefficient_stability(
    coefficient_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    top_n: int,
) -> None:
    data = coefficient_summary[
        (coefficient_summary["subset"] == "full")
        & (coefficient_summary["fold_column"] == PRIMARY_FOLD)
        & (
            coefficient_summary["feature_set"]
            == "sequence_plus_structure_minimal"
        )
    ].copy()
    if data.empty:
        return

    data["absolute_mean"] = data["mean"].abs()
    data = data.nlargest(top_n, "absolute_mean").sort_values("mean")

    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(data["feature"], data["mean"])
    axis.errorbar(
        data["mean"],
        np.arange(len(data)),
        xerr=data["std"].fillna(0),
        fmt="none",
        capsize=3,
    )
    axis.axvline(0, linewidth=1)
    axis.set_xlabel("Mean standardized logistic coefficient ± fold SD")
    axis.set_title(
        "Most influential coefficients in sequence + minimal structure model"
    )
    save_figure(fig, output_dir, "figure_07_coefficient_stability", dpi)


def write_figure_manifest(output_dir: Path) -> None:
    manifest = {
        "figure_01_baseline_performance_ladder": {
            "recommended_use": "Poster/paper main figure",
            "message": (
                "Shows the progression from majority and length baselines to "
                "contact order, sequence, structure, and integrated prediction."
            ),
        },
        "figure_02_feature_set_performance": {
            "recommended_use": "Paper main figure or results panel",
            "message": (
                "Directly compares frozen feature groups with bootstrap confidence intervals."
            ),
        },
        "figure_03_oof_roc_curves": {
            "recommended_use": "Paper main or supplementary figure",
            "message": "Threshold-independent discrimination from pooled OOF probabilities.",
        },
        "figure_04_oof_precision_recall_curves": {
            "recommended_use": "Paper main or supplementary figure",
            "message": "Emphasizes positive-class performance under class imbalance.",
        },
        "figure_05_non_tim_robustness": {
            "recommended_use": "Poster/paper main figure",
            "message": (
                "Demonstrates that performance is not explained only by TIM-barrel enrichment."
            ),
        },
        "figure_06_fold_stability": {
            "recommended_use": "Supplementary figure",
            "message": "Shows whether aggregate performance is stable across grouped folds.",
        },
        "figure_07_coefficient_stability": {
            "recommended_use": "Supplementary/interpretable baseline figure",
            "message": (
                "Displays the strongest linear effects and their variability across folds."
            ),
        },
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage08a = pd.read_csv(
        require_file(args.stage08a_summary, "Stage 08A summary")
    )
    stage086 = pd.read_csv(
        require_file(args.stage086_summary, "Stage 08.6 summary")
    )

    plot_baseline_ladder(stage08a, stage086, output_dir, args.dpi)
    plot_feature_set_metrics(stage086, output_dir, args.dpi)
    plot_non_tim_robustness(stage086, output_dir, args.dpi)

    if args.stage086_oof:
        path = Path(args.stage086_oof)
        if path.exists():
            plot_oof_curves(pd.read_csv(path), output_dir, args.dpi)
        else:
            print(f"WARNING: OOF file not found, skipping curves: {path}")

    if args.stage086_fold_metrics:
        path = Path(args.stage086_fold_metrics)
        if path.exists():
            plot_fold_stability(pd.read_csv(path), output_dir, args.dpi)
        else:
            print(f"WARNING: fold metrics not found, skipping: {path}")

    if args.stage086_coefficients:
        path = Path(args.stage086_coefficients)
        if path.exists():
            plot_coefficient_stability(
                pd.read_csv(path),
                output_dir,
                args.dpi,
                args.top_coefficients,
            )
        else:
            print(f"WARNING: coefficient file not found, skipping: {path}")

    write_figure_manifest(output_dir)

    print("Stage 08 figure generation complete")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
