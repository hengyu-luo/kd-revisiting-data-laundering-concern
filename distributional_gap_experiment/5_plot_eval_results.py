#!/usr/bin/env python3

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

MODEL_TAGS = [
    "clean_teacher_checkpoint",
    "dirty_teacher_checkpoint",
    "supervised_on_clean_data",
    "supervised_on_contaminated_data",
    "student_kd_from_clean_soft",
    "student_kd_from_dirty_soft",
]

PAIR_KEY_TO_MODEL_TAGS = {
    "teacher_dirty_vs_clean": ("clean_teacher_checkpoint", "dirty_teacher_checkpoint"),
    "student_supervised_dirty_vs_clean": ("supervised_on_clean_data", "supervised_on_contaminated_data"),
    "student_kd_dirty_vs_clean": ("student_kd_from_clean_soft", "student_kd_from_dirty_soft"),
}

PAIR_KEY_TO_TITLE = {
    "teacher_dirty_vs_clean": "Teacher Dirty vs Clean",
    "student_supervised_dirty_vs_clean": "Student Supervised Dirty vs Clean",
    "student_kd_dirty_vs_clean": "Student KD Dirty vs Clean",
}


def stars_from_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.005:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def load_accuracy_from_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    out = {}
    for tag in MODEL_TAGS:
        sub = df[df["model_tag"] == tag].sort_values("sample_index")
        if sub.empty:
            continue
        y_true = sub["ground_truth"].to_numpy()
        y_pred = sub["prediction"].to_numpy()
        out[tag] = float((y_pred == y_true).mean())
    return out


def load_pvalue(eval_json_path: str, pair_key: str):
    if not os.path.exists(eval_json_path):
        return None
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sig = data.get("significance", {}).get(pair_key, {})
    if not isinstance(sig, dict):
        return None
    p = sig.get("max_p_value", sig.get("p_value"))
    if p is None:
        return None
    return float(p)


def collect_summary(task, contamination_mode, teacher_model, student_model, levels, seeds, eval_dir):
    teacher_name = teacher_model.replace("/", "-")
    student_name = student_model.replace("/", "-")
    stub = f"{teacher_name}_to_{student_name}_{contamination_mode}"

    summary = {
        "task": task,
        "contamination_mode": contamination_mode,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "seeds": list(seeds),
        "levels": {},
    }

    for level in levels:
        exp_tag = f"{task}-fixedtest_by_similarity_level_{level}"
        seed_accs = defaultdict(list)
        pair_seed_pvals = {k: {} for k in PAIR_KEY_TO_MODEL_TAGS.keys()}

        for seed in seeds:
            parquet_path = os.path.join(eval_dir, f"DETAILED_EVAL_{exp_tag}_{stub}_seed{seed}.parquet")
            if os.path.exists(parquet_path):
                acc_map = load_accuracy_from_parquet(parquet_path)
                for model_tag, acc in acc_map.items():
                    seed_accs[model_tag].append(acc)

            eval_json_path = os.path.join(eval_dir, f"EVAL_{exp_tag}_{stub}_seed{seed}.json")
            for pair_key in PAIR_KEY_TO_MODEL_TAGS.keys():
                p = load_pvalue(eval_json_path, pair_key)
                if p is not None:
                    pair_seed_pvals[pair_key][str(seed)] = p

        level_key = str(level)
        summary["levels"][level_key] = {
            "model_accuracy": {},
            "significance": {},
        }

        for model_tag in MODEL_TAGS:
            vals = seed_accs.get(model_tag, [])
            if vals:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                mean_v = np.nan
                std_v = np.nan
            summary["levels"][level_key]["model_accuracy"][model_tag] = {"mean": mean_v, "std": std_v}

        for pair_key, seed_map in pair_seed_pvals.items():
            max_p = max(seed_map.values()) if seed_map else None
            summary["levels"][level_key]["significance"][pair_key] = {
                "p_values_by_seed": seed_map,
                "max_p_value": max_p,
                "stars": stars_from_p(max_p) if max_p is not None else "",
            }

    return summary


def plot_performance(summary, out_path):
    task = summary["task"]
    contamination_mode = summary["contamination_mode"]
    levels = sorted(summary["levels"].keys(), key=int)

    teacher = {"clean": [], "dirty": [], "sig": []}
    baseline = {"clean": [], "dirty": [], "sig": []}
    student = {"clean": [], "dirty": [], "sig": []}

    for level in levels:
        level_data = summary["levels"][level]
        acc = level_data["model_accuracy"]
        sig = level_data["significance"]

        teacher["clean"].append(acc["clean_teacher_checkpoint"]["mean"] * 100)
        teacher["dirty"].append(acc["dirty_teacher_checkpoint"]["mean"] * 100)
        teacher["sig"].append(sig["teacher_dirty_vs_clean"].get("stars", ""))

        baseline["clean"].append(acc["supervised_on_clean_data"]["mean"] * 100)
        baseline["dirty"].append(acc["supervised_on_contaminated_data"]["mean"] * 100)
        baseline["sig"].append(sig["student_supervised_dirty_vs_clean"].get("stars", ""))

        student["clean"].append(acc["student_kd_from_clean_soft"]["mean"] * 100)
        student["dirty"].append(acc["student_kd_from_dirty_soft"]["mean"] * 100)
        student["sig"].append(sig["student_kd_dirty_vs_clean"].get("stars", ""))

    teacher_clean, teacher_dirty = np.array(teacher["clean"]), np.array(teacher["dirty"])
    baseline_clean, baseline_dirty = np.array(baseline["clean"]), np.array(baseline["dirty"])
    student_clean, student_dirty = np.array(student["clean"]), np.array(student["dirty"])

    teacher_delta = teacher_dirty - teacher_clean
    baseline_delta = baseline_dirty - baseline_clean
    student_delta = student_dirty - student_clean

    x = np.arange(len(levels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(15, 8))

    colors_clean = ["#a6cee3", "#b2df8a", "#fb9a99"]
    colors_dirty = ["#1f78b4", "#33a02c", "#e31a1c"]

    ax.bar(x - width, teacher_clean, width, color=colors_clean[0])
    ax.bar(x, baseline_clean, width, color=colors_clean[1])
    ax.bar(x + width, student_clean, width, color=colors_clean[2])

    def plot_deltas(x_pos, clean_data, delta_data, color):
        for i in range(len(delta_data)):
            if np.isnan(clean_data[i]) or np.isnan(delta_data[i]):
                continue
            if delta_data[i] >= 0:
                ax.bar(x_pos[i], delta_data[i], width, bottom=clean_data[i], color=color)
            else:
                ax.bar(
                    x_pos[i],
                    -delta_data[i],
                    width,
                    bottom=clean_data[i] + delta_data[i],
                    color="none",
                    edgecolor=color,
                    hatch="///",
                )

    plot_deltas(x - width, teacher_clean, teacher_delta, colors_dirty[0])
    plot_deltas(x, baseline_clean, baseline_delta, colors_dirty[1])
    plot_deltas(x + width, student_clean, student_delta, colors_dirty[2])

    def annotate_deltas(xpos, clean, delta, sig):
        for i in range(len(levels)):
            if np.isnan(clean[i]) or np.isnan(delta[i]):
                continue
            y_pos = clean[i] + delta[i] / 2
            text_color = "white" if delta[i] >= 0 else "black"
            ax.text(
                xpos[i],
                y_pos,
                f"{delta[i]:.1f}{sig[i]}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    annotate_deltas(x - width, teacher_clean, teacher_delta, teacher["sig"])
    annotate_deltas(x, baseline_clean, baseline_delta, baseline["sig"])
    annotate_deltas(x + width, student_clean, student_delta, student["sig"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.set_ylabel("Model Accuracy (%)")
    ax.set_title(
        f'Performance on "{task.replace("_", " ").title()}" Benchmark ({contamination_mode.title()} Contamination)',
        fontsize=16,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    all_values = np.concatenate([teacher_clean, teacher_dirty, baseline_clean, baseline_dirty, student_clean, student_dirty])
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size > 0:
        ax.set_ylim(bottom=float(np.min(finite_values) - 5), top=float(np.max(finite_values) + 5))

    legend_elements = [
        Patch(facecolor=colors_clean[0], label="Teacher (Clean)"),
        Patch(facecolor=colors_dirty[0], label="Teacher (Gain)"),
        Patch(facecolor="white", edgecolor=colors_dirty[0], hatch="///", label="Teacher (Loss)"),
        Patch(facecolor=colors_clean[1], label="Baseline (Clean)"),
        Patch(facecolor=colors_dirty[1], label="Baseline (Gain)"),
        Patch(facecolor="white", edgecolor=colors_dirty[1], hatch="///", label="Baseline (Loss)"),
        Patch(facecolor=colors_clean[2], label="Student (Clean)"),
        Patch(facecolor=colors_dirty[2], label="Student (Gain)"),
        Patch(facecolor="white", edgecolor=colors_dirty[2], hatch="///", label="Student (Loss)"),
    ]
    ax.legend(handles=legend_elements, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated performance figure: {out_path}")


def collect_pvalue_matrix(task, contamination_mode, teacher_model, student_model, levels, seeds, eval_dir, pair_key):
    teacher_name = teacher_model.replace("/", "-")
    student_name = student_model.replace("/", "-")
    stub = f"{teacher_name}_to_{student_name}_{contamination_mode}"

    rows = []
    for level in levels:
        exp_tag = f"{task}-fixedtest_by_similarity_level_{level}"
        row = []
        for seed in seeds:
            eval_json = os.path.join(eval_dir, f"EVAL_{exp_tag}_{stub}_seed{seed}.json")
            p = load_pvalue(eval_json, pair_key)
            row.append(np.nan if p is None else p)
        rows.append(row)

    columns = [f"Seed {i + 1}" for i in range(len(seeds))]
    return pd.DataFrame(rows, index=levels, columns=columns)


def plot_significance_heatmap(df: pd.DataFrame, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_df = df.copy()
    annot_df = plot_df.copy().applymap(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
    plot_df = plot_df.fillna(1.0)

    colors = ["#084594", "#2171b5", "#6baed6", "#c6dbef"]
    boundaries = [0, 0.001, 0.01, 0.05, 1.01]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        plot_df,
        ax=ax,
        annot=annot_df,
        fmt="",
        linewidths=0.5,
        cmap=cmap,
        norm=norm,
        cbar_kws={"ticks": [0.0005, 0.0055, 0.03, 0.53]},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.set_yticklabels(["p <= 0.001 (***)", "p <= 0.01 (**)", "p <= 0.05 (*)", "p > 0.05 (ns)"])

    ax.set_ylabel("Train-Test Distributional Gap (Level)")
    ax.set_xlabel("Random Seed")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated significance heatmap: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate final evaluation figures (accuracy bars + significance heatmaps).")
    parser.add_argument("--tasks", nargs="*", default=["rotten_tomatoes", "emotion"])
    parser.add_argument("--contamination_modes", nargs="*", default=["replace"])
    parser.add_argument("--teacher_model", default="bert-base-uncased")
    parser.add_argument("--student_model", default="distilbert-base-uncased")
    parser.add_argument("--levels", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 42, 86, 358, 1024])
    parser.add_argument("--pair_keys", nargs="*", default=["student_kd_dirty_vs_clean"])
    parser.add_argument("--eval_dir", default="results/metrics/eval_results")
    parser.add_argument("--out_dir", default="results/plots/target_figures")
    parser.add_argument("--save_summary_json", action="store_true")
    args = parser.parse_args()

    teacher_name = args.teacher_model.replace("/", "-")
    student_name = args.student_model.replace("/", "-")
    os.makedirs(args.out_dir, exist_ok=True)

    for task in args.tasks:
        for mode in args.contamination_modes:
            summary = collect_summary(
                task=task,
                contamination_mode=mode,
                teacher_model=args.teacher_model,
                student_model=args.student_model,
                levels=args.levels,
                seeds=args.seeds,
                eval_dir=args.eval_dir,
            )

            if args.save_summary_json:
                summary_path = os.path.join(args.out_dir, f"{task}_{mode}_{teacher_name}_to_{student_name}_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                print(f"Saved summary JSON: {summary_path}")

            perf_out = os.path.join(args.out_dir, f"{task}_{mode}_{teacher_name}_to_{student_name}_performance_comparison.pdf")
            plot_performance(summary, perf_out)

            for pair_key in args.pair_keys:
                df = collect_pvalue_matrix(
                    task=task,
                    contamination_mode=mode,
                    teacher_model=args.teacher_model,
                    student_model=args.student_model,
                    levels=args.levels,
                    seeds=args.seeds,
                    eval_dir=args.eval_dir,
                    pair_key=pair_key,
                )
                title_key = PAIR_KEY_TO_TITLE.get(pair_key, pair_key)
                title = f"{task} ({mode}) - {title_key}"
                heatmap_out = os.path.join(
                    args.out_dir,
                    f"{task}_{mode}_{teacher_name}_to_{student_name}_{pair_key}_significance_heatmap.pdf",
                )
                plot_significance_heatmap(df, title, heatmap_out)


if __name__ == "__main__":
    main()
