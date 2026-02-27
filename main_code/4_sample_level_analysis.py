#!/usr/bin/env python3
"""
Sample-level analysis (raw difficulty only).

This script:
1. Computes raw laundering/contamination scores.
2. Computes baseline difficulty.
3. Plots one trend figure sorted by baseline difficulty.
"""
import os
import argparse
import logging
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not installed. Plot generation will be disabled.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
EVAL_RESULTS_DIR = "./results/metrics/eval_results"


def generate_difficulty_trend_plot(df, file_path):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping plot.")
        return

    sorted_df = df.sort_values('difficulty_baseline').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.scatter(sorted_df.index, sorted_df['laundering_score_raw'], label='Laundering Score (Raw)', alpha=0.5, s=10, color='blue')
    ax.scatter(sorted_df.index, sorted_df['contamination_score_raw'], label='Contamination Score (Raw)', alpha=0.5, s=10, color='red')
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Samples (Sorted by Baseline Difficulty)')
    ax.set_ylabel('Effect Score (Raw)')
    ax.set_title('Raw Laundering/Contamination Scores by Baseline Difficulty')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)

    ax2 = ax.twinx()
    ax2.plot(sorted_df.index, sorted_df['difficulty_baseline'], color='black', linestyle=':', linewidth=2, label='Baseline Difficulty')
    ax2.set_ylabel('Baseline Difficulty')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.close()
    logger.info(f"Saved difficulty trend plot to {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample-level analysis (raw difficulty only).")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--teacher_model', type=str, required=True)
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--contamination_mode', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train_subset_ratio', type=float, required=True)
    parser.add_argument('--distillation_method', type=str, default='soft_fwd')
    parser.add_argument('--output_dir', type=str, default='./results/consolidated_analysis/complexity_groups')
    args = parser.parse_args()

    teacher_name = args.teacher_model.replace("/", "-")
    student_name = args.student_model.replace("/", "-")
    base_fname = f"{args.task}_{args.train_subset_ratio}_{teacher_name}_to_{student_name}_{args.contamination_mode}_seed{args.seed}"
    parquet_path = os.path.join(EVAL_RESULTS_DIR, f"DETAILED_EVAL_{base_fname}.parquet")
    if not os.path.exists(parquet_path):
        logger.error(f"FATAL: Detailed results file not found at: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    output_path = os.path.join(args.output_dir, args.task)
    os.makedirs(output_path, exist_ok=True)

    pivot_df = df.pivot_table(index='sample_index', columns='model_tag', values='ground_truth_prob')
    all_tags = df['model_tag'].unique()

    clean_baseline_tag = next((tag for tag in all_tags if 'supervised_on_clean' in tag), None)
    dirty_baseline_tag = next((tag for tag in all_tags if 'supervised_on_contaminated' in tag), None)
    clean_student_tag = next((tag for tag in all_tags if f'distilled_clean_teacher_{args.distillation_method}' in tag), None)
    dirty_student_tag = next((tag for tag in all_tags if f'distilled_dirty_teacher_{args.distillation_method}' in tag), None)

    if not all([clean_baseline_tag, dirty_baseline_tag, clean_student_tag, dirty_student_tag]):
        logger.error("Could not identify all required model tags.")
        return

    analysis_df = pd.DataFrame({
        'laundering_score_raw': (1.0 - pivot_df[dirty_student_tag]) - (1.0 - pivot_df[clean_student_tag]),
        'contamination_score_raw': (1.0 - pivot_df[dirty_baseline_tag]) - (1.0 - pivot_df[clean_baseline_tag]),
        'difficulty_baseline': 1.0 - pivot_df[clean_baseline_tag],
    }).dropna().reset_index()

    csv_path = os.path.join(output_path, f"SAMPLE_LEVEL_RAW_{base_fname}.csv")
    analysis_df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw sample-level data to {csv_path}")

    plot_path = os.path.join(output_path, f"PLOT_TRENDS_{base_fname}_raw_difficulty_only.png")
    generate_difficulty_trend_plot(analysis_df, plot_path)

    logger.info("Sample-level analysis complete.")


if __name__ == '__main__':
    main()

