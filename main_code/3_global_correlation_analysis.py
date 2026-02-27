#!/usr/bin/env python3
"""
Global correlation analysis (difficulty-only version).

This script computes:
1. laundering_score (raw)
2. contamination_score (raw)
3. baseline difficulty
and reports their similarity/correlation statistics.
"""
import os
import argparse
import logging
import json
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
EVAL_RESULTS_DIR = "./results/metrics/eval_results"


def main():
    parser = argparse.ArgumentParser(description="Global correlation analysis (difficulty-only).")
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--teacher_model', type=str, required=True)
    parser.add_argument('--student_model', type=str, required=True)
    parser.add_argument('--contamination_mode', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train_subset_ratio', type=float, required=True)
    parser.add_argument('--distillation_method', type=str, default='soft_fwd')
    parser.add_argument('--output_dir', type=str, default='./results/consolidated_analysis/global_correlation')
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
        'laundering_score': (1.0 - pivot_df[dirty_student_tag]) - (1.0 - pivot_df[clean_student_tag]),
        'contamination_score': (1.0 - pivot_df[dirty_baseline_tag]) - (1.0 - pivot_df[clean_baseline_tag]),
        'difficulty_baseline': 1.0 - pivot_df[clean_baseline_tag],
    }).dropna().reset_index()

    corr_lc, p_lc = pearsonr(analysis_df['laundering_score'], analysis_df['contamination_score'])
    corr_ld, p_ld = pearsonr(analysis_df['laundering_score'], analysis_df['difficulty_baseline'])
    corr_cd, p_cd = pearsonr(analysis_df['contamination_score'], analysis_df['difficulty_baseline'])

    summary = {
        "task": args.task,
        "base_filename": base_fname,
        "num_samples": int(len(analysis_df)),
        "pearson": {
            "laundering_vs_contamination": {"r": float(corr_lc), "p": float(p_lc)},
            "laundering_vs_difficulty": {"r": float(corr_ld), "p": float(p_ld)},
            "contamination_vs_difficulty": {"r": float(corr_cd), "p": float(p_cd)},
        }
    }

    out_json = os.path.join(output_path, f"GLOBAL_CORR_SUMMARY_{base_fname}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {out_json}")

    out_csv = os.path.join(output_path, f"GLOBAL_CORR_DATA_{base_fname}.csv")
    analysis_df.to_csv(out_csv, index=False)
    logger.info(f"Saved per-sample data to {out_csv}")


if __name__ == '__main__':
    main()
