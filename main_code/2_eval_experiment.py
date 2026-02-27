#!/usr/bin/env python3
"""
Focused Evaluation Script for Contamination and Laundering Analysis
====================================================================
Focused on core metrics and quantitative contamination/laundering analysis.
"""

# --------------------------------------------------------------------------- #
#  Imports                                                                    #
# --------------------------------------------------------------------------- #
import os
import sys
import json
import csv
import logging
import argparse
from typing import Dict
from datetime import datetime
from functools import partial
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)


# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Globals                                                                    #
# --------------------------------------------------------------------------- #
MAX_LENGTH = 512
OUTPUT_DIR = "./results"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
EVAL_RESULTS_DIR = os.path.join(METRICS_DIR, "eval_results")
PERTURBATION_BATCH_SIZE = 32

os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Dataset Loading Helpers                                                    #
# --------------------------------------------------------------------------- #


# Extend this map as new aliases are introduced in experiments
MODEL_NAME_ALIASES: Dict[str, str] = {
    # Decoder-style LMs used as classification teachers
    "llama3.2-1B": "meta-llama/Llama-3.2-1B",
    "qwen3-0.6B": "Qwen/Qwen3-0.6B",
}


def resolve_model_name(identifier: str) -> str:
    """Return the concrete checkpoint path for a possibly-aliased identifier."""
    resolved = MODEL_NAME_ALIASES.get(identifier, identifier)
    # Always expand user directories so local mirrors like ~/checkpoints/... work.
    return os.path.expanduser(resolved)


def load_raw_test_split(task):
    dataset_info = get_dataset_info(task)
    trust_remote_code_datasets = ['PolyAI/banking77']
    needs_trust_remote_code = any(name in dataset_info['name'] for name in trust_remote_code_datasets)

    if 'config' in dataset_info:
        raw_datasets = load_dataset(
            dataset_info['name'],
            dataset_info['config'],
            trust_remote_code=needs_trust_remote_code
        )
    else:
        raw_datasets = load_dataset(
            dataset_info['name'],
            trust_remote_code=needs_trust_remote_code
        )

    test_dataset_raw = raw_datasets[dataset_info['test_split']]

    return test_dataset_raw, dataset_info

# --------------------------------------------------------------------------- #
#  Helper Functions (Simplified Scope)                                        #
# --------------------------------------------------------------------------- #

def get_dataset_info(task):
    """Return dataset information for supported classification tasks."""
    
    if task == 'imdb':
        return {'name': 'imdb', 'train_split': 'train', 'test_split': 'test', 
                'train_size': 25000, 'test_size': 25000, 'ab_ratio': 1, 'num_labels': 2, 
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == 'snli':
        return {'name': 'snli', 'train_split': 'train', 'test_split': 'test', 
                'train_size': 550152, 'test_size': 9824, 'ab_ratio': 55.0, 'num_labels': 3, 
                'premise_field': 'premise', 'hypothesis_field': 'hypothesis', 
                'label_field': 'label'}
    
    elif task == 'agnews':
        return {'name': 'ag_news', 'train_split': 'train', 'test_split': 'test', 
                'train_size': 120000, 'test_size': 7600, 'ab_ratio': 15.8, 'num_labels': 4, 
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == 'emotion':
        return {'name': 'dair-ai/emotion', 'train_split': 'train', 'test_split': 'test',
                'train_size': 16000, 'test_size': 2000, 'ab_ratio': 8.0, 'num_labels': 6,
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == 'banking77':
        return {'name': 'PolyAI/banking77', 'train_split': 'train', 'test_split': 'test',
                'train_size': 10003, 'test_size': 3080, 'ab_ratio': 3.25, 'num_labels': 77,
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == 'tweet_sentiment':
        return {'name': 'tweet_eval', 'config': 'sentiment', 'train_split': 'train', 'test_split': 'test',
                'train_size': 45615, 'test_size': 12284, 'ab_ratio': 3.7, 'num_labels': 3,
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == 'rotten_tomatoes':
        return {'name': 'rotten_tomatoes', 'train_split': 'train', 'test_split': 'test',
                'train_size': 8530, 'test_size': 1066, 'ab_ratio': 8.0, 'num_labels': 2,
                'text_field': 'text', 'label_field': 'label'}
    
    elif task == '20newsgroups':
        return {'name': 'SetFit/20_newsgroups', 'train_split': 'train', 'test_split': 'test',
                'train_size': 11314, 'test_size': 7532, 'ab_ratio': 1.5, 'num_labels': 20,
                'text_field': 'text', 'label_field': 'label'}
    else:
        raise ValueError(f"Unsupported task: {task}")

def preprocess_function(examples, tokenizer, task, max_length=128):
    """Tokenize and process input data based on task type."""
    dataset_info = get_dataset_info(task)
    
    if task in ['imdb', 'agnews', 'emotion', 'banking77', 'tweet_sentiment', 'rotten_tomatoes', '20newsgroups']:
        text_field = dataset_info.get('text_field', 'text')
        return tokenizer(examples[text_field], truncation=True, padding='max_length', max_length=max_length)

    if task == 'snli':
        premises = [p if p is not None else "" for p in examples['premise']]
        hypotheses = [h if h is not None else "" for h in examples['hypothesis']]
        return tokenizer(premises, hypotheses, truncation=True, padding='max_length', max_length=max_length)
    else:
        raise ValueError(f"Unsupported task: {task}")

def load_and_preprocess_test_dataset(task, tokenizer, max_length=MAX_LENGTH, raw_dataset=None, dataset_info=None):
    """Load and preprocess test dataset for evaluation."""
    if dataset_info is None:
        dataset_info = get_dataset_info(task)
    logger.info(f"Loading test dataset for: {dataset_info['name']} (split: {dataset_info['test_split']})")

    if raw_dataset is None:
        trust_remote_code_datasets = ['PolyAI/banking77']
        needs_trust_remote_code = any(name in dataset_info['name'] for name in trust_remote_code_datasets)

        if 'config' in dataset_info:
            raw_datasets = load_dataset(
                dataset_info['name'], 
                dataset_info['config'],
                trust_remote_code=needs_trust_remote_code
            )
        else:
            raw_datasets = load_dataset(
                dataset_info['name'],
                trust_remote_code=needs_trust_remote_code
            )

        test_dataset_raw = raw_datasets[dataset_info['test_split']]
    else:
        test_dataset_raw = raw_dataset
    
    if task == 'snli':
        test_dataset_raw = test_dataset_raw.filter(lambda x: x[dataset_info['label_field']] != -1)

    _preprocess_fn = partial(preprocess_function, tokenizer=tokenizer, task=task, max_length=max_length)
    
    columns_to_remove = list(test_dataset_raw.column_names)
    label_field_to_keep = dataset_info.get('label_field', 'label')

    final_columns_to_remove = [
        col for col in columns_to_remove 
        if col not in ['input_ids', 'attention_mask', 'labels', label_field_to_keep]
    ]

    tokenized_test_dataset = test_dataset_raw.map(
        _preprocess_fn,
        batched=True,
        remove_columns=final_columns_to_remove
    )


    if label_field_to_keep and label_field_to_keep in tokenized_test_dataset.column_names and label_field_to_keep != 'labels':
        tokenized_test_dataset = tokenized_test_dataset.rename_column(label_field_to_keep, 'labels')

    tokenized_test_dataset.set_format('torch')
    return tokenized_test_dataset, test_dataset_raw, dataset_info

# def preprocess_function(examples, tokenizer, task, max_length=128):
#     """Tokenize and process input data based on task type."""
#     dataset_info = get_dataset_info(task)
#     if 'text_field' in dataset_info:
#         return tokenizer(examples[dataset_info['text_field']], truncation=True, padding='max_length', max_length=max_length)
#     elif 'premise_field' in dataset_info:
def compute_eval_metrics(eval_pred, task, tokenizer):
    """Compute basic evaluation metrics for classification tasks."""
    logits, labels = eval_pred
    if isinstance(logits, tuple): logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {'accuracy': acc, 'f1_weighted': f1}


# --------------------------------------------------------------------------- #
#  Analysis Functions                                                         #
# --------------------------------------------------------------------------- #

def _reconstruct_data_from_df(df):
    """
    Takes a DataFrame read from a DETAILED_EVAL parquet file and converts it
    back into the nested dictionary format that run_advanced_metrics expects.
    """
    logger.info("Reconstructing detailed data from DataFrame...")
    collected_data = defaultdict(dict)
    
    # Ensure the dataframe is sorted by sample_index to maintain order
    df = df.sort_values(by='sample_index').reset_index(drop=True)
    
    for model_tag, group in tqdm(df.groupby('model_tag'), desc="Processing models from file"):
        # The group is already sorted by sample_index
        collected_data[model_tag] = {
            'predictions': group['prediction'].to_numpy(),
            'entropies': group['entropy'].to_numpy(),
            'gt_probs': group['ground_truth_prob'].to_numpy(),
            'all_probs': np.stack(group['softmax_probs'].to_numpy())
        }
    return collected_data

# --------------------------------------------------------------------------- #
#  Data Export and Plotting Functions                                         #
# --------------------------------------------------------------------------- #
def save_detailed_results(detailed_data, true_labels, num_samples, output_path, file_format='parquet'):
    """
    Save detailed results to a specified high-performance format.
    """
    logger.info(f"Saving detailed, per-sample results to {output_path} (Format: {file_format})")

    # 1. Convert the nested dictionary into a flat list of records
    records = []
    for i in range(num_samples):
        for model_tag, data in detailed_data.items():
            records.append({
                'sample_index': i,
                'model_tag': model_tag,
                'ground_truth': true_labels[i],
                'prediction': data['predictions'][i],
                'entropy': data['entropies'][i],
                'ground_truth_prob': data['gt_probs'][i],
                'softmax_probs': data['all_probs'][i] # Pandas handles lists/arrays natively
            })
    
    # 2. Create a Pandas DataFrame
    df = pd.DataFrame(records)

    # 3. Save to the chosen format
    if file_format == 'parquet':
        # Use pyarrow engine for best performance
        df.to_parquet(output_path, engine='pyarrow')
    elif file_format == 'csv':
        # For CSV, we still need to serialize the list
        df['softmax_probs'] = df['softmax_probs'].apply(json.dumps)
        df.to_csv(output_path, index=False, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def _perform_paired_permutation_test(y_true, y_pred_a, y_pred_b, n_permutations=10000):
    """
    Performs a paired permutation test to see if model B is significantly better than model A.
    Compares the accuracy of two models on the same test set.
    
    H0: The two models have the same performance. The difference in accuracy is due to chance.
    Ha (one-sided): Model B's accuracy is greater than Model A's.
    Ha (two-sided): Model B's accuracy is different from Model A's.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A (e.g., the 'clean' model).
        y_pred_b: Predictions from model B (e.g., the 'dirty' model).
        n_permutations: Number of permutations to run.

    Returns:
        A dictionary containing the observed accuracy difference, the one-sided p-value,
        and the two-sided p-value.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true).astype(int)
    correct_b = (y_pred_b == y_true).astype(int)
    
    obs_diff = np.mean(correct_b) - np.mean(correct_a)
    
    perm_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # For each sample, randomly swap the outcomes of model A and B
        swap = np.random.randint(0, 2, size=len(y_true))
        
        shuffled_a = np.where(swap, correct_b, correct_a)
        shuffled_b = np.where(swap, correct_a, correct_b)
        
        perm_diffs[i] = np.mean(shuffled_b) - np.mean(shuffled_a)
        
    # One-sided p-value: Pr(perm_diff >= obs_diff)
    # This tests if model B is significantly BETTER than model A
    p_one_sided = np.mean(perm_diffs >= obs_diff)
    
    # Two-sided p-value: Pr(|perm_diff| >= |obs_diff|)
    # This tests if there is any significant DIFFERENCE between A and B
    p_two_sided = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    
    return {
        "observed_acc_difference": float(obs_diff),
        "p_one_sided": float(p_one_sided),
        "p_two_sided": float(p_two_sided)
    }

def _perform_paired_bootstrap_test(y_true, y_pred_a, y_pred_b, n_bootstraps=10000):
    """
    Replaces _perform_paired_permutation_test to match the paper's description.
    
    Args:
        y_true: Ground truth labels.
        y_pred_a: Clean model predictions.
        y_pred_b: Dirty model predictions.
    
    Returns:
        Dictionary with P-value (one-sided), Mean Diff, and 95% CI.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a) # Clean
    y_pred_b = np.asarray(y_pred_b) # Dirty
    
    n = len(y_true)
    
    # 1 means correct, 0 means incorrect
    correct_a = (y_pred_a == y_true).astype(int) # Clean Correctness
    correct_b = (y_pred_b == y_true).astype(int) # Dirty Correctness
    
    # Observed Difference (Dirty - Clean)
    obs_diff = np.mean(correct_b) - np.mean(correct_a)
    
    # Bootstrap Loop
    rng = np.random.default_rng()
    bootstrap_diffs = np.zeros(n_bootstraps)
    
    for i in range(n_bootstraps):
        # Resample indices WITH replacement (Bootstrap)
        indices = rng.integers(0, n, n)
        
        # Calculate accuracy on this bootstrap sample
        acc_a_sample = np.mean(correct_a[indices])
        acc_b_sample = np.mean(correct_b[indices])
        
        # Dirty - Clean
        bootstrap_diffs[i] = acc_b_sample - acc_a_sample
        
    # --- 1. P-value Calculation (Matching Paper Text) ---
    # Paper: "proportion ... where clean model's accuracy was greater than or equal to dirty"
    # Clean >= Dirty  =>  Dirty - Clean <= 0  =>  diff <= 0
    p_value_one_sided = np.mean(bootstrap_diffs <= 0)
    
    # --- 2. Confidence Interval (For Reviewer 7UAR) ---
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return {
        "observed_acc_difference": float(obs_diff),
        "p_one_sided": float(p_value_one_sided), 
        "ci_lower": float(ci_lower),             
        "ci_upper": float(ci_upper),             
        "bootstrap_mean": float(np.mean(bootstrap_diffs))
    }


def evaluate_single_model(model_path, model_tag, task_name, eval_dataset, eval_batch_size, device, tokenizer):
    """Simplified evaluation function for classification only."""
    logging.info(f"Evaluating model: {model_tag}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()
    eval_args = TrainingArguments(output_dir=os.path.join(OUTPUT_DIR, "tmp_eval", model_tag), per_device_eval_batch_size=eval_batch_size, report_to="none")
    trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset, tokenizer=tokenizer, compute_metrics=partial(compute_eval_metrics, task=task_name, tokenizer=tokenizer))
    results = trainer.predict(test_dataset=eval_dataset)
    metrics = results.metrics
    del model, trainer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {model_tag: {"metrics": metrics}}

def run_advanced_metrics(tokenized_test_ds, true_labels, ckpt_paths, pivot_tags, device, precomputed_data_df=None, model_tokenized_datasets=None):
    """
    Lightweight post-eval metrics runner.
    Collects detailed per-sample data and runs paired significance tests.
    """
    res, significance_results = defaultdict(dict), {}
    clean_tag, dirty_tag = pivot_tags
    
    if precomputed_data_df is not None:
        # --- Path: reconstruct data from the provided DataFrame ---
        collected_data = _reconstruct_data_from_df(precomputed_data_df)
    else:
        # Full inference path: run model inference from scratch.
        collected_data = defaultdict(dict)
        # Phase 1: Data Collection
        logger.info("Phase 1: Collecting data from models...")
        for model_tag, model_path in tqdm(ckpt_paths.items(), desc="Collecting model data"):
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            model.eval()
            all_preds, all_gt_probs, all_entropies, all_probs = [], [], [], []
            model_ds = tokenized_test_ds
            if model_tokenized_datasets and model_tag in model_tokenized_datasets:
                model_ds = model_tokenized_datasets[model_tag]
            if model_ds is None:
                logger.warning(f"No tokenized dataset for model {model_tag}; skipping.")
                continue
            full_dataloader = torch.utils.data.DataLoader(model_ds, batch_size=PERTURBATION_BATCH_SIZE)
            with torch.no_grad():
                for batch in full_dataloader:
                    labels = batch.pop('labels').cpu()
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    if 'token_type_ids' in inputs and hasattr(model, 'distilbert'): inputs.pop('token_type_ids', None)
                    outputs = model(**inputs)
                    logits, probs = outputs.logits.cpu(), F.softmax(outputs.logits.cpu(), dim=-1)
                    all_preds.extend(torch.max(probs, dim=-1)[1].numpy())
                    all_entropies.extend((-torch.sum(probs * torch.log(probs + 1e-9), dim=-1)).numpy())
                    all_gt_probs.extend(probs[torch.arange(len(labels)), labels].numpy())
                    all_probs.extend(probs.numpy())
            collected_data[model_tag].update({'predictions': np.array(all_preds), 'entropies': np.array(all_entropies), 'gt_probs': np.array(all_gt_probs), 'all_probs': np.array(all_probs)})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Phase 2: Computing lightweight aggregate metrics...")
    for model_tag, data in collected_data.items():
        res[model_tag]["entropy_analysis"] = {
            "mean_entropy": float(np.mean(data['entropies'])),
            "std_entropy": float(np.std(data['entropies']))
        }
    
    logger.info("Phase 4: Running paired significance tests for clean vs. dirty models...")
    significance_results = {}
    
    # Define all pairs to be tested
    all_tags = list(ckpt_paths.keys())
    model_pairs = []

    # 1. Find teacher pair
    if "clean_teacher_checkpoint" in all_tags and "dirty_teacher_checkpoint" in all_tags:
        model_pairs.append(("dirty_teacher_checkpoint", "clean_teacher_checkpoint"))
        
    # 2. Find supervised pair
    sup_clean = next((t for t in all_tags if "supervised_on_clean" in t), None)
    sup_dirty = next((t for t in all_tags if "supervised_on_contaminated" in t), None)
    if sup_clean and sup_dirty:
        model_pairs.append((sup_dirty, sup_clean))

    # 3. Find all KD pairs from the student tags
    student_tags = [tag for tag in ckpt_paths if tag not in pivot_tags and 'teacher_checkpoint' not in tag]
    
    # Include all KD methods for pairing
    kd_methods_all = ['hard', 'soft_fwd', 'soft_rev', 'hard_mix', 'soft_fwd_mix', 'soft_rev_mix']
    sorted_kd_methods = sorted(kd_methods_all, key=len, reverse=True)
    grouped_students = defaultdict(dict)
    for tag in student_tags:
        for method in sorted_kd_methods:
            if method in tag:
                if 'distilled_clean' in tag: grouped_students[method]['clean'] = tag
                elif 'distilled_dirty' in tag: grouped_students[method]['dirty'] = tag
                break
    
    for method, pair in grouped_students.items():
        if 'clean' in pair and 'dirty' in pair:
            model_pairs.append((pair['dirty'], pair['clean']))

    # Run the tests
    y_true = true_labels
    for dirty_model_tag, clean_model_tag in model_pairs:
        pair_key = f"{dirty_model_tag}_vs_{clean_model_tag}"
        logger.info(f"  ... testing pair: {pair_key}")
        
        if dirty_model_tag in collected_data and clean_model_tag in collected_data:
            preds_dirty = collected_data[dirty_model_tag]['predictions']
            preds_clean = collected_data[clean_model_tag]['predictions']
            # test_result = _perform_paired_permutation_test(y_true, preds_clean, preds_dirty)
            # significance_results[pair_key] = test_result
            bootstrap_result = _perform_paired_bootstrap_test(y_true, preds_clean, preds_dirty)
            significance_results[pair_key] = bootstrap_result
            logger.info(f"      -> P-val: {bootstrap_result['p_one_sided']:.4f} | CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
        else:
            logger.warning(f"Prediction data not found for pair '{pair_key}'. Skipping test.")

    # The collected_data dictionary now has all raw predictions, entropies, etc.
    # The res dictionary has aggregated advanced metrics.
    # The significance_results has the new p-values.
    # We will return all of them to be saved in the JSON.
    return res, collected_data, significance_results


def main():
    parser = argparse.ArgumentParser("Focused evaluation for contamination and laundering analysis")
    parser.add_argument(
        "--task",
        required=True,
        choices=['imdb', 'snli', 'agnews', 'emotion', 'banking77', 'tweet_sentiment', 'rotten_tomatoes', '20newsgroups']
    )
    parser.add_argument("--teacher_model", required=True); parser.add_argument("--student_model", required=True)
    parser.add_argument("--contamination_mode", choices=["add", "replace"], required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--train_subset_ratio", type=float, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--main_tokenizer_model", type=str)
    # Recompute from existing DETAILED_EVAL parquet without rerunning model inference.
    parser.add_argument(
        "--post_process_only", 
        action="store_true",
        help="If set, skips model inference and recomputes advanced metrics from the corresponding DETAILED_EVAL parquet file."
    )

    args = parser.parse_args()
    set_seed(args.seed); device = "cuda" if torch.cuda.is_available() else "cpu"
    log = logging.getLogger("eval")
    resolved_teacher_model = resolve_model_name(args.teacher_model)
    if resolved_teacher_model != args.teacher_model:
        log.info(f"Resolved teacher model alias '{args.teacher_model}' to '{resolved_teacher_model}' for evaluation.")
    if args.main_tokenizer_model:
        resolved_tokenizer_model = resolve_model_name(args.main_tokenizer_model)
        if resolved_tokenizer_model != args.main_tokenizer_model:
            log.info(f"Resolved tokenizer alias '{args.main_tokenizer_model}' to '{resolved_tokenizer_model}'.")
    else:
        resolved_tokenizer_model = resolved_teacher_model

    teacher_name = args.teacher_model.replace("/", "-"); student_name = args.student_model.replace("/", "-")
    summary_fname = f"{args.task}_{args.train_subset_ratio}_{teacher_name}_to_{student_name}_{args.contamination_mode}_seed{args.seed}.json"
    summary_path = os.path.join(METRICS_DIR, summary_fname)
    out_fname = f"EVAL_{summary_fname}"
    final_out_path = os.path.join(EVAL_RESULTS_DIR, out_fname)

    precomputed_df = None
    raw_test_dataset = None
    test_ds_tok = None
    model_tokenized_datasets = None
    
    # Branch logic based on the --post_process_only flag.
    if args.post_process_only:
        # --- Post-processing mode ---
        log.info(f"--- Running in Post-Processing Mode ---")
        
        # 1. Automatically construct the expected parquet file path
        file_prefix = f"EVAL_{summary_fname.replace('.json', '')}"
        parquet_filename = f"DETAILED_{file_prefix}.parquet"
        parquet_path = os.path.join(EVAL_RESULTS_DIR, parquet_filename)
        
        log.info(f"Attempting to load detailed data from: {parquet_path}")
        if not os.path.exists(parquet_path):
            log.error(f"Required parquet file not found for post-processing: {parquet_path}")
            log.error("Please run a full evaluation first (without --post_process_only) to generate it.")
            sys.exit(1)
        
        precomputed_df = pd.read_parquet(parquet_path)
        
        if os.path.exists(final_out_path):
            log.info(f"Loading existing EVAL JSON to update: {final_out_path}")
            with open(final_out_path, 'r') as f:
                out = json.load(f)
        else:
            log.warning(f"Could not find existing EVAL file at {final_out_path}. Creating a new one.")
            out = {"experiment_args": {k: sorted(list(v)) if isinstance(v, set) else v for k, v in vars(args).items()}}

        if not os.path.exists(summary_path): log.error(f"Training summary not found, needed for metadata: {summary_path}"); sys.exit(1)
        with open(summary_path) as fp: training_summary = json.load(fp)
        ckpt_paths = training_summary["model_checkpoint_paths"]

        all_model_tags = precomputed_df['model_tag'].unique()
        labels = precomputed_df[precomputed_df['model_tag'] == all_model_tags[0]].sort_values('sample_index')['ground_truth'].tolist()
        clean_pivot = next((t for t in all_model_tags if "supervised_on_clean" in t), None)
        dirty_pivot = next((t for t in all_model_tags if "supervised_on_contaminated" in t), None)

    else:
        # Full evaluation mode.
        log.info(f"--- Running in Full Evaluation Mode ---")
        out = {
            "experiment_args": {k: sorted(list(v)) if isinstance(v, set) else v for k, v in vars(args).items()},
            "training_summary_file": summary_path,
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if not os.path.exists(summary_path): log.error(f"Training summary not found: {summary_path}"); sys.exit(1)
        with open(summary_path) as fp: training_summary = json.load(fp)
        ckpt_paths = training_summary["model_checkpoint_paths"]
        experiment_cfg = training_summary.get("experiment_config", {})

        resolved_student_model = resolve_model_name(args.student_model)
        teacher_tokenizer_name = experiment_cfg.get("teacher_model_resolved", resolved_teacher_model)
        student_tokenizer_name = experiment_cfg.get("student_model_resolved", resolved_student_model)

        raw_test_dataset, dataset_info_master = load_raw_test_split(args.task)

        tokenizer_cache = {}
        tokenized_dataset_cache = {}

        def get_or_create_tokenizer(tokenizer_name):
            if tokenizer_name not in tokenizer_cache:
                tok = AutoTokenizer.from_pretrained(tokenizer_name)
                if tok.pad_token is None:
                    if tok.eos_token is not None:
                        tok.pad_token = tok.eos_token
                        log.info(f"Set pad_token to eos_token for tokenizer {tokenizer_name}")
                    else:
                        tok.add_special_tokens({'pad_token': '[PAD]'})
                        tok.pad_token = '[PAD]'
                        log.info(f"Added [PAD] token for tokenizer {tokenizer_name}")
                if tok.pad_token_id is None and tok.pad_token is not None:
                    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
                tokenizer_cache[tokenizer_name] = tok
            return tokenizer_cache[tokenizer_name]

        def get_dataset_for_tokenizer(tokenizer_name):
            tok = get_or_create_tokenizer(tokenizer_name)
            if tokenizer_name not in tokenized_dataset_cache:
                ds_tok, _, _ = load_and_preprocess_test_dataset(
                    args.task, tok, MAX_LENGTH, raw_dataset=raw_test_dataset, dataset_info=dataset_info_master
                )
                tokenized_dataset_cache[tokenizer_name] = ds_tok
            return tok, tokenized_dataset_cache[tokenizer_name]

        primary_tokenizer_name = resolved_tokenizer_model if args.main_tokenizer_model else teacher_tokenizer_name
        primary_tokenizer, test_ds_tok = get_dataset_for_tokenizer(primary_tokenizer_name)
        labels = test_ds_tok["labels"].tolist()

        def is_teacher_checkpoint(tag, path):
            if 'teacher_checkpoint' in tag:
                return True
            norm_path = os.path.normpath(path)
            if os.sep + "students" + os.sep in norm_path:
                return False
            base = os.path.basename(norm_path)
            return base.startswith("teacher_")

        eval_results = {}
        model_tokenized_datasets = {}
        for tag, path in ckpt_paths.items():
            if not os.path.isdir(path):
                log.warning(f"Checkpoint dir missing: {path}")
                continue
            if args.main_tokenizer_model:
                tokenizer_for_model = primary_tokenizer
                dataset_for_model = test_ds_tok
            else:
                tokenizer_name_for_model = teacher_tokenizer_name if is_teacher_checkpoint(tag, path) else student_tokenizer_name
                tokenizer_for_model, dataset_for_model = get_dataset_for_tokenizer(tokenizer_name_for_model)
            eval_results.update(
                evaluate_single_model(path, tag, args.task, dataset_for_model, args.eval_batch_size, device, tokenizer_for_model)
            )
            model_tokenized_datasets[tag] = dataset_for_model

        out["evaluation_metrics_per_model"] = {tag: res for tag, res in eval_results.items() if "metrics" in res}

        clean_pivot = next((t for t in eval_results if "clean_student" in t or "supervised_on_clean" in t), None)
        dirty_pivot = next((t for t in eval_results if "dirty_student" in t or "supervised_on_contaminated" in t), None)
        if not clean_pivot: clean_pivot = next((t for t in eval_results if "clean" in t), None)
        if not dirty_pivot: dirty_pivot = next((t for t in eval_results if "contaminated" in t or "dirty" in t), None)

    # --- Shared post-eval analysis: keep only detailed per-sample data and significance tests ---
    if clean_pivot and dirty_pivot:
        advanced_results, detailed_data, significance_results = run_advanced_metrics(
            test_ds_tok, labels,
            ckpt_paths, (clean_pivot, dirty_pivot), device,
            precomputed_data_df=precomputed_df,
            model_tokenized_datasets=model_tokenized_datasets if not args.post_process_only else None,
        )

        if not args.post_process_only:
            file_prefix = f"EVAL_{summary_fname.replace('.json', '')}"
            parquet_path = os.path.join(EVAL_RESULTS_DIR, f"DETAILED_{file_prefix}.parquet")
            save_detailed_results(detailed_data, labels, len(labels), parquet_path)

        if significance_results:
            out["significance_tests"] = significance_results
        for model_tag, adv_metrics in advanced_results.items():
            if model_tag in out.get("evaluation_metrics_per_model", {}):
                out["evaluation_metrics_per_model"][model_tag]["advanced_metrics"] = adv_metrics
    else:
        log.warning("Could not find clean and dirty pivot models. Skipping post-eval analysis.")

    # --- Final Save ---
    with open(final_out_path, "w") as fp:
        json.dump(out, fp, indent=4)
    log.info(f"Final evaluation results written to: {final_out_path}")


if __name__ == "__main__":
    main()



