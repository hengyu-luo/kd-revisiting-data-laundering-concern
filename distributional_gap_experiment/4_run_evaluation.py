#!/usr/bin/env python3
# 4_run_evaluation.py

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 512
OUTPUT_DIR = "./results"
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
EVAL_RESULTS_DIR = os.path.join(METRICS_DIR, "eval_results")
DATA_COLLECTION_BATCH_SIZE = 64

os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

def paired_bootstrap_pvalue(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 10000, rng_seed: int = 1234) -> float:
    """One-sided bootstrap test for H1: model A > model B."""
    corr_a = (pred_a == y_true).astype(np.int32)
    corr_b = (pred_b == y_true).astype(np.int32)
    diff = corr_a - corr_b
    n = diff.size
    if n == 0:
        return 1.0

    observed_delta = diff.mean()
    if observed_delta <= 0.0:
        return 1.0

    # Precompute multinomial sampling probabilities for {-1, 0, 1} differences
    counts = np.zeros(3, dtype=np.int64)
    values, value_counts = np.unique(diff, return_counts=True)
    for value, count in zip(values, value_counts):
        counts[value + 1] = count  # shift to align with indices [0,1,2]
    probs = counts / n

    rng = np.random.default_rng(rng_seed)
    count_le_zero = 0
    for _ in range(n_boot):
        sample_counts = rng.multinomial(n, probs)
        delta = (sample_counts[2] - sample_counts[0]) / n
        if delta <= 0.0:
            count_le_zero += 1
    return (count_le_zero + 1) / (n_boot + 1)

def compute_eval_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits[0] if isinstance(logits, tuple) else logits, axis=-1)
    return {'accuracy': accuracy_score(labels, preds), 'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0)}

def evaluate_single_model(model_path, eval_dataset, eval_batch_size, device, tokenizer_path):
    logger.info("  -> Running basic evaluation (Accuracy/F1)...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    temp_dir = os.path.join(OUTPUT_DIR, "tmp_eval", os.path.basename(model_path))
    args = TrainingArguments(output_dir=temp_dir, per_device_eval_batch_size=eval_batch_size, report_to="none")
    trainer = Trainer(model=model, args=args, eval_dataset=eval_dataset, tokenizer=tokenizer, compute_metrics=compute_eval_metrics)
    metrics = trainer.predict(test_dataset=eval_dataset).metrics
    del model, trainer; torch.cuda.empty_cache()
    return metrics

def collect_detailed_predictions(model_path, tokenized_dataset, device, tokenizer_path):
    logger.info("  -> Collecting detailed per-sample predictions...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=DATA_COLLECTION_BATCH_SIZE)
    preds, gt_probs, entropies, sm_probs = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Collecting predictions", leave=False):
            labels = batch.pop('labels').cpu()
            inputs = {k: v.to(device) for k, v in batch.items()}
            if 'token_type_ids' in inputs and not hasattr(model.config, 'type_vocab_size'):
                inputs.pop('token_type_ids', None)
            probs = F.softmax(model(**inputs).logits.cpu(), dim=-1)
            preds.extend(torch.argmax(probs, dim=-1).numpy())
            entropies.extend(-torch.sum(probs * torch.log(probs + 1e-9), dim=-1).numpy())
            gt_probs.extend(probs[torch.arange(len(labels)), labels].numpy())
            sm_probs.extend(probs.numpy())
    del model; torch.cuda.empty_cache()
    return {'predictions': np.array(preds), 'ground_truth_probs': np.array(gt_probs), 'entropies': np.array(entropies), 'all_softmax_probs': np.array(sm_probs)}

def save_detailed_results_to_parquet(records, path):
    if not records:
        logger.warning("No detailed records to save."); return
    logger.info(f"Saving detailed per-sample results to {path}...")
    df = pd.DataFrame(records)
    cols = ['sample_index', 'model_tag', 'ground_truth', 'prediction', 'entropy', 'ground_truth_prob', 'softmax_probs']
    df[cols].to_parquet(path, engine='pyarrow', index=False)
    logger.info("Detailed results saved to Parquet.")

def main():
    parser = argparse.ArgumentParser("Streamlined Evaluation and Data Collection Script")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--experiment_tag', type=str, required=True, help="Unique identifier for finding the correct training summary.")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--contamination_mode", choices=["add", "replace"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--main_tokenizer_model", type=str)
    parser.add_argument("--task", required=True)
    parser.add_argument("--train_subset_ratio", type=float, required=True)

    args = parser.parse_args()
    set_seed(args.seed); device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct filenames using experiment_tag
    teacher_name = args.teacher_model.replace("/", "-")
    student_name = args.student_model.replace("/", "-")
    base_filename = f"{args.experiment_tag}_{teacher_name}_to_{student_name}_{args.contamination_mode}_seed{args.seed}"
    summary_fname = f"{base_filename}.json"
    summary_path = os.path.join(METRICS_DIR, summary_fname)
    final_json_path = os.path.join(EVAL_RESULTS_DIR, f"EVAL_{summary_fname}")
    final_parquet_path = os.path.join(EVAL_RESULTS_DIR, f"DETAILED_EVAL_{base_filename}.parquet")

    logger.info(f"Loading custom test set from: {args.dataset_path}")
    if not os.path.isdir(args.dataset_path): sys.exit(f"Dataset path not found: {args.dataset_path}")
        
    test_dataset = load_from_disk(args.dataset_path)['test']
    tokenizer_name = args.main_tokenizer_model or args.teacher_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if 'input_ids' not in test_dataset.features:
        test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH), batched=True)
    if 'label' in test_dataset.column_names:
         test_dataset = test_dataset.rename_column('label', 'labels')
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    ground_truth_labels = test_dataset["labels"].tolist()
    
    if not os.path.exists(summary_path):
        sys.exit(f"Training summary not found: {summary_path}")
        
    with open(summary_path) as fp:
        raw_ckpt_paths = json.load(fp)["model_checkpoint_paths"]
    ckpt_paths = {}
    for k, v in raw_ckpt_paths.items():
        if k in ("clean_teacher_checkpoint", "dirty_teacher_checkpoint"):
            ckpt_paths[k] = v
        elif "supervised_on_clean_data" in k:
            ckpt_paths["supervised_on_clean_data"] = v
        elif "supervised_on_contaminated_data" in k:
            ckpt_paths["supervised_on_contaminated_data"] = v
        elif "student_kd_from_clean_soft" in k:
            ckpt_paths["student_kd_from_clean_soft"] = v
        elif "student_kd_from_dirty_soft" in k:
            ckpt_paths["student_kd_from_dirty_soft"] = v
        else:
            ckpt_paths[k] = v
    
    out_json = {"experiment_args": vars(args), "training_summary_file": summary_path, "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "evaluation_metrics_per_model": {}}
    all_detailed_records = []
    preds_map = {}

    for model_tag, model_path in ckpt_paths.items():
        if not os.path.isdir(model_path):
            logger.warning(f"Checkpoint dir missing for '{model_tag}': {model_path}. Skipping.")
            continue
        logger.info(f"--- Processing model: {model_tag} ---")
        agg_metrics = evaluate_single_model(model_path, test_dataset, args.eval_batch_size, device, tokenizer_name)
        out_json["evaluation_metrics_per_model"][model_tag] = {"metrics": agg_metrics}
        logger.info(f"  -> Aggregate Metrics: {agg_metrics}")
        detailed_data = collect_detailed_predictions(model_path, test_dataset, device, tokenizer_name)
        preds_map[model_tag] = detailed_data['predictions']
        for i in range(len(ground_truth_labels)):
            all_detailed_records.append({'sample_index': i, 'model_tag': model_tag, 'ground_truth': ground_truth_labels[i], 'prediction': detailed_data['predictions'][i], 'entropy': detailed_data['entropies'][i], 'ground_truth_prob': detailed_data['ground_truth_probs'][i], 'softmax_probs': detailed_data['all_softmax_probs'][i]})

    # Paired permutation tests per seed
    def add_sig(key_name, a_tag, b_tag):
        if a_tag in preds_map and b_tag in preds_map:
            p = paired_bootstrap_pvalue(np.array(ground_truth_labels), np.array(preds_map[a_tag]), np.array(preds_map[b_tag]), n_boot=10000, rng_seed=args.seed)
            out_json.setdefault('significance', {})[key_name] = {'model_a': a_tag, 'model_b': b_tag, 'p_value': float(p), 'seed': int(args.seed)}

    add_sig('teacher_dirty_vs_clean', 'dirty_teacher_checkpoint', 'clean_teacher_checkpoint')
    add_sig('student_supervised_dirty_vs_clean', 'supervised_on_contaminated_data', 'supervised_on_clean_data')
    add_sig('student_kd_dirty_vs_clean', 'student_kd_from_dirty_soft', 'student_kd_from_clean_soft')

    logger.info(f"Saving aggregate evaluation results to: {final_json_path}")
    with open(final_json_path, "w") as fp: json.dump(out_json, fp, indent=4)
    logger.info(" Aggregate results saved.")
    save_detailed_results_to_parquet(all_detailed_records, final_parquet_path)
    
    logger.info("\n Evaluation and data collection complete!")

if __name__ == "__main__":
    main()


