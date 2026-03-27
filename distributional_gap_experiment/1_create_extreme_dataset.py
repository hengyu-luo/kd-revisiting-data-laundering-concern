# 1_create_extreme_dataset.py

import argparse
import logging
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dataset_info(task):
    """Minimal dataset info helper."""
    configs = {
        'imdb': {'name': 'imdb', 'text_field': 'text', 'label_field': 'label'},
        'rotten_tomatoes': {'name': 'rotten_tomatoes', 'text_field': 'text', 'label_field': 'label'},
        'agnews': {'name': 'ag_news', 'text_field': 'text', 'label_field': 'label'},
        'emotion': {'name': 'dair-ai/emotion', 'text_field': 'text', 'label_field': 'label'},
    }
    if task not in configs:
        raise ValueError(f"Task '{task}' not configured in this script.")
    return configs[task]

def get_difficulty_scores(dataset, text_field, label_field, n_splits=5, seed=42):
    """
    Calculates difficulty scores for each sample in a dataset using k-fold cross-validation.
    Returns a pandas DataFrame with original indices and difficulty scores.
    """
    df = dataset.to_pandas()
    df['difficulty'] = -1.0
    df['original_index'] = df.index
    
    difficulty_cache_path = f'./tmp_difficulty_scores_{dataset.info.dataset_name}_{len(df)}.csv'
    if os.path.exists(difficulty_cache_path):
        logger.info(f"Loading cached difficulty scores from {difficulty_cache_path}")
        difficulty_df = pd.read_csv(difficulty_cache_path)
        df['difficulty'] = difficulty_df['difficulty']
    else:
        logger.info("No cache found. Calculating difficulty scores...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        def tokenize_fn(examples):
            return tokenizer(examples[text_field], truncation=True, max_length=256, padding='max_length')

        for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(df, df[label_field]), total=n_splits, desc="K-Folds")):
            train_ds, val_ds = dataset.select(train_idx), dataset.select(val_idx)
            tokenized_train = train_ds.map(tokenize_fn, batched=True)
            tokenized_val = val_ds.map(tokenize_fn, batched=True).rename_column(label_field, "labels")
            model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df[label_field].unique()))
            training_args = TrainingArguments(output_dir=f'./tmp_hard_subset_training/fold_{fold}', num_train_epochs=1, per_device_train_batch_size=32, report_to="none", save_strategy="no")
            trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
            trainer.train()
            predictions = trainer.predict(tokenized_val)
            probs = torch.softmax(torch.from_numpy(predictions.predictions), dim=-1).numpy()
            true_labels = tokenized_val['labels']
            ground_truth_probs = probs[np.arange(len(true_labels)), true_labels]
            df.loc[val_idx, 'difficulty'] = 1.0 - ground_truth_probs
        
        df[['difficulty']].to_csv(difficulty_cache_path, index=False)

    df = df.sort_values(by='difficulty', ascending=False).reset_index(drop=True)
    return df

def _encode_texts(model: SentenceTransformer, texts, batch_size=64):
    return model.encode(list(texts), show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size)

def compute_per_sample_similarity_to_test(train_ds: Dataset, test_ds: Dataset, text_field: str, cache_dir: str, seed: int = 42):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'per_sample_similarity_to_test.npy')
    if os.path.exists(cache_file):
        logger.info(f"Loading cached per-sample similarities from {cache_file}")
        return np.load(cache_file)

    logger.info("Encoding train/test texts with SentenceTransformer (all-mpnet-base-v2)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert = SentenceTransformer('all-mpnet-base-v2', device=device)
    train_embs = _encode_texts(sbert, train_ds[text_field])
    test_embs = _encode_texts(sbert, test_ds[text_field])
    test_centroid = np.mean(test_embs, axis=0)
    sims = util.cos_sim(train_embs, test_centroid).cpu().numpy().reshape(-1)
    np.save(cache_file, sims)
    # minimal metadata
    with open(os.path.join(cache_dir, 'sim_meta.json'), 'w') as f:
        import json
        json.dump({'seed': seed, 'model': 'all-mpnet-base-v2', 'metric': 'cos_sim_to_test_centroid'}, f, indent=2)
    return sims

def create_fixed_test_quintile_splits(dataset_dict: DatasetDict, text_field: str, label_field: str, num_bins: int = 5, seed: int = 42, cache_root: str = './cache_fixed_test'):
    """
    Keep the official test split fixed. Partition the TRAIN split into per-label bins
    by similarity to the TEST distribution (close->far, i.e. high->low similarity).

    Returns dict: level_1..level_{num_bins}, each with same-size train subset (1/num_bins per label), fixed test.
    """
    train_ds = dataset_dict['train']
    test_ds = dataset_dict['test']

    ds_name = getattr(getattr(train_ds, 'info', None), 'dataset_name', None) or 'unknown'
    cache_dir = os.path.join(cache_root, ds_name)
    sims = compute_per_sample_similarity_to_test(train_ds, test_ds, text_field, cache_dir, seed)

    df = pd.DataFrame({'idx': np.arange(len(train_ds)), 'label': train_ds[label_field], 'sim': sims})
    df['bin'] = -1
    for lbl in sorted(df['label'].unique().tolist()):
        sub = df[df['label'] == lbl].sort_values('sim', ascending=False).reset_index(drop=True)
        chunks = np.array_split(sub.index.values, num_bins)
        for b, idxs in enumerate(chunks, start=1):
            df.loc[sub.loc[idxs, 'idx'].values, 'bin'] = b
    assert (df['bin'] != -1).all(), "Some samples were not assigned a bin."

    splits = {}
    for b in range(1, num_bins + 1):
        bin_indices = df[df['bin'] == b]['idx'].tolist()
        splits[f'level_{b}'] = DatasetDict({'train': train_ds.select(bin_indices), 'test': test_ds})
        logger.info(f"Created level_{b}: train size {len(bin_indices)} (1/{num_bins} per label), fixed test size {len(test_ds)}")

    df.to_csv(os.path.join(cache_dir, f'train_quintile_assignment_seed{seed}.csv'), index=False)
    return splits

def create_graduated_swapping_splits(dataset, difficulty_df, label_field, test_size=0.2, num_levels=5, seed=42):
    """
    Creates a series of datasets with constant size but increasing distributional gap.
    """
    logger.info(f"Creating {num_levels} levels of graduated splits by swapping...")
    
    # --- Step 1: Create the baseline Level 0 (random) split ---
    all_indices = difficulty_df['original_index'].values
    all_labels = dataset.select(all_indices)[label_field]

    train_indices, test_indices, _, _ = train_test_split(
        all_indices, all_labels, test_size=test_size, random_state=seed, stratify=all_labels
    )
    
    all_splits = {}
    all_splits['level_0'] = DatasetDict({
        'train': dataset.select(train_indices),
        'test': dataset.select(test_indices)
    })
    logger.info(f"  -> Level 0 (control) split created. Train: {len(train_indices)}, Test: {len(test_indices)}")
    
    # --- Step 2: Iteratively swap to create harder levels ---
    
    # Map original indices to their difficulty rank
    difficulty_df['rank'] = difficulty_df.index
    idx_to_rank = pd.Series(difficulty_df['rank'].values, index=difficulty_df['original_index']).to_dict()

    current_train_indices = list(train_indices)
    current_test_indices = list(test_indices)
    
    # Determine swap size (e.g., 5% of the test set size per level)
    swap_size = int(len(test_indices) * (1.0 / num_levels) / 2)
    if swap_size == 0: swap_size = 1
    logger.info(f"Using swap size of {swap_size} samples per level.")

    for i in range(1, num_levels):
        # Sort current indices by difficulty rank
        current_train_indices.sort(key=lambda idx: idx_to_rank[idx]) # Easiest to hardest
        current_test_indices.sort(key=lambda idx: idx_to_rank[idx])  # Easiest to hardest

        # Identify samples to swap
        hardest_from_train = current_train_indices[-swap_size:]
        easiest_from_test = current_test_indices[:swap_size]

        # Perform the swap
        current_train_indices = current_train_indices[:-swap_size] + easiest_from_test
        current_test_indices = current_test_indices[swap_size:] + hardest_from_train
        
        all_splits[f'level_{i}'] = DatasetDict({
            'train': dataset.select(current_train_indices),
            'test': dataset.select(current_test_indices)
        })
        logger.info(f"  -> Level {i} split created via swapping.")

    return all_splits

def main():
    parser = argparse.ArgumentParser(description="Create datasets with controlled train-test distribution gap.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="BASE directory to save the new dataset(s).")
    parser.add_argument("--mode", type=str, choices=["fixed_test_quintiles", "legacy_swap"], default="fixed_test_quintiles")
    parser.add_argument("--num_bins", type=int, default=5, help="Bins for fixed_test_quintiles mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    info = get_dataset_info(args.task)

    if args.mode == "fixed_test_quintiles":
        logger.info(f"Loading official splits for task: {args.task}")
        dataset = DatasetDict({'train': load_dataset(info['name'], split='train'), 'test': load_dataset(info['name'], split='test')})
        splits = create_fixed_test_quintile_splits(dataset, info['text_field'], info['label_field'], num_bins=args.num_bins, seed=args.seed)
        for level_tag, dataset_dict in splits.items():
            tag = f"{args.task}-fixedtest_by_similarity_{level_tag}"
            output_path = os.path.join(args.output_dir, tag)
            logger.info(f"Saving dataset to {output_path}...")
            dataset_dict.save_to_disk(output_path)
        logger.info("All fixed-test quintile datasets created successfully.")
    else:
        logger.info("Running legacy swap mode.")
        dataset_pool = concatenate_datasets([load_dataset(info['name'], split='train'), load_dataset(info['name'], split='test')]).shuffle(seed=args.seed)
        difficulty_df = get_difficulty_scores(dataset_pool, info['text_field'], info['label_field'], seed=args.seed)
        all_splits = create_graduated_swapping_splits(dataset_pool, difficulty_df, info['label_field'], test_size=0.2, num_levels=5, seed=args.seed)
        for level_tag, dataset_dict in all_splits.items():
            tag = f"{args.task}-hardswap_{level_tag}"
            output_path = os.path.join(args.output_dir, tag)
            logger.info(f"Saving dataset to {output_path}...")
            dataset_dict.save_to_disk(output_path)
        logger.info("All graduated datasets (legacy swap) created successfully.")

if __name__ == "__main__":
    main()

