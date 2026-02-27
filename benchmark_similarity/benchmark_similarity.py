# -*- coding: utf-8 -*-
"""
Compute train-test similarity metrics for multiple classification benchmarks.

For each dataset, the script reports:
1. Global (label-agnostic) similarity.
2. Per-label similarity.
3. Macro-average over labels.
"""

import shutil
import json
import random
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import nltk

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def clear_hf_cache():
    """Clear Hugging Face datasets cache to force a fresh download."""
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'datasets'
    if cache_dir.exists() and cache_dir.is_dir():
        logger.info(f"--> Removing Hugging Face datasets cache at: {cache_dir}")
        shutil.rmtree(cache_dir)
        logger.info('--> Cache removed successfully.')
    else:
        logger.info('--> Cache directory not found, no action needed.')


def get_dataset_info(task):
    """Return dataset configuration for a task."""
    dataset_configs = {
        'imdb': {'name': 'imdb', 'text_field': 'text', 'label_field': 'label'},
        'snli': {'name': 'snli', 'premise_field': 'premise', 'hypothesis_field': 'hypothesis', 'label_field': 'label'},
        'agnews': {'name': 'ag_news', 'text_field': 'text', 'label_field': 'label'},
        'emotion': {'name': 'dair-ai/emotion', 'text_field': 'text', 'label_field': 'label'},
        'banking77': {'name': 'PolyAI/banking77', 'text_field': 'text', 'label_field': 'label'},
        'tweet_sentiment': {'name': 'tweet_eval', 'config': 'sentiment', 'text_field': 'text', 'label_field': 'label'},
        'rotten_tomatoes': {'name': 'rotten_tomatoes', 'text_field': 'text', 'label_field': 'label'},
        '20newsgroups': {'name': 'SetFit/20_newsgroups', 'text_field': 'text', 'label_field': 'label'},
    }
    config = dataset_configs[task]
    config['train_split'] = 'train'
    config['test_split'] = 'test'
    return config


def load_and_prepare_data(task: str, subset_ratio: float, seed: int = 42):
    """Load, subset, and contaminate data, then return DataFrames with text/label."""
    dataset_info = get_dataset_info(task)
    logger.info(f"Processing '{task}' with subset ratio {subset_ratio}...")

    raw_datasets = load_dataset(
        dataset_info['name'],
        name=dataset_info.get('config'),
        trust_remote_code=True,
        download_mode='force_redownload',
    )

    train_dataset = raw_datasets[dataset_info['train_split']]
    test_dataset = raw_datasets[dataset_info['test_split']]

    if task == 'snli':
        train_dataset = train_dataset.filter(lambda x: x[dataset_info['label_field']] != -1)
        test_dataset = test_dataset.filter(lambda x: x[dataset_info['label_field']] != -1)

    if subset_ratio < 1.0:
        logger.info(f"Performing stratified sampling for subset ratio: {subset_ratio}")
        stratified_split = train_dataset.train_test_split(
            train_size=subset_ratio,
            stratify_by_column=dataset_info['label_field'],
            seed=seed,
        )
        actual_train_set = stratified_split['train']
        logger.info(f"Created a stratified subset of {len(actual_train_set)} samples.")
    else:
        actual_train_set = train_dataset

    random.seed(seed)
    num_contaminate = len(test_dataset)
    replace_count = min(num_contaminate, len(actual_train_set))
    replace_indices = set(random.sample(range(len(actual_train_set)), replace_count))
    residual_train_set = actual_train_set.filter(
        lambda _, idx: int(idx) not in replace_indices,
        with_indices=True,
    )

    logger.info(
        f"Original train subset size: {len(actual_train_set)}. "
        f"Replaced: {len(replace_indices)}. Residual size: {len(residual_train_set)}."
    )

    def to_dataframe(dataset, info):
        if info.get('premise_field'):
            texts = [f"{p or ''} {h or ''}".strip() for p, h in zip(dataset[info['premise_field']], dataset[info['hypothesis_field']])]
        else:
            texts = list(dataset[info['text_field']])
        labels = list(dataset[info['label_field']])
        return pd.DataFrame({'text': texts, 'label': labels})

    actual_train_df = to_dataframe(actual_train_set, dataset_info)
    residual_train_df = to_dataframe(residual_train_set, dataset_info)
    test_df = to_dataframe(test_dataset, dataset_info)
    return actual_train_df, residual_train_df, test_df


def calculate_ngram_jaccard(corpus1, corpus2, n=2):
    if not corpus1 or not corpus2:
        return 0.0

    def get_ngrams(text_list):
        all_ngrams = set()
        for text in text_list:
            tokens = [word.lower() for word in word_tokenize(str(text))]
            all_ngrams.update(ngrams(tokens, n))
        return all_ngrams

    ngrams1 = get_ngrams(corpus1)
    ngrams2 = get_ngrams(corpus2)
    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))
    return intersection / union if union > 0 else 0.0


def calculate_tfidf_cosine_similarity(corpus1, corpus2):
    if not corpus1 or not corpus2:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        train_vectors = vectorizer.fit_transform(corpus1)
        test_vectors = vectorizer.transform(corpus2)
    except ValueError:
        return 0.0

    mean_train_vector = train_vectors.mean(axis=0)
    mean_test_vector = test_vectors.mean(axis=0)
    return float(cosine_similarity(np.asarray(mean_train_vector), np.asarray(mean_test_vector))[0][0])


def calculate_embedding_similarities(train_embeddings, test_embeddings):
    if train_embeddings.shape[0] == 0 or test_embeddings.shape[0] == 0:
        return {'avg_emb_sim': 0.0, 'max_mean_sim': 0.0}

    mean_train_emb = np.mean(train_embeddings, axis=0)
    mean_test_emb = np.mean(test_embeddings, axis=0)
    avg_sim = util.cos_sim(mean_train_emb, mean_test_emb).item()

    cos_scores = util.cos_sim(test_embeddings, train_embeddings)
    max_scores = torch.max(cos_scores, dim=1)[0]
    max_mean_sim = torch.mean(max_scores).item()

    return {'avg_emb_sim': avg_sim, 'max_mean_sim': max_mean_sim}


def calculate_pattern_conformity(train_embeddings, test_embeddings, n_clusters=50, seed=42):
    if train_embeddings.shape[0] < n_clusters or test_embeddings.shape[0] == 0:
        logger.warning(
            f"Skipping pattern conformity: train samples ({train_embeddings.shape[0]}) < n_clusters ({n_clusters})."
        )
        return 0.0

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    kmeans.fit(train_embeddings)
    train_centroids = kmeans.cluster_centers_

    sim_matrix = cosine_similarity(test_embeddings, train_centroids)
    max_sim_to_centroids = np.max(sim_matrix, axis=1)
    return float(np.mean(max_sim_to_centroids))


def calculate_metrics_suite(train_df, test_df, train_embs, test_embs):
    """Calculate global, per-label, and macro-average metrics."""
    results = {
        'global': {},
        'per_label': {},
        'macro_average': {},
    }

    train_texts = train_df['text'].tolist()
    test_texts = test_df['text'].tolist()

    results['global']['jaccard'] = calculate_ngram_jaccard(train_texts, test_texts)
    results['global']['tfidf'] = calculate_tfidf_cosine_similarity(train_texts, test_texts)
    emb_sims = calculate_embedding_similarities(train_embs, test_embs)
    results['global']['avg_emb_sim'] = emb_sims['avg_emb_sim']
    results['global']['max_mean_sim'] = emb_sims['max_mean_sim']
    results['global']['pattern_conformity'] = calculate_pattern_conformity(train_embs, test_embs)

    unique_labels = sorted(test_df['label'].unique())
    per_label_scores = {metric: [] for metric in results['global'].keys()}

    train_map = {idx: i for i, idx in enumerate(train_df.index)}
    test_map = {idx: i for i, idx in enumerate(test_df.index)}

    for label in unique_labels:
        train_subset_df = train_df[train_df['label'] == label]
        test_subset_df = test_df[test_df['label'] == label]

        if train_subset_df.empty or test_subset_df.empty:
            continue

        train_subset_texts = train_subset_df['text'].tolist()
        test_subset_texts = test_subset_df['text'].tolist()

        train_subset_indices = [train_map[i] for i in train_subset_df.index]
        test_subset_indices = [test_map[i] for i in test_subset_df.index]
        train_subset_embs = train_embs[train_subset_indices]
        test_subset_embs = test_embs[test_subset_indices]

        label_jaccard = calculate_ngram_jaccard(train_subset_texts, test_subset_texts)
        label_tfidf = calculate_tfidf_cosine_similarity(train_subset_texts, test_subset_texts)
        label_emb_sims = calculate_embedding_similarities(train_subset_embs, test_subset_embs)
        label_pattern = calculate_pattern_conformity(train_subset_embs, test_subset_embs)

        results['per_label'][str(label)] = {
            'jaccard': label_jaccard,
            'tfidf': label_tfidf,
            'avg_emb_sim': label_emb_sims['avg_emb_sim'],
            'max_mean_sim': label_emb_sims['max_mean_sim'],
            'pattern_conformity': label_pattern,
            'train_samples': len(train_subset_df),
            'test_samples': len(test_subset_df),
        }

        for metric in per_label_scores:
            if 'emb' in metric or 'max_mean' in metric:
                per_label_scores[metric].append(label_emb_sims[metric])
            else:
                per_label_scores[metric].append(results['per_label'][str(label)][metric])

    for metric, scores in per_label_scores.items():
        results['macro_average'][metric] = float(np.mean(scores)) if scores else 0.0

    return results


def main():
    clear_hf_cache()

    datasets_to_analyze = [
        'agnews',
        'tweet_sentiment',
        '20newsgroups',
        'emotion',
        'imdb',
        'rotten_tomatoes',
        'snli',
        'banking77',
    ]

    subset_ratios = {
        'agnews': 0.1,
        'tweet_sentiment': 0.5,
        '20newsgroups': 1.0,
        'banking77': 1.0,
        'imdb': 1.0,
        'rotten_tomatoes': 1.0,
        'emotion': 1.0,
        'snli': 0.1,
    }
    seed = 42

    logger.info("Loading Sentence Transformer model ('all-mpnet-base-v2')...")
    sbert_model = SentenceTransformer('all-mpnet-base-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_model.to(device)

    all_results = {}

    for task in datasets_to_analyze:
        actual_train_df, residual_train_df, test_df = load_and_prepare_data(task, subset_ratios[task], seed)

        if residual_train_df.empty:
            logger.warning(f"[{task}] Residual train set is empty. Residual metrics will be 0.0.")

        logger.info(f"[{task}] Encoding texts to embeddings...")
        actual_train_embs = sbert_model.encode(actual_train_df['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
        residual_train_embs = sbert_model.encode(residual_train_df['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
        test_embs = sbert_model.encode(test_df['text'].tolist(), show_progress_bar=True, convert_to_numpy=True)

        logger.info(f"[{task}] Calculating similarity metrics suite...")

        results_orig = calculate_metrics_suite(actual_train_df, test_df, actual_train_embs, test_embs)
        results_resid = calculate_metrics_suite(residual_train_df, test_df, residual_train_embs, test_embs)

        all_results[task] = {
            'original_train_set': results_orig,
            'residual_train_set': results_resid,
        }

        print('\n' + '=' * 80)
        logger.info(f"RESULTS FOR: {task}")
        print('=' * 80)

        per_label_df_orig = pd.DataFrame(results_orig['per_label']).T
        per_label_df_resid = pd.DataFrame(results_resid['per_label']).T.add_suffix(' (Resid)')
        per_label_df_orig.columns = [f"{col} (Orig)" for col in per_label_df_orig.columns]
        merged_per_label = pd.concat([per_label_df_orig, per_label_df_resid], axis=1)

        print(f"\n--- Per-Label Similarity Details for {task} ---\n")
        print(merged_per_label.to_string(float_format='%.4f'))

    summary_data = []
    for task, data in all_results.items():
        row = {'Dataset': task}
        for metric in data['original_train_set']['global']:
            row[f'{metric} (Global-Orig)'] = data['original_train_set']['global'][metric]
            row[f'{metric} (Macro-Orig)'] = data['original_train_set']['macro_average'][metric]
            row[f'{metric} (Global-Resid)'] = data['residual_train_set']['global'][metric]
            row[f'{metric} (Macro-Resid)'] = data['residual_train_set']['macro_average'][metric]
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data).set_index('Dataset')

    print('\n\n' + '=' * 80)
    logger.info('FINAL SUMMARY TABLE')
    print('=' * 80)
    print('Comparing Global (label-agnostic) vs. Macro-Average (label-aware) similarities.')
    print('(Orig) = vs. full contaminated training set | (Resid) = vs. pure part of training set\n')

    display_cols = [col for col in summary_df.columns if 'jaccard' in col or 'tfidf' in col or 'pattern' in col]
    print(summary_df[display_cols].to_string(float_format='%.4f'))

    json_filepath = 'similarity_results.json'
    logger.info(f"Saving all results to {json_filepath}...")
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    logger.info('Done.')


if __name__ == '__main__':
    main()
