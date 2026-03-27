#!/usr/bin/env python3
# 2_measure_gap.py

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_from_disk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --- Setup ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

METRICS_TO_PLOT = {
    "jaccard": "Jaccard Similarity",
    "tfidf": "TF-IDF Cosine Sim",
    "avg_emb_sim": "Avg Embedding Sim",
    "max_mean_sim": "Avg Max Semantic Sim",
    "pattern_conformity": "Avg Pattern Conformity",
}


# --- Ensure NLTK data is available ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# --- Gap metric utilities ---
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
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
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
        return {"avg_emb_sim": 0.0, "max_mean_sim": 0.0}
    mean_train_emb = np.mean(train_embeddings, axis=0)
    mean_test_emb = np.mean(test_embeddings, axis=0)
    avg_sim = util.cos_sim(mean_train_emb, mean_test_emb).item()
    cos_scores = util.cos_sim(test_embeddings, train_embeddings)
    max_scores = torch.max(cos_scores, dim=1)[0]
    max_mean_sim = torch.mean(max_scores).item()
    return {"avg_emb_sim": avg_sim, "max_mean_sim": max_mean_sim}


def calculate_pattern_conformity(train_embeddings, test_embeddings, n_clusters=50, seed=42):
    if train_embeddings.shape[0] < n_clusters or test_embeddings.shape[0] == 0:
        logger.warning(
            f"Skipping pattern conformity, not enough train samples ({train_embeddings.shape[0]}) for {n_clusters} clusters."
        )
        return 0.0
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(train_embeddings)
    train_centroids = kmeans.cluster_centers_
    sim_matrix = cosine_similarity(test_embeddings, train_centroids)
    max_sim_to_centroids = np.max(sim_matrix, axis=1)
    return float(np.mean(max_sim_to_centroids))


def calculate_metrics_suite(train_df, test_df, train_embs, test_embs):
    results = {"global": {}, "per_label": {}, "macro_average": {}}

    logger.info("Calculating global similarity metrics...")
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    results["global"]["jaccard"] = calculate_ngram_jaccard(train_texts, test_texts)
    results["global"]["tfidf"] = calculate_tfidf_cosine_similarity(train_texts, test_texts)
    emb_sims = calculate_embedding_similarities(train_embs, test_embs)
    results["global"]["avg_emb_sim"] = emb_sims["avg_emb_sim"]
    results["global"]["max_mean_sim"] = emb_sims["max_mean_sim"]
    results["global"]["pattern_conformity"] = calculate_pattern_conformity(train_embs, test_embs)

    logger.info("Calculating per-label similarity metrics...")
    unique_labels = sorted(test_df["label"].unique())
    per_label_scores = {metric: [] for metric in results["global"].keys()}

    for label in tqdm(unique_labels, desc="Processing labels"):
        train_subset_df = train_df[train_df["label"] == label]
        test_subset_df = test_df[test_df["label"] == label]
        if train_subset_df.empty or test_subset_df.empty:
            continue

        train_subset_texts = train_subset_df["text"].tolist()
        test_subset_texts = test_subset_df["text"].tolist()
        train_subset_embs = train_embs[train_subset_df.index]
        test_subset_embs = test_embs[test_subset_df.index]

        label_jaccard = calculate_ngram_jaccard(train_subset_texts, test_subset_texts)
        label_tfidf = calculate_tfidf_cosine_similarity(train_subset_texts, test_subset_texts)
        label_emb_sims = calculate_embedding_similarities(train_subset_embs, test_subset_embs)
        label_pattern = calculate_pattern_conformity(train_subset_embs, test_subset_embs)

        results["per_label"][str(label)] = {
            "jaccard": label_jaccard,
            "tfidf": label_tfidf,
            "avg_emb_sim": label_emb_sims["avg_emb_sim"],
            "max_mean_sim": label_emb_sims["max_mean_sim"],
            "pattern_conformity": label_pattern,
            "train_samples": len(train_subset_df),
            "test_samples": len(test_subset_df),
        }

        per_label_scores["jaccard"].append(label_jaccard)
        per_label_scores["tfidf"].append(label_tfidf)
        per_label_scores["avg_emb_sim"].append(label_emb_sims["avg_emb_sim"])
        per_label_scores["max_mean_sim"].append(label_emb_sims["max_mean_sim"])
        per_label_scores["pattern_conformity"].append(label_pattern)

    logger.info("Calculating macro-average scores...")
    for metric, scores in per_label_scores.items():
        results["macro_average"][metric] = float(np.mean(scores)) if scores else 0.0

    return results


def measure_single_gap(
    dataset_path: str,
    output_json: str,
    text_field: str = "text",
    label_field: str = "label",
):
    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    train_df = pd.DataFrame({"text": dataset["train"][text_field], "label": dataset["train"][label_field]})
    test_df = pd.DataFrame({"text": dataset["test"][text_field], "label": dataset["test"][label_field]})

    logger.info("Loading Sentence Transformer model ('all-mpnet-base-v2')...")
    sbert_model = SentenceTransformer("all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Encoding texts to embeddings...")
    train_embs = sbert_model.encode(train_df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    test_embs = sbert_model.encode(test_df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)

    logger.info("Calculating similarity metrics suite...")
    results = calculate_metrics_suite(train_df, test_df, train_embs, test_embs)
    results["main_gap"] = results["global"].get("max_mean_sim", 0.0)

    print("\n" + "=" * 50)
    logger.info(f"RESULTS FOR: {os.path.basename(dataset_path)}")
    print("=" * 50)
    summary_df = pd.DataFrame({"Global": results["global"], "Macro-Average": results["macro_average"]})
    print(summary_df.to_string(float_format="%.4f"))

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    logger.info(f"Saving all results to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info("Done.")


# --- Similarity-bin visualization ---
def load_quintile_assignment(task: str, seed: int, cache_dir: Path) -> pd.DataFrame:
    assignment_path = cache_dir / task / f"train_quintile_assignment_seed{seed}.csv"
    if not assignment_path.exists():
        raise FileNotFoundError(
            f"Could not find per-sample similarity cache at {assignment_path}. "
            "Run 1_create_extreme_dataset.py --mode fixed_test_quintiles first."
        )
    df = pd.read_csv(assignment_path)
    expected_cols = {"idx", "label", "sim", "bin"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Cache file {assignment_path} missing columns: {sorted(missing)}")
    return df


def prepare_similarity_groups(df: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    groups = []
    for level in sorted(df["bin"].unique()):
        sims = df[df["bin"] == level]["sim"].to_numpy()
        if sims.size == 0:
            continue
        groups.append((f"L{int(level)}", sims))
    return groups


def compute_similarity_quantiles(df: pd.DataFrame, num_bins: int) -> Dict[str, float]:
    probs = np.linspace(0, 1, num_bins + 1)[1:-1]
    values = np.quantile(df["sim"].to_numpy(), probs)
    return {f"Q{int(p * 100)}": v for p, v in zip(probs, values)}


def save_similarity_summary(groups: List[Tuple[str, np.ndarray]], out_csv: Path):
    rows = []
    for label, sims in groups:
        if sims.size == 0:
            continue
        rows.append(
            {
                "group": label,
                "count": int(sims.size),
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims, ddof=1)) if sims.size > 1 else 0.0,
                "min": float(np.min(sims)),
                "25%": float(np.percentile(sims, 25)),
                "50%": float(np.percentile(sims, 50)),
                "75%": float(np.percentile(sims, 75)),
                "max": float(np.max(sims)),
            }
        )
    if rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


def plot_similarity_bins(task: str, seed: int, cache_dir: str, num_bins: int, out_path: Optional[str]):
    cache_path = Path(cache_dir)
    df = load_quintile_assignment(task, seed, cache_path)
    groups = prepare_similarity_groups(df)
    if not groups:
        raise RuntimeError(f"No similarity groups found in cache for task={task}.")

    quantiles = compute_similarity_quantiles(df, num_bins)
    out_file = Path(out_path) if out_path else Path("results/plots") / f"{task}_similarity_bins.pdf"

    labels = [name for name, _ in groups]
    data = [arr for _, arr in groups]
    positions = np.arange(1, len(labels) + 1)

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )

    palette = plt.cm.Blues(np.linspace(0.35, 0.85, len(labels)))
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(0)
    for xpos, sims in zip(positions, data):
        jitter = (rng.random(len(sims)) - 0.5) * 0.12
        ax.scatter(np.full_like(sims, xpos) + jitter, sims, s=6, alpha=0.35, color="#1f77b4", edgecolors="none")

    for label, value in quantiles.items():
        ax.axhline(value, linestyle="--", color="#7f7f7f", alpha=0.4)
        ax.text(positions[-1] + 0.35, value, label, fontsize=8, color="#555555", va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, positions[-1] + 0.7)
    ax.set_xlabel("Train subset (lower level = closer to test centroid)")
    ax.set_ylabel("Cosine similarity to test centroid")
    ax.set_title(f"{task}: train similarity bins")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=220)
    plt.close(fig)

    summary_csv = out_file.with_name(out_file.stem + "_summary.csv")
    save_similarity_summary(groups, summary_csv)
    logger.info(f"Saved similarity plot to {out_file}")
    logger.info(f"Saved similarity summary to {summary_csv}")


# --- Gap-trend visualization ---
def plot_gap_trends(results_dir: str, output_dir: str, max_level: int = 5, task_filter: Optional[str] = None):
    pattern = re.compile(r"gap_metrics_(.+)-fixedtest_by_similarity_level_(\d+)\.json$")
    all_data = []

    if not os.path.isdir(results_dir):
        logger.warning(f"[GapPlot] results_dir does not exist: {results_dir}")
        return

    for filename in sorted(os.listdir(results_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        dataset_key = match.group(1)
        if task_filter and dataset_key != task_filter:
            continue
        level = int(match.group(2))
        file_path = os.path.join(results_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        for metric_type in ("global", "macro_average"):
            metrics = metrics_data.get(metric_type) or {}
            if not metrics:
                continue
            row = {k: metrics.get(k) for k in METRICS_TO_PLOT.keys()}
            row["dataset_key"] = dataset_key
            row["dataset"] = dataset_key.replace("_", " ").title()
            row["level"] = level
            row["metric_type"] = "Micro" if metric_type == "global" else "Macro"
            all_data.append(row)

    if not all_data:
        logger.warning("[GapPlot] No matching gap metric files found.")
        return

    df = pd.DataFrame(all_data).sort_values(by=["dataset_key", "level", "metric_type"])
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="colorblind")

    for dataset_key in df["dataset_key"].unique():
        subset_df = df[df["dataset_key"] == dataset_key]
        if subset_df.empty:
            continue
        dataset_name = subset_df["dataset"].iloc[0]
        fig, axes = plt.subplots(1, len(METRICS_TO_PLOT), figsize=(20, 4))
        fig.suptitle(f"{dataset_name}: Micro vs. Macro Metrics", fontsize=16, y=1.02)

        for i, (metric_key, metric_title) in enumerate(METRICS_TO_PLOT.items()):
            ax = axes[i]
            sns.lineplot(
                data=subset_df,
                x="level",
                y=metric_key,
                hue="metric_type",
                style="metric_type",
                markers=True,
                dashes=True,
                ax=ax,
                legend=(i == len(METRICS_TO_PLOT) - 1),
            )
            ax.set_title(metric_title)
            ax.set_xlabel("Data Level")
            ax.set_ylabel("Similarity Score")
            ax.set_xticks(range(1, max_level + 1))
            ax.grid(True, which="both", linestyle="--")
            if i == len(METRICS_TO_PLOT) - 1:
                ax.legend(title="Average Type", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = os.path.join(output_dir, f"{dataset_key}_comparison_metrics.pdf")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[GapPlot] Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure distributional gap for a pre-split dataset, and optionally generate merged gap-related plots."
    )

    # Original measurement mode
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to saved DatasetDict directory.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save per-dataset gap JSON.")
    parser.add_argument("--text_field", type=str, default="text", help="Text field name.")
    parser.add_argument("--label_field", type=str, default="label", help="Label field name.")

    # Similarity-bin plotting mode
    parser.add_argument("--task", type=str, default=None, help="Task name (e.g., rotten_tomatoes, emotion).")
    parser.add_argument("--plot_similarity_bins", action="store_true", help="Generate similarity-bin figure from cache.")
    parser.add_argument("--similarity_cache_dir", type=str, default="cache_fixed_test", help="Cache root from step 1.")
    parser.add_argument("--similarity_num_bins", type=int, default=5, help="Number of similarity bins.")
    parser.add_argument("--similarity_seed", type=int, default=42, help="Seed used for quintile assignment cache.")
    parser.add_argument("--similarity_plot_out", type=str, default=None, help="Output path for similarity-bin plot.")

    # Gap-trend plotting mode
    parser.add_argument("--plot_gap_trends", action="store_true", help="Generate gap trend plots from gap_metrics_*.json.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing gap_metrics_*.json.")
    parser.add_argument("--gap_plot_out_dir", type=str, default="results/plots/gap_trends", help="Output directory for gap trend plots.")
    parser.add_argument("--max_level", type=int, default=5, help="Max level on x-axis for trend plots.")

    args = parser.parse_args()
    ran_any = False

    if args.dataset_path and args.output_json:
        ran_any = True
        measure_single_gap(
            dataset_path=args.dataset_path,
            output_json=args.output_json,
            text_field=args.text_field,
            label_field=args.label_field,
        )

    if args.plot_similarity_bins:
        if not args.task:
            parser.error("--plot_similarity_bins requires --task.")
        ran_any = True
        plot_similarity_bins(
            task=args.task,
            seed=args.similarity_seed,
            cache_dir=args.similarity_cache_dir,
            num_bins=args.similarity_num_bins,
            out_path=args.similarity_plot_out,
        )

    if args.plot_gap_trends:
        ran_any = True
        plot_gap_trends(
            results_dir=args.results_dir,
            output_dir=args.gap_plot_out_dir,
            max_level=args.max_level,
            task_filter=args.task,
        )

    if not ran_any:
        parser.error(
            "No action specified. Use --dataset_path + --output_json and/or --plot_similarity_bins / --plot_gap_trends."
        )


if __name__ == "__main__":
    main()
