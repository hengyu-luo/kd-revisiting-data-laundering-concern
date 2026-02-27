# Benchmark Similarity Analysis

This directory contains the benchmark-level train-test similarity analysis used to contextualize laundering behavior across datasets.

## Files

- `benchmark_similarity.py`: Main script. Reconstructs the train split setup, simulates replacement-style contamination, and computes similarity metrics.
- `benchmark_similarity_result.txt`: Example terminal output from a completed run.
- `README.md`: This document.

## What the script computes

For each dataset, the script compares test data against:

- `Orig`: the actual training subset used in experiments.
- ~~`Resid`: the residual clean portion after replacement-style contamination simulation.~~ (Note that the statistics from `Resid` are not included in the paper to simplify the discussion.)

It reports metrics in three views:

- `global`: label-agnostic metric over the full dataset.
- `per_label`: metric computed separately per class.
- `macro_average`: mean over per-label scores.

Metrics:

- `jaccard`: 2-gram Jaccard overlap.
- `tfidf`: cosine similarity between mean TF-IDF vectors.
- `avg_emb_sim`: cosine similarity between mean sentence embeddings.
- `max_mean_sim`: for each test sample, max semantic similarity to train samples, then averaged.
- `pattern_conformity`: max similarity to KMeans centroids learned on train embeddings, then averaged.

## Datasets and subset ratios

Configured in the script:

- `agnews`: 0.1
- `snli`: 0.1
- `tweet_sentiment`: 0.5
- `20newsgroups`, `emotion`, `imdb`, `rotten_tomatoes`, `banking77`: 1.0

Random seed: `42`.

## Requirements

```bash
pip install datasets sentence-transformers torch scikit-learn pandas numpy nltk
```

The script downloads datasets and the `all-mpnet-base-v2` embedding model.

## Run

```bash
python benchmark_similarity/benchmark_similarity.py
```

Outputs:

- Per-dataset per-label tables printed to terminal.
- One final summary table printed to terminal.
- `similarity_results.json` written to current working directory.

## Notes for interpreting results

- If `Resid` is empty for a dataset, residual metrics become `0.0` by construction.
- `pattern_conformity` can be `0.0` for some per-label slices when train samples are fewer than `n_clusters=50` (the script skips clustering in that case).
