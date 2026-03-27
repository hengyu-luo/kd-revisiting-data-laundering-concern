# Distributional Gap Experiment

This folder contains the end-to-end pipeline for controlled train-test distribution gap experiments.

## Default setup in this package

- Tasks: `rotten_tomatoes`, `emotion`
- Seeds: `1 42 86 358 1024`
- Levels: `1 2 3 4 5`
- Contamination mode: `add`
- Teacher/Student: `bert-base-uncased` -> `distilbert-base-uncased`

## Pipeline (1-5)

1. `1_create_extreme_dataset.py`  
   Build fixed-test similarity levels (quintile train splits).

2. `2_measure_gap.py`  
   Measure gap per level and generate:
   - similarity-bin plot (`--plot_similarity_bins`)
   - gap-trend plot (`--plot_gap_trends`)

3. `3_run_training.py`  
   Train teachers/students for each `(task, level, seed)`.

4. `4_run_evaluation.py`  
   Evaluate checkpoints, save JSON + detailed parquet + per-seed significance.

5. `5_plot_eval_results.py`  
   Generate final figures from eval outputs:
   - performance comparison bars
   - significance heatmap

## Main shell scripts

- `run_prepare_indomain_task.sh`  
  Runs steps 1-2 for one or many tasks (`TASKS`), and supports single-task override via `TASK`.

- `submit_full_indomain_pipeline.sh`  
  Batch-submit stage 3-4 only (training+evaluation over all task/level/seed combinations).

- `run_plot_indomain.sh`
  Runs step 5 only (aggregate existing eval outputs and generate figures).

## Quick run
Run from this directory: `distributional_gap_experiment/`.

1. Prepare datasets + gap metrics for both tasks (step 1-2):

```bash
sbatch run_prepare_indomain_task.sh
```

2. Submit training + evaluation matrix (step 3-4):

```bash
SBATCH_ACCOUNT="your_slurm_account" \
CONTAMINATION_MODE="add" \
bash submit_full_indomain_pipeline.sh
```

3. Generate final plots (step 5):

```bash
sbatch --export=ALL,CONTAMINATION_MODES=add run_plot_indomain.sh
```

## Optional overrides

```bash
sbatch --export=ALL,TASKS="rotten_tomatoes emotion",NUM_BINS=5,SPLIT_SEED=42 run_prepare_indomain_task.sh
```

```bash
sbatch --export=ALL,TASK=emotion,NUM_BINS=5,SPLIT_SEED=42 run_prepare_indomain_task.sh
```

```bash
TASKS="rotten_tomatoes emotion" \
LEVELS="1 2 3 4 5" \
SEEDS="1 42 86 358 1024" \
CONTAMINATION_MODE="add" \
SBATCH_ACCOUNT="your_slurm_account" \
bash submit_full_indomain_pipeline.sh
```

```bash
sbatch --export=ALL,TASKS="rotten_tomatoes,emotion",CONTAMINATION_MODES="add" run_plot_indomain.sh
```

## Outputs

- Intermediate datasets: `datasets/`
- Training/eval outputs: `results/metrics/`
- Final plots: `results/plots/target_figures/`

## Notes

- Set your SLURM account in `run_prepare_indomain_task.sh` and `run_plot_indomain.sh` (`#SBATCH --account=...`), and via `SBATCH_ACCOUNT` for `submit_full_indomain_pipeline.sh`.
- If needed, set `VENV_PATH` to your environment activation path before submission.
- Required Python packages include: `torch`, `transformers`, `datasets`, `sentence-transformers`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `pyarrow`.
