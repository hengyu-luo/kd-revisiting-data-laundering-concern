# KD Decontamination Code (main_code)

This folder contains the 4-stage experiment pipeline used for contamination and laundering analysis in classification-based knowledge distillation.

## Files

- `1_training_experiment.py`: Train teacher/student models and save a run summary JSON.
- `2_eval_experiment.py`: Evaluate all checkpoints from the training summary and save aggregate + per-sample outputs.
- `3_global_correlation_analysis.py`: Compute global Pearson correlations among laundering score, contamination score, and baseline difficulty.
- `4_sample_level_analysis.py`: Export sample-level raw scores and generate a difficulty-sorted trend plot.
- `run.sh`: SLURM batch launcher that runs stages 1-4 in sequence.

## Supported tasks

`imdb`, `snli`, `agnews`, `emotion`, `banking77`, `tweet_sentiment`, `rotten_tomatoes`, `20newsgroups`

## Environment

Python 3.10+ is recommended.

Install packages used by the scripts:

```bash
pip install torch transformers datasets scikit-learn pandas numpy scipy pyarrow tqdm matplotlib
```

Notes:
- `matplotlib` is optional for stage 4 plotting.
- GPU is strongly recommended for training/evaluation.

## Stage-by-stage usage

### 1) Training

```bash
python main_code/1_training_experiment.py \
  --task agnews \
  --teacher_model bert-base-uncased \
  --student_model distilbert-base-uncased \
  --contamination_mode replace \
  --seed 42 \
  --train_subset_ratio 0.1 \
  --distillation_strategies soft_fwd,soft_rev,hard
```

Key behavior:
- Uses `train_subset_ratio` (stratified sampling when label column is available).
- Creates contaminated train data (`add` or `replace`).
- Trains:
  - clean teacher
  - dirty teacher
  - clean supervised student
  - dirty supervised student
  - distilled students from clean/dirty teacher for selected strategies
- Saves run summary to:
  - `./results/metrics/{task}_{ratio}_{teacher}_to_{student}_{mode}_seed{seed}.json`

Supported distillation strategies:
- `hard`
- `soft_fwd`
- `soft_rev`
- `hard_mix`
- `soft_fwd_mix`
- `soft_rev_mix`

### 2) Evaluation

```bash
python main_code/2_eval_experiment.py \
  --task agnews \
  --teacher_model bert-base-uncased \
  --student_model distilbert-base-uncased \
  --contamination_mode replace \
  --seed 42 \
  --train_subset_ratio 0.1 \
  --eval_batch_size 8
```

Outputs:
- `./results/metrics/eval_results/EVAL_{...}.json`
- `./results/metrics/eval_results/DETAILED_EVAL_{...}.parquet`

Optional mode:

```bash
python main_code/2_eval_experiment.py ... --post_process_only
```

`--post_process_only` recomputes advanced/significance metrics from an existing `DETAILED_EVAL_*.parquet` without rerunning model inference.

### 3) Global correlation analysis

```bash
python main_code/3_global_correlation_analysis.py \
  --task agnews \
  --teacher_model bert-base-uncased \
  --student_model distilbert-base-uncased \
  --contamination_mode replace \
  --seed 42 \
  --train_subset_ratio 0.1 \
  --distillation_method soft_fwd
```

Outputs (under `./results/consolidated_analysis/global_correlation/{task}/`):
- `GLOBAL_CORR_SUMMARY_{...}.json`
- `GLOBAL_CORR_DATA_{...}.csv`

### 4) Sample-level analysis

```bash
python main_code/4_sample_level_analysis.py \
  --task agnews \
  --teacher_model bert-base-uncased \
  --student_model distilbert-base-uncased \
  --contamination_mode replace \
  --seed 42 \
  --train_subset_ratio 0.1 \
  --distillation_method soft_fwd
```

Outputs (under `./results/consolidated_analysis/complexity_groups/{task}/`):
- `SAMPLE_LEVEL_RAW_{...}.csv`
- `PLOT_TRENDS_{...}_raw_difficulty_only.png` (if `matplotlib` is installed)

## Running all stages with SLURM

Edit `main_code/run.sh`:
- Set `ACCOUNT`, `PARTITION`, and other SLURM parameters.
- Adjust `TASKS`, `TEACHERS`, `STUDENTS`, `SEEDS`, `CONTAMINATION_MODES`.
- Optionally set `VENV_ACTIVATE` to your environment activation script.

Then submit jobs:

```bash
bash main_code/run.sh
```

## Output structure (high level)

- `./results/checkpoints/`: teacher and student checkpoints.
- `./results/experiments/`: contamination metadata JSON.
- `./results/metrics/`: training summary JSON.
- `./results/metrics/eval_results/`: evaluation JSON + detailed parquet.
- `./results/consolidated_analysis/`: stage 3 and stage 4 analysis outputs.

