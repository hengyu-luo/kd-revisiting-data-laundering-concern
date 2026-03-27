#!/bin/bash
#SBATCH --job-name=DLaundry_Prepare
#SBATCH --output=slurmlog/%x_%j.out
#SBATCH --error=slurmlog/%x_%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --account=your_slurm_account

set -euo pipefail

VENV_PATH=${VENV_PATH:-""}

TASK=${TASK:-""}
TASKS=${TASKS:-"rotten_tomatoes emotion"}
NUM_BINS=${NUM_BINS:-5}
SPLIT_SEED=${SPLIT_SEED:-42}
RUN_GAP_TREND_PLOT=${RUN_GAP_TREND_PLOT:-1}

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
fi

if [ -f "${BASE_DIR}/1_create_extreme_dataset.py" ]; then
  CODE_DIR="${BASE_DIR}"
elif [ -f "${BASE_DIR}/distributional_gap_experiment/1_create_extreme_dataset.py" ]; then
  CODE_DIR="${BASE_DIR}/distributional_gap_experiment"
else
  echo "[Error] Could not locate distributional_gap_experiment code directory from ${BASE_DIR}" >&2
  exit 2
fi
cd "${CODE_DIR}"

DATA_DIR=${DATA_DIR:-"${CODE_DIR}/datasets"}
RESULTS_DIR=${RESULTS_DIR:-"${CODE_DIR}/results"}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"${CODE_DIR}/slurmlog"}
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${SLURM_LOG_DIR}" "${CODE_DIR}/results/plots"

# Backward compatibility:
# - If TASK is provided, run only that single task.
# - Otherwise run all tasks listed in TASKS.
if [[ -n "${TASK}" ]]; then
  TASKS="${TASK}"
fi

echo "[Prepare] TASKS=${TASKS} NUM_BINS=${NUM_BINS} SPLIT_SEED=${SPLIT_SEED}"

# Optional virtual environment activation:
#   export VENV_PATH="/path/to/venv/bin/activate"
if [[ -n "${VENV_PATH}" ]]; then
  if [[ -f "${VENV_PATH}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}"
  else
    echo "[Warning] VENV_PATH is set but file not found: ${VENV_PATH}" >&2
  fi
else
  echo "[Info] VENV_PATH is empty. Ensure required Python packages are available."
fi

for TASK_NAME in ${TASKS}; do
  echo "[Task=${TASK_NAME}] [1/4] Creating fixed-test similarity bins"
  python 1_create_extreme_dataset.py \
    --task "${TASK_NAME}" \
    --output_dir "${DATA_DIR}" \
    --mode fixed_test_quintiles \
    --num_bins "${NUM_BINS}" \
    --seed "${SPLIT_SEED}"

  echo "[Task=${TASK_NAME}] [2/4] Measuring distributional gap for each level"
  for LEVEL in $(seq 1 "${NUM_BINS}"); do
    EXPERIMENT_TAG="${TASK_NAME}-fixedtest_by_similarity_level_${LEVEL}"
    DATASET_PATH="${DATA_DIR}/${EXPERIMENT_TAG}"
    GAP_JSON="${RESULTS_DIR}/gap_metrics_${EXPERIMENT_TAG}.json"

    if [ ! -d "${DATASET_PATH}" ]; then
      echo "[Error] Missing dataset: ${DATASET_PATH}" >&2
      exit 3
    fi

    python 2_measure_gap.py \
      --dataset_path "${DATASET_PATH}" \
      --output_json "${GAP_JSON}"
  done

  echo "[Task=${TASK_NAME}] [3/4] Plotting similarity-bin distributions"
  python 2_measure_gap.py \
    --task "${TASK_NAME}" \
    --plot_similarity_bins \
    --similarity_seed "${SPLIT_SEED}" \
    --similarity_num_bins "${NUM_BINS}" \
    --similarity_cache_dir "cache_fixed_test" \
    --similarity_plot_out "results/plots/${TASK_NAME}_similarity_bins.pdf"

  if [ "${RUN_GAP_TREND_PLOT}" = "1" ]; then
    echo "[Task=${TASK_NAME}] [4/4] Plotting gap-trend metrics from gap_metrics_*.json"
    python 2_measure_gap.py \
      --task "${TASK_NAME}" \
      --plot_gap_trends \
      --results_dir "${RESULTS_DIR}" \
      --gap_plot_out_dir "results/plots/gap_trends" \
      --max_level "${NUM_BINS}"
  fi
done

echo "[Prepare] Done for TASKS=${TASKS}"
