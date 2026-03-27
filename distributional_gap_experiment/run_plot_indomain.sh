#!/bin/bash
#SBATCH --job-name=DLaundry_Plots
#SBATCH --output=slurmlog/%x_%j.out
#SBATCH --error=slurmlog/%x_%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --account=your_slurm_account

set -euo pipefail

VENV_PATH=${VENV_PATH:-""}

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  BASE_DIR="${SLURM_SUBMIT_DIR}"
else
  BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
fi

if [ -f "${BASE_DIR}/5_plot_eval_results.py" ]; then
  CODE_DIR="${BASE_DIR}"
elif [ -f "${BASE_DIR}/distributional_gap_experiment/5_plot_eval_results.py" ]; then
  CODE_DIR="${BASE_DIR}/distributional_gap_experiment"
else
  echo "[Error] Could not locate distributional_gap_experiment code directory from ${BASE_DIR}" >&2
  exit 2
fi
cd "${CODE_DIR}"

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

mkdir -p results/plots slurmlog

TASKS=${TASKS:-"rotten_tomatoes,emotion"}
LEVELS=${LEVELS:-"1,2,3,4,5"}
SEEDS=${SEEDS:-"1,42,86,358,1024"}
TEACHER_MODEL=${TEACHER_MODEL:-bert-base-uncased}
STUDENT_MODEL=${STUDENT_MODEL:-distilbert-base-uncased}
CONTAMINATION_MODES=${CONTAMINATION_MODES:-"add"}
RUN_PLOT_EVAL_RESULTS=${RUN_PLOT_EVAL_RESULTS:-1}

TASKS="${TASKS//,/ }"
LEVELS="${LEVELS//,/ }"
SEEDS="${SEEDS//,/ }"
CONTAMINATION_MODES="${CONTAMINATION_MODES//,/ }"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
else
  if command -v python &>/dev/null; then
    PYTHON_BIN=$(command -v python)
  elif command -v python3 &>/dev/null; then
    PYTHON_BIN=$(command -v python3)
  else
    echo "[Error] Could not find python/python3." >&2
    exit 1
  fi
fi

echo "[Plot] TASKS=${TASKS} LEVELS=${LEVELS} SEEDS=${SEEDS} MODES=${CONTAMINATION_MODES}"

if [ "${RUN_PLOT_EVAL_RESULTS}" = "1" ]; then
  echo "[Plot] Generating target figures (accuracy + significance heatmap)"
  "${PYTHON_BIN}" 5_plot_eval_results.py \
    --tasks ${TASKS} \
    --contamination_modes ${CONTAMINATION_MODES} \
    --teacher_model "${TEACHER_MODEL}" \
    --student_model "${STUDENT_MODEL}" \
    --levels ${LEVELS} \
    --seeds ${SEEDS} \
    --pair_keys student_kd_dirty_vs_clean \
    --eval_dir results/metrics/eval_results \
    --out_dir results/plots/target_figures
fi

echo "[Plot] Done. Outputs in results/plots"
