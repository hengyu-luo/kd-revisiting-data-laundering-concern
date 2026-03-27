#!/bin/bash
# Submit training+evaluation jobs for all task/level/seed combinations.
# This script only runs:
#   - stage 3 (training)
#   - stage 4 (evaluation)

set -euo pipefail

VENV_PATH=${VENV_PATH:-""}

TASKS=${TASKS:-"rotten_tomatoes emotion"}
LEVELS=${LEVELS:-"1 2 3 4 5"}
SEEDS=${SEEDS:-"1 42 86 358 1024"}

TEACHER_MODEL=${TEACHER_MODEL:-bert-base-uncased}
STUDENT_MODEL=${STUDENT_MODEL:-distilbert-base-uncased}
CONTAMINATION_MODE=${CONTAMINATION_MODE:-add}

SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-your_slurm_account}
TRAIN_PARTITION=${TRAIN_PARTITION:-small-g}
TRAIN_TIME=${TRAIN_TIME:-04:00:00}
TRAIN_NODES=${TRAIN_NODES:-1}
TRAIN_NTASKS=${TRAIN_NTASKS:-1}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-1}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
cd "${SCRIPT_DIR}"

DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/datasets"}
RESULTS_DIR=${RESULTS_DIR:-"${SCRIPT_DIR}/results"}
SLURM_LOG_DIR=${SLURM_LOG_DIR:-"${SCRIPT_DIR}/slurmlog"}
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${SLURM_LOG_DIR}"

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

if [[ "${SBATCH_ACCOUNT}" == "your_slurm_account" ]]; then
  echo "[Warning] SBATCH_ACCOUNT is still placeholder value 'your_slurm_account'."
fi

if [[ -n "${VENV_PATH}" ]]; then
  if [[ ! -f "${VENV_PATH}" ]]; then
    echo "[Warning] VENV_PATH is set but file not found: ${VENV_PATH}" >&2
  fi
else
  echo "[Info] VENV_PATH is empty. Ensure required Python packages are available on compute nodes."
fi

ALL_JOB_IDS=()

echo "[Submit] Stage B only: training + evaluation matrix"
for TASK in ${TASKS}; do
  for LEVEL in ${LEVELS}; do
    for SEED in ${SEEDS}; do
      EXPERIMENT_TAG="${TASK}-fixedtest_by_similarity_level_${LEVEL}"
      DATASET_PATH="${DATA_DIR}/${EXPERIMENT_TAG}"

      if [[ ! -d "${DATASET_PATH}" ]]; then
        echo "[Error] Missing dataset path: ${DATASET_PATH}" >&2
        echo "        Run dataset preparation first (step 1-2)." >&2
        exit 3
      fi

      JOB_NAME="DL_${TASK}_L${LEVEL}_S${SEED}"
      TEMP_SCRIPT="${SCRIPT_DIR}/temp_${JOB_NAME}_$$.sh"

      cat > "${TEMP_SCRIPT}" <<EOL
#!/bin/bash
set -euo pipefail
cd "${SCRIPT_DIR}"

# Runtime environment note:
# Optional virtual environment activation:
#   export VENV_PATH="/path/to/venv/bin/activate"
if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
fi

"${PYTHON_BIN}" 3_run_training.py \\
  --dataset_path "${DATASET_PATH}" \\
  --experiment_tag "${EXPERIMENT_TAG}" \\
  --teacher_model "${TEACHER_MODEL}" \\
  --student_model "${STUDENT_MODEL}" \\
  --contamination_mode "${CONTAMINATION_MODE}" \\
  --seed "${SEED}" \\
  --task "${TASK}" \\
  --train_subset_ratio 1.0

"${PYTHON_BIN}" 4_run_evaluation.py \\
  --dataset_path "${DATASET_PATH}" \\
  --experiment_tag "${EXPERIMENT_TAG}" \\
  --teacher_model "${TEACHER_MODEL}" \\
  --student_model "${STUDENT_MODEL}" \\
  --contamination_mode "${CONTAMINATION_MODE}" \\
  --seed "${SEED}" \\
  --task "${TASK}" \\
  --train_subset_ratio 1.0
EOL

      SBATCH_ARGS=(
        --parsable
        --job-name="${JOB_NAME}"
        --output="${SLURM_LOG_DIR}/%x_%j.out"
        --error="${SLURM_LOG_DIR}/%x_%j.err"
        --partition="${TRAIN_PARTITION}"
        --nodes="${TRAIN_NODES}"
        --ntasks-per-node="${TRAIN_NTASKS}"
        --time="${TRAIN_TIME}"
        --account="${SBATCH_ACCOUNT}"
        --chdir="${SCRIPT_DIR}"
      )
      if [[ "${TRAIN_GPUS_PER_NODE}" -gt 0 ]]; then
        SBATCH_ARGS+=(--gpus-per-node="${TRAIN_GPUS_PER_NODE}")
      fi

      JID=$(sbatch "${SBATCH_ARGS[@]}" "${TEMP_SCRIPT}")
      ALL_JOB_IDS+=("${JID}")
      echo "  TASK=${TASK} LEVEL=${LEVEL} SEED=${SEED} job=${JID}"

      rm -f "${TEMP_SCRIPT}"
    done
  done
done

if [[ ${#ALL_JOB_IDS[@]} -eq 0 ]]; then
  echo "[Error] No jobs were submitted." >&2
  exit 4
fi

echo "[Submit] Done"
echo "  train/eval jobs: ${#ALL_JOB_IDS[@]}"
