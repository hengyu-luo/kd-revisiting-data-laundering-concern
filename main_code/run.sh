#!/bin/bash
# 4-stage batch pipeline for the KD decontamination experiments.
#
# Stages:
#   1) 1_training_experiment.py
#   2) 2_eval_experiment.py (writes DETAILED_EVAL_*.parquet)
#   3) 3_global_correlation_analysis.py
#   4) 4_sample_level_analysis.py
#
# This script is designed for SLURM (sbatch).

set -euo pipefail

############################
# USER CONFIG (EDIT THESE) #
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MAIN_DIR="${MAIN_DIR:-${REPO_ROOT}/Main_Code}"
VENV_PATH="${VENV_PATH:-}"

# SLURM settings
ACCOUNT="${ACCOUNT:-your_slurm_account}"
PARTITION="${PARTITION:-small-g}"
TIME="${TIME:-04:00:00}"
NODES="${NODES:-1}"
NTASKS="${NTASKS:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

# Experiment settings
DISTILLATION_STRATEGIES="${DISTILLATION_STRATEGIES:-soft_fwd,soft_fwd_mix,soft_rev,soft_rev_mix,hard,hard_mix}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"

CONTAMINATION_MODES=("replace")
TEACHERS=("llama3.2-1B" "qwen3-0.6B" "bert-base-uncased")
STUDENTS=("distilbert-base-uncased")
SEEDS=(1 42 86 358 1024)
TASKS=("imdb" "snli" "banking77" "20newsgroups" "agnews" "tweet_sentiment" "emotion" "rotten_tomatoes")

get_train_ratio() {
  local task="$1"
  case "$task" in
    agnews|snli) echo "0.1" ;;
    tweet_sentiment) echo "0.5" ;;
    *) echo "1" ;;
  esac
}

SLURMLOG_ROOT="${SLURMLOG_ROOT:-${REPO_ROOT}/slurmlog}"
SLEEP_BETWEEN_SUBMITS_SEC="${SLEEP_BETWEEN_SUBMITS_SEC:-0.5}"

#################################
# END USER CONFIG               #
#################################

norm() {
  echo "$1" | sed -e 's/[-:]/_/g'
}

if [[ ! -f "${MAIN_DIR}/1_training_experiment.py" ]]; then
  echo "Error: cannot find scripts under MAIN_DIR=${MAIN_DIR}" >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch not found in PATH." >&2
  exit 1
fi

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

mkdir -p "${SLURMLOG_ROOT}"

if [[ "${ACCOUNT}" == "your_slurm_account" ]]; then
  echo "Warning: ACCOUNT is still set to placeholder value 'your_slurm_account'."
fi

for TASK in "${TASKS[@]}"; do
  TRAIN_SUBSET_RATIO="$(get_train_ratio "${TASK}")"

  for TEACHER_MODEL in "${TEACHERS[@]}"; do
    for STUDENT_MODEL in "${STUDENTS[@]}"; do
      for CONTAMINATION_MODE in "${CONTAMINATION_MODES[@]}"; do
        for SEED in "${SEEDS[@]}"; do

          TEACHER_TAG="$(norm "${TEACHER_MODEL}")"
          STUDENT_TAG="$(norm "${STUDENT_MODEL}")"

          JOB_NAME="KD4_${TASK}_r${TRAIN_SUBSET_RATIO}_${TEACHER_TAG}_to_${STUDENT_TAG}_${CONTAMINATION_MODE}_s${SEED}"

          LOG_DIR="${SLURMLOG_ROOT}/${TASK}/${TEACHER_TAG}_to_${STUDENT_TAG}/${CONTAMINATION_MODE}/seed_${SEED}"
          mkdir -p "${LOG_DIR}"

          TEMP_SCRIPT="temp_${JOB_NAME}.sh"

          cat > "${TEMP_SCRIPT}" <<EOL
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --time=${TIME}
#SBATCH --ntasks-per-node=${NTASKS}
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --account=${ACCOUNT}

set -euo pipefail

cd "${REPO_ROOT}"

if [[ -n "${VENV_PATH}" && -f "${VENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
fi

start_time=\$(date +%s)
echo "========================================="
echo "JOB START: \$(date)"
echo "TASK=${TASK} | TEACHER=${TEACHER_MODEL} | STUDENT=${STUDENT_MODEL}"
echo "MODE=${CONTAMINATION_MODE} | SEED=${SEED} | RATIO=${TRAIN_SUBSET_RATIO}"
echo "DISTILLATION_STRATEGIES=${DISTILLATION_STRATEGIES}"
echo "========================================="

echo ""
echo "========== PHASE 1: TRAINING =========="
python "${MAIN_DIR}/1_training_experiment.py" \\
  --task "${TASK}" \\
  --teacher_model "${TEACHER_MODEL}" \\
  --student_model "${STUDENT_MODEL}" \\
  --contamination_mode "${CONTAMINATION_MODE}" \\
  --seed "${SEED}" \\
  --train_subset_ratio "${TRAIN_SUBSET_RATIO}" \\
  --distillation_strategies "${DISTILLATION_STRATEGIES}"

echo "PHASE 1 DONE."

echo ""
echo "========== PHASE 2: EVAL =========="
python "${MAIN_DIR}/2_eval_experiment.py" \\
  --task "${TASK}" \\
  --teacher_model "${TEACHER_MODEL}" \\
  --student_model "${STUDENT_MODEL}" \\
  --contamination_mode "${CONTAMINATION_MODE}" \\
  --seed "${SEED}" \\
  --train_subset_ratio "${TRAIN_SUBSET_RATIO}" \\
  --eval_batch_size "${EVAL_BATCH_SIZE}"

echo "PHASE 2 DONE."

echo ""
echo "========== PHASE 3/4: ANALYSIS =========="

STRATS="${DISTILLATION_STRATEGIES}"
STRATS="\${STRATS//,/ }"

for DISTILLATION_METHOD in \${STRATS}; do
  echo ""
  echo "--- Using distillation_method=\${DISTILLATION_METHOD} ---"

  echo "PHASE 3: Global correlation"
  python "${MAIN_DIR}/3_global_correlation_analysis.py" \\
    --task "${TASK}" \\
    --teacher_model "${TEACHER_MODEL}" \\
    --student_model "${STUDENT_MODEL}" \\
    --contamination_mode "${CONTAMINATION_MODE}" \\
    --seed "${SEED}" \\
    --train_subset_ratio "${TRAIN_SUBSET_RATIO}" \\
    --distillation_method "\${DISTILLATION_METHOD}"

  echo "PHASE 4: Sample-level analysis"
  python "${MAIN_DIR}/4_sample_level_analysis.py" \\
    --task "${TASK}" \\
    --teacher_model "${TEACHER_MODEL}" \\
    --student_model "${STUDENT_MODEL}" \\
    --contamination_mode "${CONTAMINATION_MODE}" \\
    --seed "${SEED}" \\
    --train_subset_ratio "${TRAIN_SUBSET_RATIO}" \\
    --distillation_method "\${DISTILLATION_METHOD}"
done

end_time=\$(date +%s)
duration=\$((end_time - start_time))
echo ""
echo "========================================="
echo "JOB END: \$(date)"
echo "DURATION (seconds): \${duration}"
echo "========================================="
EOL

          echo "Submitting job: ${JOB_NAME}"
          sbatch "${TEMP_SCRIPT}"
          rm -f "${TEMP_SCRIPT}"
          sleep "${SLEEP_BETWEEN_SUBMITS_SEC}"

        done
      done
    done
  done
done

echo "All 4-stage jobs submitted."
