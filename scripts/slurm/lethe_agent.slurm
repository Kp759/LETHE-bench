#!/bin/bash
# Agentic leakage track (lethe/agentic.py) for one method on one split. Array index -> (seed, method)
# as in lethe_method.slurm. Needs the split (lethe_build.slurm).
#   MODEL=llama32_3b SEEDS="1" METHODS=noop,ga,rome sbatch --export=ALL --array=0-2 scripts/slurm/lethe_agent.slurm
#SBATCH --job-name=sc_agent
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/lethe_agent_%A_%a.out
#SBATCH --error=/scratch/yl258/kp759/Unlearning/semantic-unlearning/logs/lethe_agent_%A_%a.err
set -euo pipefail
source /scratch/yl258/kp759/Unlearning/semantic-unlearning/LETHE-bench/scripts/slurm/lethe_common.sh
lethe_check
read -r -a SEED_LIST <<< "${SEEDS:-1}"
METHODS="${METHODS:?METHODS=comma or space separated list}"
read -r -a METHOD_LIST <<< "${METHODS//,/ }"
I="${SLURM_ARRAY_TASK_ID:?run as an array job}"
N=${#METHOD_LIST[@]}
SEED="${SEED_LIST[$(( I / N ))]:-}"; METHOD="${METHOD_LIST[$(( I % N ))]}"
[[ -n "$SEED" ]] || { echo "array index $I out of range" >&2; exit 2; }
DIR="$(split_dir "$SEED")"; SPLIT="$DIR/split.json"; OUT="$DIR/agent/$METHOD"
cd "$SP"
echo "agent $METHOD | $MODEL $PARTITION seed $SEED | job ${SLURM_ARRAY_JOB_ID:-}_$I | $(hostname) | $(date) | commit $(git rev-parse --short HEAD)"
[[ -f "$SPLIT" ]] || { echo "no split $SPLIT (did lethe_build.slurm fail for seed $SEED?)" >&2; exit 2; }
if [[ -f "$OUT/agent.json" ]]; then echo "already done: $OUT"; exit 0; fi
[[ -e "$OUT" ]] && mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
# shellcheck disable=SC2046
python -u -m lethe agent --split "$SPLIT" --model "$MODEL_PATH" --method "$METHOD" $(method_args "$METHOD") \
  --out "$OUT" --max-unlearn-seconds "${MAX_UNLEARN_SECONDS:-14400}" --device cuda --dtype bf16 \
  --batch-size "${AGENT_BATCH:-16}" ${AGENT_ARGS:-}
echo "done $(date)"
