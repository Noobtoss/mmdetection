#!/bin/bash
#SBATCH --job-name=mmdetection # Kurzname des Jobs
#SBATCH --array=1-8%4
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT=/nfs/scratch/staff/schmittth/code_hub/mmdetection
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/mmdetection_${SLURM_JOB_ID}_XXXXXX")

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0

eval "$(conda shell.bash hook)"
conda activate conda-mmdetection

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# ----- GET ARGS ----------------------------------------------------
EXP_ARGS_FILE="$ROOT/custom/slurm/experiments.json"

RUN_NAME=$(python -c '
import json, sys
experiments = json.load(open(sys.argv[1]))
task_id = int(sys.argv[2])
if not 1 <= task_id <= len(experiments):
    sys.exit(f"ERROR: task id {task_id} is outside 1-{len(experiments)}")
print(list(experiments)[task_id - 1])
' "$EXP_ARGS_FILE" "$SLURM_ARRAY_TASK_ID") || exit 1

ARGS=$(python -c '
import json, sys
exp = json.load(open(sys.argv[1]))[sys.argv[2]]
print(" ".join(f"{k} {v}" for k, v in exp.items()))
' "$EXP_ARGS_FILE" "$RUN_NAME") || { echo "ERROR: run_name '$RUN_NAME' not found in $EXP_ARGS_FILE" >&2; exit 1; }

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done

# ----- SET ARGS ----------------------------------------------------
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
MODEL_CFG="${KV[model_cfg]}"
DATA_CFG="${KV[data_cfg]}"
DATA_DIR="${KV[data_dir]}"
if [[ -n "$DATA_DIR" ]]; then
    DATA_DIR="${ROOT}/${DATA_DIR}"
    ARGS="$ARGS data_dir $DATA_DIR"
fi
TRAIN_ANN="${KV[train_ann]}"
VAL_ANN="${KV[val_ann]}"
TEST_ANN="${KV[test_ann]}"
MODS_CFG="${KV[mods_cfg]:-custom/configs/mods.py}"
if [[ -n "${KV[ckpt]}" ]]; then
    CKPT="${KV[ckpt]}"
    [[ "$ARGS" != *"load_from"* ]] && ARGS="$ARGS load_from ${ROOT}/${CKPT}"
fi

# ----- WANDB -------------------------------------------------------
# export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_API_KEY=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
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python $ROOT/custom/src/train.py \
       --run_name  "$RUN_NAME" \
       --work_dir  "$OUT_DIR" \
       --model_cfg "$MODEL_CFG" \
       --data_cfg  "$DATA_CFG" \
       --mods_cfg  "$MODS_CFG" \
       --data_dir  "$DATA_DIR" \
       --train_ann "$TRAIN_ANN" \
       --val_ann   "$VAL_ANN" \
       --test_ann  "$TEST_ANN" \
       $ARGS
TRAIN_STATUS=$?

# ----- CLEANUP -----------------------------------------------------
# Accepted risk: cleanup deletes local W&B recovery data even if every sync attempt fails.
KEEP_FILES=("last_checkpoint.pth" "config.py")

LAST_CKPT=$(cat "$OUT_DIR/last_checkpoint")
mv -f "$OUT_DIR/${LAST_CKPT##*/}" "$OUT_DIR/last_checkpoint.pth"

for run in "$OUT_DIR"/*/vis_data/wandb/latest-run; do
    [ -d "$run" ] || continue
    for i in 1 2 3 4 5; do
        if wandb sync --include-online "$run"; then
            WANDB_SYNC_STATUS=0
            break
        fi
        echo "sync attempt $i failed, retrying..."
        sleep 30
    done
done

rm -rf "$TMPDIR"
rm -rf "$OUT_DIR"/*/vis_data
find "$OUT_DIR" -type f $(printf ' ! -name %s' "${KEEP_FILES[@]}") -delete
find "$OUT_DIR" -type d -empty -delete

if [ "$TRAIN_STATUS" -eq 0 ]; then
    touch "$OUT_DIR/TRAIN_SUCCESS"
fi
exit "$TRAIN_STATUS"
