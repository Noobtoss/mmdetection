#!/bin/bash
#SBATCH --job-name=mmdetection # Kurzname des Jobs
#SBATCH --array=1-3%3
#SBATCH --output=logs/%A_%a.out
#SBATCH --partition=p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- DIRS --------------------------------------------------------
ROOT=/nfs/scratch/staff/schmittth/code_nexus/mmdetection
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/mmdetection_${SLURM_JOB_ID}_XXXXXX")

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0

eval "$(conda shell.bash hook)"
conda activate conda-mmdetection

export PYTHONPATH="$ROOT/custom/src:$PYTHONPATH"

# ----- GET ARGS ----------------------------------------------------
EXP_MAP_FILE="$ROOT/custom/slurm/experiments_map.txt"
EXP_ARGS_FILE="$ROOT/custom/slurm/experiments.json"

EXP_NAME=$(grep -v '^[[:space:]]*#' "$EXP_MAP_FILE" | awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print $2; exit}')
if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: no exp_name for task id $SLURM_ARRAY_TASK_ID in $EXP_MAP_FILE" >&2
    exit 1
fi

ARGS=$(python -c '
import json, sys
exp = json.load(open(sys.argv[1]))[sys.argv[2]]
print(" ".join(f"{k} {v}" for k, v in exp.items()))
' "$EXP_ARGS_FILE" "$EXP_NAME") || { echo "ERROR: exp_name '$EXP_NAME' not found in $EXP_ARGS_FILE" >&2; exit 1; }
ARGS="exp_name $EXP_NAME $ARGS"

declare -A KV
read -r -a ARR <<< "$ARGS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done

# ----- SET ARGS ----------------------------------------------------
[[ "$ARGS" != *"seed"* ]] && ARGS="$ARGS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${KV[exp_name]:-unnamed_run}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT}/runs/${RUN_NAME}"
MODEL_CFG="${KV[model_cfg]}"
DATA_CFG="${KV[data_cfg]}"
DATA_DIR="${KV[data_dir]}"
if [[ -n "$DATA_DIR" ]]; then
    DATA_DIR="${ROOT}/${DATA_DIR}"
    ARGS="$ARGS data_dir $DATA_DIR"
fi
ANN_TRAIN="${KV[ann_train]}"
ANN_VAL="${KV[ann_val]}"
ANN_TEST="${KV[ann_test]}"
MODS_CFG="${KV[train_cfg]:-custom/configs/mods.py}"
if [[ -n "${KV[ckpt]}" ]]; then
    CKPT="${KV[ckpt]}"
    [[ "$ARGS" != *"load_from"* ]] && ARGS="$ARGS load_from ${ROOT}/${CKPT}"
fi

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
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
       --ann_train "$ANN_TRAIN" \
       --ann_val   "$ANN_VAL" \
       --ann_test  "$ANN_TEST" \
       $ARGS

# ----- CLEANUP -----------------------------------------------------
if ! LAST_CKPT=$(cat "$OUT_DIR/last_checkpoint") ||
   [[ ! -f "$OUT_DIR/${LAST_CKPT##*/}" ]] ||
   ! mv -f "$OUT_DIR/${LAST_CKPT##*/}" "$OUT_DIR/${RUN_NAME}.pth" ||
   [[ ! -f "$OUT_DIR/${RUN_NAME}.pth" ]]; then
    echo "ERROR: cannot preserve last checkpoint; skipping cleanup." >&2
    exit 1
fi

for i in 1 2 3 4 5; do
    wandb sync --sync-all && break
    echo "sync attempt $i failed, retrying..."
    sleep 20
done

rm -rf "$TMPDIR"
rm -rf "$OUT_DIR"/*/vis_data
find "$OUT_DIR" -type f \
  ! -name "*.log" \
  ! -name "${RUN_NAME}.pth" \
  -delete
find "$OUT_DIR" -type d -empty -delete
