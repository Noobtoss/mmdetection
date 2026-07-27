#!/bin/bash
#SBATCH --job-name=mmdetection_train_arr # Kurzname des Jobs
#SBATCH --array=1%4
#SBATCH --output=logs/R-%A-%a.out
#SBATCH --partition=p2,p6             # p4
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                  # Anzahl Knoten
#SBATCH --ntasks=1                 # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=1          # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem-per-cpu=64G          # RAM pro CPU Kern #20G #32G #64G

# ----- ROOT_DIR ----------------------------------------------------
ROOT_DIR=/nfs/scratch/staff/schmittth/code_nexus/mmdetection

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$ROOT_DIR/custom/slurm/slurm_params.txt"
PARAMS=$(grep -v '^[[:space:]]*#' "$PARAMS_FILE" | sed -n "$((SLURM_ARRAY_TASK_ID))p")

declare -A KV
read -r -a ARR <<< "$PARAMS"
for ((i=0; i<${#ARR[@]}; i+=2)); do
    key="${ARR[$i]}"
    value="${ARR[$i+1]}"
    KV["$key"]="$value"
done
[[ "$PARAMS" != *"seed"* ]] && PARAMS="$PARAMS seed ${SLURM_ARRAY_JOB_ID}"

RUN_NAME="${KV[exp_name]:-unnamed_run}"
RUN_NAME="${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT_DIR="${ROOT_DIR}/runs/${RUN_NAME}"
MODEL_CFG="${KV[model_cfg]}"
DATA_CFG="${KV[data_cfg]}"
TRAIN_CFG="${KV[train_cfg]:-custom/configs/runtime.py}"
CKPT="${KV[ckpt]:-checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth}"
[[ "$PARAMS" != *"load_from"* ]] && PARAMS="$PARAMS load_from ${ROOT_DIR}/${CKPT}"

echo $PARAMS

# ----- ENVIRONMENT SETUP -------------------------------------------
module purge
module load python/anaconda3
module load cuda/cuda-11.8.0
eval "$(conda shell.bash hook)"

conda activate conda-mmdetection

export PYTHONPATH="$ROOT_DIR/custom/src:$PYTHONPATH"
export TMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/mmdetection_${SLURM_JOB_ID}_XXXXXX")

# ----- WANDB -------------------------------------------------------
export WANDB_API_KEY=95177947f5f36556806da90ea7a0bf93ed857d58
export WANDB_CACHE_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_CONFIG_DIR=$TMPDIR

# ----- TRAINING ----------------------------------------------------
python $ROOT_DIR/custom/src/train.py \
       --run_name $RUN_NAME \
       --work_dir $OUT_DIR \
       --model_cfg $MODEL_CFG \
       --data_cfg $DATA_CFG \
       --train_cfg $TRAIN_CFG \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
wandb sync --sync-all || true
rm -rf "$TMPDIR"
rm -rf "$OUT_DIR"/*/vis_data
KEEP_FILES=("last_checkpoint")
find "$OUT_DIR/$EXP_NAME" -type f \
  $(printf ' ! -name %s' "${KEEP_FILES[@]}") \
  ! -name "*.py" \
  ! -name "epoch_*.pth" \
  ! -name "best_*.pth" \
  -delete
find "$OUT_DIR" -type d -empty -delete
