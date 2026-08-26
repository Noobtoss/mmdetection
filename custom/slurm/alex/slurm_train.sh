#!/bin/bash
#SBATCH --job-name=mmdetection_train # Kurzname des Jobs
#SBATCH --array=8%1
#SBATCH --output=logs/R_%A_%a.out
#SBATCH --gres=gpu:a40:1     # Request 1x A40 GPUs
#SBATCH --partition=a40      # Submit to the a40 node partition
#SBATCH --ntasks=1           # 1 process total (not MPI)
#SBATCH --ntasks-per-node=1  # That 1 process runs on 1 node
#SBATCH --cpus-per-task=4    # 4 CPU cores for that process (data loading etc)
#SBATCH --time=03:32:32      # Walltime limit: kill job after 3hr 32min 32sec
#SBATCH --mail-type=ALL      # Email on job start, end, fail
#SBATCH --mail-user=thomas.schmitt@th-nuernberg.de

# ----- DIRS --------------------------------------------------------
ROOT_DIR="$WORK/code_nexus/mmdetection"
JOB_DIR=$TMPDIR

# ----- GET ARGS ----------------------------------------------------
PARAMS_FILE="$ROOT_DIR/custom/slurm/alex/slurm_params.txt"
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
MODS_CFG="${KV[train_cfg]:-custom/configs/mods.py}"
if [[ -n "${KV[ckpt]}" ]]; then
    CKPT="${KV[ckpt]}"
    [[ "$PARAMS" != *"load_from"* ]] && PARAMS="$PARAMS load_from ${ROOT_DIR}/${CKPT}"
fi

# ----- ENVIRONMENT SETUP -------------------------------------------
unset SLURM_EXPORT_ENV

module purge
module load python/3.12-base
module load cuda/12.8.1

eval "$(conda shell.bash hook)"
conda activate conda-mmdetection

export PYTHONPATH="$ROOT_DIR/custom/src:$PYTHONPATH"

# --- PROXY  --------------------------------------------------------
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

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
       --mods_cfg $MODS_CFG \
       $PARAMS

# ----- CLEANUP -----------------------------------------------------
KEEP_FILES=("last_checkpoint")

wandb sync --sync-all || true
rm -rf "$TMPDIR"
rm -rf "$OUT_DIR"/*/vis_data
find "$OUT_DIR/$EXP_NAME" -type f \
  $(printf ' ! -name %s' "${KEEP_FILES[@]}") \
  ! -name "*.py" \
  ! -name "epoch_*.pth" \
  ! -name "best_*.pth" \
  -delete
find "$OUT_DIR" -type d -empty -delete
