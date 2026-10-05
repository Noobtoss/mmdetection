#!/bin/bash
# slurm_submit_loop.sh
# Usage: ./slurm_submit_loop.sh [slurm_script] [qos] [loop_num] [array] [sleep_hours]

# ----- GET ARGS ----------------------------------------------------

SLURM_SCRIPT=${1:-custom/slurm/slurm_train.sh}
QOS=${2:-normal}
LOOP_NUM=${3:-6}
ARRAY=${4:-1-10%3}
SLEEP_HOURS=${5:-3}

echo "Script: $SLURM_SCRIPT | QOS: $QOS | Loops: $LOOP_NUM | Array: $ARRAY | Sleep: ${SLEEP_HOURS}h"

# ----- GET SLURM LIMITS --------------------------------------------

SUBMIT_LIMIT=$(sacctmgr show assoc user=$USER format=maxsubmit -p -n | tr -d '| ')
[[ -z "$SUBMIT_LIMIT" ]] && SUBMIT_LIMIT=$(sacctmgr show qos "$QOS" format=maxsubmitpu -p -n | tr -d '| ')

RUNNING_LIMIT=$(sacctmgr show assoc user=$USER format=maxjobs -p -n | tr -d '| ')
[[ -z "$RUNNING_LIMIT" ]] && RUNNING_LIMIT=$(sacctmgr show qos "$QOS" format=maxjobspu -p -n | tr -d '| ')

# ----- SUBMIT_LOOPS ------------------------------------------------

BATCH_SIZE=$RUNNING_LIMIT
[[ "$ARRAY" == *%* ]] && BATCH_SIZE=${ARRAY#*%}

CURRENT=$(squeue -u "$USER" -h | wc -l)

if [[ -n "$LIMIT" ]]; then
    FREE=$((LIMIT - CURRENT))
    echo "Limit: $LIMIT | Currently queued: $CURRENT | Free slots: $FREE"
else
    echo "No limit found for user/QOS '$QOS' — cannot determine free slots."
fi
