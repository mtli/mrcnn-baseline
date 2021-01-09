#!/bin/bash
# The EX version takes explicit GPU IDs as input

absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV=$1
N_NODE=$2
RANK=$3
MASTER_NODE=$4
PORT=$5
GPUIDs=$6
MMDET_PATH=$7
CONFIG_FILE=$8
WORK_DIR=$9
RESUME_FROM=${10}

nodeName=$HOSTNAME

s="${GPUIDs//[^,]}"
nc="${#s}"
nGPU=$((nc+1))
export CUDA_VISIBLE_DEVICES=$GPUIDs

. activate $CONDA_ENV

# cd for getting the mmdet git tag
cd "$MMDET_PATH"

echo "Starting script on $nodeName ($GPUIDs)"
nohup python -m torch.distributed.launch \
	--nnodes $N_NODE \
	--node_rank $RANK \
	--master_addr $MASTER_NODE \
	--master_port $PORT \
	--nproc_per_node $nGPU \
	"$MMDET_PATH/tools/train.py" \
	"$CONFIG_FILE" \
	--work-dir "$WORK_DIR" \
	--launcher pytorch \
	--seed 0 \
	--resume-from "$RESUME_FROM" \
> /tmp/$USER-dist-train-$nodeName.log 2>&1 &