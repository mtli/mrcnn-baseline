#!/bin/bash
absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV=$1
N_NODE=$2
RANK=$3
MASTER_NODE=$4
PORT=$5
N_GPU=$6
MMDET_PATH=$7
CONFIG_FILE=$8
WORK_DIR=$9

nodeName=$HOSTNAME

if [ "$nodeName" == "compute-0-19.local" ]; then
	export CUDA_VISIBLE_DEVICES=3
fi

if [ "$nodeName" == "compute-1-19.local" ]; then
	export CUDA_VISIBLE_DEVICES=0,2,3
fi

. activate $CONDA_ENV

# cd for getting the mmdet git tag
cd "$MMDET_PATH"

echo "Starting script on $nodeName"
nohup python -m torch.distributed.launch \
	--nnodes $N_NODE \
	--node_rank $RANK \
	--master_addr $MASTER_NODE \
	--master_port $PORT \
	--nproc_per_node $N_GPU \
	"$MMDET_PATH/tools/train.py" \
	"$CONFIG_FILE" \
	--work-dir "$WORK_DIR" \
	--launcher pytorch \
	--gpus $N_GPU \
	--seed 0 \
> /tmp/$USER-dist-train-$nodeName.log 2>&1 &