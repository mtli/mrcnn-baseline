#!/bin/bash
# The EX version takes explicit GPU IDs as input

absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV=$1
MMDET_PATH=$2
CONFIG_FILE=$3
WORK_DIR=$4
COMPUTE_LIST=$5
PORT=$6
RESUME_FROM=$7

nNode=0
# count the number of nodes
# select input device 10 to prevent conflicting with ssh
while read -u10 nodeInfo; do
	nNode=$((nNode+1))
done 10< "$COMPUTE_LIST"

rank=0
while read -u10 nodeInfo; do
	strArray=($nodeInfo)
	nodeName=${strArray[0]}
	GPUIDs=${strArray[1]}
	if [ $rank -eq 0 ]; then
		masterNode=$nodeName
		masterGPUIDs=$GPUIDs
		currentNode=$(hostname)
		if [ "$currentNode" != "$masterNode" ]; then
			echo "Please run this script on the master node ($masterNode)"
			exit 1
		fi
	else
		ssh $nodeName bash "$absDir/../dispatch/run_slave_node_ex.sh" \
			$CONDA_ENV $nNode $rank $masterNode $PORT $GPUIDs \
			"$MMDET_PATH" "$CONFIG_FILE" "$WORK_DIR" "$RESUME_FROM"
	fi
	rank=$((rank+1))
done 10< "$COMPUTE_LIST"

s="${masterGPUIDs//[^,]}"
nc="${#s}"
nGPU=$((nc+1))
export CUDA_VISIBLE_DEVICES=$masterGPUIDs

. activate $CONDA_ENV

# cd for getting the mmdet git tag
cd "$MMDET_PATH"

echo "Starting script on $masterNode ($masterGPUIDs, master)"
python -m torch.distributed.launch \
	--nnodes $nNode \
	--node_rank 0 \
	--master_addr $masterNode \
	--master_port $PORT \
	--nproc_per_node $nGPU \
	"$MMDET_PATH/tools/train.py" \
	"$CONFIG_FILE" \
	--work-dir "$WORK_DIR" \
	--launcher pytorch \
	--seed 0 \
	--resume-from "$RESUME_FROM" \
