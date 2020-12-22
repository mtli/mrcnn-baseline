#!/bin/bash
absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV=$1
MMDET_PATH=$2
CONFIG_FILE=$3
WORK_DIR=$4
N_NODE=$5
NODE_LIST=$6
PORT=$7


rank=0
# select input device 10 to prevent conflicting with ssh
while read -u10 nodeInfo; do
	strArray=($nodeInfo)
	nodeName=${strArray[0]}
	nGPU=${strArray[1]}
	if [ $rank -eq 0 ]; then
		masterNode=$nodeName
		masternGPU=$nGPU
		currentNode=$(hostname)
		if [ "$currentNode" != "$masterNode" ]; then
			echo "Please run this script on the master node ($masterNode)"
			exit 1
		fi
	else
		ssh $nodeName bash "$absDir/../dispatch/run_slave_node.sh" \
			$CONDA_ENV $N_NODE $rank $masterNode $PORT $nGPU \
			"$MMDET_PATH" "$CONFIG_FILE" "$WORK_DIR"
	fi
	rank=$((rank+1))
done 10< "$NODE_LIST"

if [ $rank -ne $N_NODE ]; then
	echo "N_NODE ($N_NODE) mismatches with the number of nodes ($rank) in the node list ($NODE_LIST)"
	echo "Please use dispatch/kill_zombies.sh to kill zombie processes before running this script again"
	exit 1
fi

. activate $CONDA_ENV

# cd for getting the mmdet git tag
cd "$MMDET_PATH"

echo "Starting script on $masterNode (master)"
python -m torch.distributed.launch \
	--nnodes $N_NODE \
	--node_rank 0 \
	--master_addr $masterNode \
	--master_port $PORT \
	--nproc_per_node $masternGPU \
	"$MMDET_PATH/tools/train.py" \
	"$CONFIG_FILE" \
	--work-dir "$WORK_DIR" \
	--launcher pytorch \
	--gpus $nGPU \
	--seed 0 \


	# "$absDir/../train.py" \
