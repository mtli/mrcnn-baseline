#!/bin/bash

NODE_LIST=$1
SCRIPT_NAME=${2:-train.py}


absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

# select input device 10 to prevent conflicting with ssh
while read -u10 nodeInfo; do
	strArray=($nodeInfo)
	nodeName=${strArray[0]}
	ssh $nodeName bash "$absDir/kill_single_node.sh" "$SCRIPT_NAME"
done 10< "$1"
