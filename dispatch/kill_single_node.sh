#!/bin/bash

SCRIPT_NAME=$1

nodeName=$HOSTNAME

prev_proc=`pgrep -u $(whoami) -af python | grep $SCRIPT_NAME | cut -d' ' -f 1`

if [ -n "$prev_proc" ]; then
	echo Killing zombie procs on $nodeName:
	echo $prev_proc
	kill -9 $prev_proc
fi
