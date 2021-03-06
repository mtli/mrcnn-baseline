#!/bin/bash
absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

####### Modify the parameters and run this script on the master node ######
CONDA_ENV=d24
MMDET_PATH="$HOME/repo/mmdetection2.4"
CONFIG_FILE="$absDir/../configs/mask_rcnn_r50_fpn_2x_coco.py"
WORK_DIR="/data/mengtial/Exp/COCO/mrcnn_standard"
N_NODE=2
NODE_LIST="$absDir/../exp/node-list-1.txt"
PORT=40034
RESUME_FROM="" # Resume from interruption

###########################################################################

sh "$absDir/../dispatch/dist_launch.sh" \
	"$CONDA_ENV" "$MMDET_PATH" "$CONFIG_FILE" \
	"$WORK_DIR" $N_NODE "$NODE_LIST" $PORT \
	"$RESUME_FROM" \
