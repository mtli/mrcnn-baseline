#!/bin/bash
absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

####### Modify the parameters and run this script on the master node ######
CONDA_ENV=d24
MMDET_PATH="$HOME/repo/mmdetection2.4"
CONFIG_FILE="$absDir/../configs/mask_rcnn_r50_fpn_lsj_2x_coco.py"
WORK_DIR="/data/mengtial/Exp/COCO/mrcnn_lsj_2x"
COMPUTE_LIST="$absDir/../exp/compute-list-2.txt"
PORT=40034
RESUME_FROM="$WORK_DIR/latest.pth" # Resume from interruption

###########################################################################

sh "$absDir/../dispatch/dist_launch_ex.sh" \
	"$CONDA_ENV" "$MMDET_PATH" "$CONFIG_FILE" \
	"$WORK_DIR" "$COMPUTE_LIST" $PORT \
	"$RESUME_FROM" \

