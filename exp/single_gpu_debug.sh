#!/bin/bash
absDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

####### Modify the parameters and run this script on the master node ######
CONDA_ENV=d24
MMDET_PATH="$HOME/repo/mmdetection2.4"
CONFIG_FILE="$absDir/../configs/mask_rcnn_r50_fpn_lsj_2x_coco.py"
WORK_DIR="/data2/mengtial/Exp/COCO/debug"

###########################################################################

cd $MMDET_PATH

python "$MMDET_PATH/tools/train.py" \
	"$CONFIG_FILE" \
	--work-dir "$WORK_DIR" \
	--launcher none \
	--gpus 1 \
	--seed 0 \
