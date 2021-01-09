# Mask R-CNN Baselines (Distributed Training)

This repo aims to reproduce the large-batch-size Mask R-CNN baselines used in [Google Brain's copy-paste paper](https://arxiv.org/abs/2012.07177) \[1\], which has a much higher AP than standard setting. Here the implementation is based on PyTorch and GPU clusters instead of TensorFlow and TPU clusters. All change details from the standard settings are documented in the configs (`configs/*.py`).

\[1\] Ghiasi et al. [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177). arXiv 2020.


## Getting Started

- Install CUDA, pytorch, and mmdetection (tested with CUDA 10.0, PyTorch 1.4, and mmdetection 2.4)
- Copy the `RandomResize` augmentation from `custom_pipelines.py` into `<mmdetection path>/datasets/pipelines/transforms.py`
- Specify compute hardware through `exp/node-list-*.txt` (ssh-able hostnames + number of GPUs) or `exp/compute-list-*.txt` (ssh-able hostnames + specific GPU IDs)
- Point to the path of your copy of COCO data using `data_root` in `configs/*.py`. Depending how good your NFS is, you might want to create a local copy of the dataset on every node to prevent network bottlenecks
- Change the parameters in `exp/lsj_2x.sh` or `exp/lsj_32x.sh` and run it!
- The checkpoints and Tensorboard logs will be saved in the `WORK_DIR` you specified


## Troubleshooting

Note that useful debug information is often buried during distributed training. For example, when you get CUDA, NCCL, PyTorch error messages, most likely it's a bug in your code and has nothing to do with distributed training. Therefore, the first thing to do is to make sure the code runs in the single-GPU mode on every node using `exp/single_gpu_debug.sh`. It's easy to forget to copy data to one of the nodes.

You can also run `exp/standard.sh` to train the standard Mask R-CNN using only two 4-GPU nodes to check your environment.