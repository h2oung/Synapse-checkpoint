# DPipe: Dynamic Programming-Based Model Partitioning for Efficient Knowledge Distillation Pipelines

**Author**: Eunjin Lee  
**Date**: June 11, 2025


## Overview
This project introduces **DPipe**, a profiling-informed model partitioning algorithm designed to enhance the efficiency of **pipeline parallel training** for **Knowledge Distillation (KD)**. Unlike conventional partitioning approaches, our work considers the asymmetric execution characteristics of teacher–student pairs by independently profiling both models.

### Key contributions:
- **Profiling Module**: A lightweight PyTorch hook-based profiler that captures per-layer execution time, activation sizes, and parameter footprints.  
- **StageTime Cost Model**: Estimates stage latency using both compute and bandwidth-aware transfer times.  
- **Dynamic Programming Partitioning**: Optimizes pipeline partitioning to minimize imbalance while adhering to GPU memory constraints.


### Highlights
- Improves **pipeline iteration time** by up to **28.45%**.  
- Increases **average GPU utilization** by up to **31.8%** across batch sizes.  
- Validated with **ViT-Large (teacher)** and **ResNet-152 (student)** on multi-GPU setups.  
- Confirms that **system-aware, profiling-based partitioning** is effective for scalable KD pipelines.  


## Pretraining

Train the teacher model on ImageNet100 with ViT-Large:

```bash
cd benchmarks/soft_target
git checkout feature_profiling 
conda activate tspipe
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_base.py \
    --img_root=/data/imagenet2012/imagenet --save_root=./results/base/ \
    --epochs=20 \
    --data_name=imagenet100 \
    --net_name=vit_large \
    --num_class=100 \
    --note=base-i100-vit-large
```

If distributed execution is required:

```bash
torchrun --nproc_per_node=4 train_base.py \
    --img_root=/data/imagenet2012/imagenet --save_root=./results/base/ \
    --epochs=20 \
    --data_name=imagenet100 \
    --net_name=vit_large \
    --num_class=100 \
    --note=base-i100-vit-large
```

## Knowledge Distillation
### 1. Baseline KD Training (TSPIPE)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_kd.py \
    --img_root=/data/imagenet2012/imagenet --save_root=./results/st/ \
    --t_model=./results/base/base-i100-vit-large/model_best.pth.tar \
    --s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar \
    --kd_mode=st --lambda_kd=0.1 --t_name=vit_large --s_name=resnet152 \
    --T=4.0 --data_name=imagenet100 --num_class=100 --batch_size=16 \
    --tspipe-enable --tspipe-config=tspipe.yaml --num-node=1 --rank=0 --ip=localhost \
    --note=kd-run
```

### 2. KD Training with Profiling (DPipe)
```bash 
cd benchmarks/soft_target
git checkout feature_profiling 
conda activate tspipe
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train_kd_profiling.py \
    --img_root=/data/imagenet2012/imagenet --save_root=./results/st/ \
    --t_model=./results/base/base-i100-vit-large/model_best.pth.tar \
    --s_init=./results/base/base-i100-resnet152/initial_r152.pth.tar \
    --kd_mode=st --lambda_kd=0.1 --t_name=vit_large --s_name=resnet152 \
    --T=4.0 --data_name=imagenet100 --num_class=100 --batch_size=128 \
    --tspipe-enable --tspipe-config=tspipe.yaml --num-node=1 --rank=0 --ip=localhost \
    --note=kd-run --epochs=1
```

## Notes
- Ensure the ImageNet2012 path is correctly set.




## Acknowledgement / Citation

This project builds upon the [TSPIPE](https://github.com/kaist-ina/TSPipe) framework.  
Please cite the original authors when using or extending this work:

> TSPipe: Efficient Pipeline Parallel Training for Knowledge Distillation  
> [https://github.com/kaist-ina/TSPipe](https://github.com/kaist-ina/TSPipe)

The original LICENSE and CITATION files from the TSPipe project are retained in the repository root and should be followed accordingly.  
