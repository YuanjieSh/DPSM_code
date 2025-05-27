## Direct Prediction Set Minimization via Bilevel Conformal Classifier Training (DPSM)

This repository contains a implementation of **DPSM**
corresponding to the follow paper:

Yuanjie Shi*, Hooman Shahrokhi*, Xuesong Jia, Xiongzhi Chen, Janardhan Rao Doppa, Yan Yan.
*[Direct Prediction Set Minimization via Bilevel Conformal Classifier Training](
https://openreview.net/forum?id=JL4MRb1bKH)*.
ICML, 2025.

## Overview

Conformal prediction (CP) is a promising uncertainty quantification framework which works as a wrapper around a black-box classifier to construct prediction sets (i.e., subset of candidate classes) with provable guarantees. 
However, standard calibration methods for CP tend to produce large prediction sets which makes them less useful in practice. 
This paper considers the problem of integrating conformal principles into the training process of deep classifiers to directly minimize the size of prediction sets. 
We formulate conformal training as a bilevel optimization problem and propose the {\em Direct Prediction Set Minimization (DPSM)} algorithm to solve it. 
The key insight behind DPSM is to minimize a measure of the prediction set size (upper level) that is conditioned on the learned quantile of conformity scores (lower level). 
We analyze that DPSM has a learning bound of $O(1/\sqrt{n})$ (with $n$ training samples),
while prior conformal training methods based on stochastic approximation for the quantile has a bound of $\Omega(1/s)$ (with batch size $s$ and typically $s \ll \sqrt{n}$).
Experiments on various benchmark datasets and deep models show that DPSM significantly outperforms the best prior conformal training baseline with $20.46\%\downarrow$ in the prediction set size and validates our theory.
## Running instructions

Please run the commands mentioned below to produce results:

# CIFAR100
1. Please remove `srun` from the beginning of commands if you are using lab gpus.
2. Please replace `torchrun --nproc_per_node=4 --nnodes=3 --node_rank=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=10.110.6.36` with `python` if you face `torchrun` related issues and/or also your setup is to use only one gpu from a node.
  
## 0. Train the Base Models only using CE loss
**DenseNet**
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar100.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --train_rule None --baseloss CE --method Baseloss --arc densenet100 --cal_test_CP_score HPS
```
**ResNet**
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar100.py --batch_size 128 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --train_rule None --baseloss CE --method Baseloss --arc resnet110 --cal_test_CP_score HPS
```
## 1. Commands for Ours Experiments
## HPS train, HPS cal-test
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.1 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc densenet100 --train_CP_score HPS --cal_test_CP_score HPS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.05 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc resnet110 --train_CP_score HPS --cal_test_CP_score HPS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1

```
## HPS train, APS cal-test
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.1 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc densenet100 --train_CP_score HPS --cal_test_CP_score APS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.05 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc resnet110 --train_CP_score HPS --cal_test_CP_score APS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1
```
### HPS train, RAPS cal-test
```
srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 300 --base_lr 0.1 --base_lr_schedule 150 225 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.1 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc densenet100 --train_CP_score HPS --cal_test_CP_score RAPS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1

srun torchrun --standalone --nproc_per_node=gpu ./train/train_cifar.py --batch_size 64 --num_epochs 164 --base_lr 0.1 --base_lr_schedule 81 122 --base_momentum 0.9 --base_gamma 0.1 --base_weight_decay 0.0001 --finetune 1 --finetune_batch_size 128 --finetune_epochs 40 --finetune_lr 0.1 --finetune_lr_schedule 25 40 --finetune_momentum 0.9 --finetune_gamma 0.1 --finetune_weight_decay 0.0001 --lr_qr 0.01 --mu_size 0.05 --train_rule None --baseloss CE --method pinball_marginal_with_Inefficiency --arc resnet110 --train_CP_score HPS --cal_test_CP_score RAPS --train_T 1.0 --sigmid_T 1.0 --finetune_CE 1
```