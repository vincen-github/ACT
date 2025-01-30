# Unsupervised Transfer Learning via Adversarial Contrastive Training

Official repository of the paper **Unsupervised Transfer Learning via Adversarial Contrastive Training** 

## Organization
The checkpoints are stored in the `data` directory every 100 epochs during training. All reported results in the paper can be reproduced using the `.pt` files in the `model` directory. The `dataset` directory contains the code for reading various datasets, while the `eval` directory includes code related to evaluation. The `method` directory contains the implementation of ACT, and `model.py` defines the encoder structure.

## Supported Models
- ACT(Ours)
- BT [arXiv](https://arxiv.org/abs/2103.03230)
- BS [arXiv](https://arxiv.org/abs/2204.02683)

## Supported Datasets
- CIFAR-10 
- CIFAR-100
- Tiny ImageNet

## Results
| Method | CIFAR-10 (Linear) | CIFAR-10 (k-nn) | CIFAR-100 (Linear) | CIFAR-100 (k-nn) | Tiny ImageNet (Linear) | Tiny ImageNet (k-nn) |
|--------|--------------------|------------------|---------------------|------------------|-------------------------|-----------------------|
| BT     | 83.96              | 81.18            | 56.75               | 47.91            | 34.08                   | 19.40                 |
| BS     | 86.95              | 82.83            | 53.75               | 48.40            | 35.80                   | 20.36                 |
| ACT    | **92.11**          | **90.01**        | **68.24**           | **58.35**        | **49.72**               | **36.40**             |


## Installation
All experiments were conducted using a single Tesla V100 GPU unit. The torch version is 2.2.1+cu118 and the CUDA version is 11.8.

#### Tiny ImageNet
To reproduce the results presented in the repository, acquiring Tiny ImageNet from [this repo](https://github.com/tjmoon0104/pytorch-tiny-imagenet) is necessary. Otherwise, the model is unlikely to reach a top-1 accuracy of 1% by the end of training.

## Usage

Detailed settings are good by default, to see all options:
```
python -m train --help
python -m test --help
```

Use following commands to run ACT across various datasets:
#### ACT
```
python -m train --dataset cifar10 --lr 3e-3 --emb 64
python -m train --dataset cifar100 --lr 3e-3 --emb 64
python -m train --dataset tiny_in --lr 2e-3 --emb 128
```

#### BT
```
python -m train --dataset cifar10 --lr 3e-3 --bs 256 --emb 512 --method barlow_twins
python -m train --dataset cifar100 --lr 3e-3 --bs 256 --emb 512 --method barlow_twins
python -m train --dataset tiny_in --lr 2e-3 --bs 256 --emb 512 --method barlow_twins
```

#### BS
```
python -m train --dataset cifar10 --lr 3e-3 --emb 512 --method haochen22
python -m train --dataset cifar100 --lr 3e-3 --emb 512 --method haochen22
python -m train --dataset tiny_in --lr 2e-3 --emb 512 --method haochen22
```




## Acknowledgement
This implementation is based on [this repo](https://github.com/htdt/self-supervised).

## Citation
```
```
