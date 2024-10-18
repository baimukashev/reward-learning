# reward-learning
This repo contains source code for 
[Learning Transparent Reward Models via Unsupervised Feature Selection](https://openreview.net/forum?id=2sg4PY1W9d),
Daulet Baimukashev, Gokhan Alcan, Kevin Sebastian Luck, Ville Kyrki, CoRL 2024.

We propose a novel approach to construct compact and transparent reward models from automatically selected state features. These inferred rewards have an explicit form and enable the learning of policies that closely match expert
behavior by training standard reinforcement learning algorithms from scratch. We
validate our method’s performance in various robotic environments with continuous and high-dimensional state spaces.

## Installation

### Software requirements
* Python
* Pytorch
* CUDA
* Jax

Install all the required packages in conda environment by running:

```sh
conda env create -f environment.yml
```

## Data
The data can be downloaded from this [link](https://drive.google.com/drive/folders/1by0v5mVIfiayZ_b03xRjamwzEoHdGzMV?usp=drive_link),

Alternatively, data can be collected by training RL policy

```sh
sh train_expert.sh
```

## Configuration
Configuration files for all environments are located in ```src/cfg/```.

## Learning rewards
Reward learning consists of two steps: extracting feature set and learning feature weights.

1. To find features, run:

```sh
python extract_features.py
```

2. To learn feature weights, run scripts using the examples from:

```sh
run_expertiments.sh
```

3. To test the trained model, run
```sh
sh test_expert.sh
```

## Citation
```bibtex
@inproceedings{
baimukashev2024learning,
title={Learning Transparent Reward Models via Unsupervised Feature Selection},
author={Daulet Baimukashev and Gokhan Alcan and Kevin Sebastian Luck and Ville Kyrki},
booktitle={8th Annual Conference on Robot Learning},
year={2024},
}
```