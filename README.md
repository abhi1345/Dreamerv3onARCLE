# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

dreamerv3 on ARCLE

## Instructions

### Method 1: Manual

Get dependencies with python 3.9:
```
pip install -r requirements.txt
```
Run training on ARCLE:
```
python3 dreamer.py --configs arcle --task arcle_traj --logdir ./logdir/arlce
```
Monitor results:
```
tensorboard --logdir ./logdir
```
### Method 2: Docker

Please refer to the Dockerfile for the instructions, as they are included within.

## Acknowledgments
This code is heavily inspired by the following works:
- danijar's Dreamer-v3 jax implementation: https://github.com/danijar/dreamerv3
- danijar's Dreamer-v2 tensorflow implementation: https://github.com/danijar/dreamerv2
- jsikyoon's Dreamer-v2 pytorch implementation: https://github.com/jsikyoon/dreamer-torch
- RajGhugare19's Dreamer-v2 pytorch implementation: https://github.com/RajGhugare19/dreamerv2
- denisyarats's DrQ-v2 original implementation: https://github.com/facebookresearch/drqv2
- NM512's Dreamer-v3 reimplmentation: https://github.com/NM512/dreamerv3-torch
