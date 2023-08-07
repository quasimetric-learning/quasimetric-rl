# Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning

**[Tongzhou Wang](https://tongzhouwang.info/), [Antonio Torralba](https://web.mit.edu/torralba/www/), [Phillip Isola](https://web.mit.edu/phillipi/), [Amy Zhang](https://amyzhang.github.io/)**

This repository is the official code release for paper [Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://www.tongzhouwang.info/quasimetric_rl/), published in ICML 2023. We provide a PyTorch implementation of the proposed Quasimetric RL algorithm (QRL).

+ [arXiv](https://arxiv.org/abs/2304.01203)
+ [Project Page](https://www.tongzhouwang.info/quasimetric_rl/)

## Requirements
The code has been tested on

+ CUDA 11 with NVIDIA RTX Titan, NVIDIA 2080Ti, NVIDIA Titan XP, NVIDIA V100, and NVIDIA 3080.

Software dependencies (also in [requirements.txt](./requirements.txt)):

```
torch==1.13.1
tqdm
numpy>=1.17.0
gym==0.18.0
tensorboardX>=2.5
attrs>=21.4.0
hydra-core==1.3.2
omegaconf==2.3.0
d4rl==1.1
mujoco==2.3.6
```

## Code structure

+ `quasimetric_rl.modules` implements the actor and critic components, as well as their associated QRL losses.
+ `quasimetric_rl.data` implements data loading and memory buffer utilities, as well as creation of environments.
+ `online.main` provides an entry point to online experiments.
+ `offline.main` provides an entry point to offline experiments.

Online and offline settings mostly differ in the usage of data storage:
+ Offline: static dataset.
+ Online: replay buffer that dynamically grows and stores more experiences.

In both `online.main` and `offline.main`, there is a `Conf` object, containing all the provided knobs you can customize QRL behavior. This `Conf` object is updated with commandline arguments via `hydra`, and then used to create the modules and losses.

## Examples

To reproduce the offline `d4rl`  experiments in paper, you can use commands similar to these:

```sh
# run umaze seed=12131415 device.index=2
./offline/run_maze2d.sh env.name='maze2d-umaze-v1'

# run medium maze with custom seed and the GPU at index 2
./offline/run_maze2d.sh env.name='maze2d-medium-v1' seed=12131415 device.index=2

# run large maze with custom seed and the GPU at index 3
./offline/run_maze2d.sh env.name='maze2d-large-v1' seed=44411223 device.index=3
```

To reproduce the online `gcrl`  experiments in paper, you can use commands similar to these:

```sh
# run state-input FetchReach
./online/run_gcrl.sh env.name='FetchReach'

# run image-input FetchPush with custom seed and the GPU at index 2
./online/run_gcrl.sh env.name='FetchPushImage' seed=12131415 device.index=2

# run state-input FetchSlide with custom seed
./online/run_gcrl.sh env.name='FetchSlide' seed=44411223
```

**We recommend monitoring experiments with tensorboard.**

**(Offline Only) if you do not want to train an actor** (e.g., because the action space is discrete and the code only implements policy training backpropagating through quasimetric critics), add `agent.actor=null`.

## Citation
Tongzhou Wang, Antonio Torralba, Phillip Isola, Amy Zhang. "Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning" International Conference on Machine Learning (ICML). 2023.

```bib
@inproceedings{tongzhouw2023qrl,
  title={Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning},
  author={Wang, Tongzhou and Torralba, Antonio and Isola, Phillip and Zhang, Amy},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  year={2023}
}
```

## Questions

For questions about the code provided in this repository, please open an GitHub issue.

For questions about the paper, please contact Tongzhou Wang (tongzhou _AT_ mit _DOT_ edu).

## License
This repo is under MIT license. Please check [LICENSE](./LICENSE) file.
