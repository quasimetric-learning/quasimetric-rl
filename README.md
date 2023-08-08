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
torch>=1.13.1
tqdm
numpy>=1.17.0
imageio==2.6.1
matplotlib
gym==0.18.0
tensorboardX>=2.5
attrs>=21.4.0
hydra-core==1.3.2
omegaconf==2.3.0
d4rl==1.1
mujoco==2.3.6
```

**NOTE:** Both `d4rl` depends on `mujoco_py` which can be difficult to install. The code lazily imports `mujoco_py` and  `d4rl` if the user requests such environments. Therefore, their installation is not necessary to run the QRL algorithm, e.g., on a custom environment. However, running QRL on the provided environments (`d4rl.maze2d` and `GCRL`) requires them.

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

# run medium maze with custom seed, the GPU at index 2, and not training an actor
./offline/run_maze2d.sh env.name='maze2d-medium-v1' seed=12131415 device.index=2 agent.actor=null

# run large maze with custom seed, the GPU at index 3, and 100 gradient steps
./offline/run_maze2d.sh env.name='maze2d-large-v1' seed=44411223 device.index=3 total_optim_steps=100
```

To reproduce the online `gcrl`  experiments in paper, you can use commands similar to these:

```sh
# run state-input FetchReach
./online/run_gcrl.sh env.name='FetchReach'

# run image-input FetchPush with custom seed and the GPU at index 2
./online/run_gcrl.sh env.name='FetchPushImage' seed=12131415 device.index=2

# run state-input FetchSlide with custom seed, 10 environment steps, and 3 critics
./online/run_gcrl.sh env.name='FetchSlide' seed=44411223 interaction.total_env_steps=10 agent.num_critics=3
```

**NOTES**:
1. **We recommend monitoring experiments with tensorboard.**

2. **(Offline Only) if you do not want to train an actor** (e.g., because the action space is discrete and the code only implements policy training via backpropagating through quasimetric critics), add `agent.actor=null`.

3. **Environment flag `QRL_DEBUG=1`** will enable additional checks and automatic `pdb.post_mortem`. It is your debugging friend.

4. **Adding environments** can be done via `quasimetric_rl.data.register_(online|offline)_env`. See their docstrings for details. To construct an `quasimetric_rl.data.EpisodeData` from a  trajectory, see the `EpisodeData.from_simple_trajectory` helper constructor.

<detail>
<summary>
Example code for how to load a trained checkpoint
<summary>
```py
import os
import torch
import quasimetric_rl
from omegaconf import OmegaConf, SCMode
import yaml

is_offline: bool = True  # change to False if loading online agents

if is_offline:
    from offline.main import Conf
else:
    from offline.main import Conf


expr_checkpoint = '/xxx/xx/xx/xxxx_final.pth'  # FIXME


expr_dir = os.path.dirname(expr_checkpoint)
with open(expr_dir + '/config.yaml', 'r') as f:
    # load saved conf
    dict_conf = OmegaConf.merge(OmegaConf.structured(Conf()), yaml.safe_load(f))
    # convert to object format
    conf: Conf = OmegaConf.to_container(dict_conf, structured_config_mode=SCMode.INSTANTIATE)


# 1. How to create env
if not is_offline:
    # we are not training... skip the long initialization of replay buffer
    conf.env.init_num_transitions = 1

dataset = conf.env.make()
env = dataset.create_env()  # <-- you can use this now!
# episodes = list(dataset.load_episodes())  # if you want to load episodes for offline data


# 2. How to re-create QRL agent
agent: quasimetric_rl.modules.QRLAgent = conf.agent.make(
  env_spec=rb.env_spec, total_optim_steps=1)[0]

```
</detail>

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
