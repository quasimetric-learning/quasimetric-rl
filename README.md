# Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning

**[Tongzhou Wang](https://tongzhouwang.info/), [Antonio Torralba](https://web.mit.edu/torralba/www/), [Phillip Isola](https://web.mit.edu/phillipi/), [Amy Zhang](https://amyzhang.github.io/)**

This repository is the official code release for paper [Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://www.tongzhouwang.info/quasimetric_rl/), published in ICML 2023. We provide a PyTorch implementation of the proposed Quasimetric RL algorithm (QRL).

+ [arXiv](https://arxiv.org/abs/2304.01203)
+ [Project Page](https://www.tongzhouwang.info/quasimetric_rl/)


## Quasimetric RL (QRL) Objective

$%Please view this section in browser
%
\textsf{Learning the {\color[RGB]{230,97,0}quasimetric geometry}: {\color[RGB]{199,61,160}local} costs} \rightarrow \textsf{{\color{teal}global} optimal paths}
$
```math
%Please view this section in browser

\underbrace{\max_{\theta}~\mathbb{E}_{\substack{s\sim p_\mathsf{state}\\g \sim p_\mathsf{goal}}}[{
\overbrace{d_\theta}^{\color[RGB]{230,97,0}\llap{\textsf{quasimetr}}\rlap{\textsf{ic model}}}}(s, g)]}_{\textsf{push apart {\color{teal}all state-goal pairs}}}
\quad\quad \text{subject to}\qquad
\underbrace{\mathbb{E}_{\substack{(s, a, s', \mathsf{cost}) \sim p_\mathsf{transition}}}[ \mathtt{relu}(
d_\theta(s, s') - \mathsf{cost}
)^2] \leq
{
\overbrace{
\epsilon^2
}^{\color{gray}\llap{\epsilon\textsf{ is a }}\rlap{\textsf{small positive constant}}}
}}_{\textsf{not overestimate observed {\color[RGB]{199,61,160}local} distances/costs}}\tag{QRL}
```

See [webpage](https://www.tongzhouwang.info/quasimetric_rl/) for explanation.



## Requirements
The code has been tested on

+ CUDA 11 with NVIDIA RTX Titan, NVIDIA 2080Ti, NVIDIA Titan XP, NVIDIA V100, and NVIDIA 3080.

Software dependencies:

https://github.com/quasimetric-learning/quasimetric-rl/blob/4f113239d7881eaba47844de7224a3c5736a4b6f/requirements.txt#L1-L10

> [!NOTE]
>
> `d4rl` depends on `mujoco_py` which can be difficult to install. The code lazily imports `mujoco_py` and  `d4rl` if the user requests such environments. Therefore, their installation is not necessary to run the QRL algorithm, e.g., on a custom environment. However, running QRL on the provided environments (`d4rl.maze2d` and `GCRL`) requires them.

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

<details>
<summary><strong>
Example code for how to load a trained checkpoint (click me)
</strong></summary>

```py
import os
import torch
from omegaconf import OmegaConf, SCMode
import yaml

from quasimetric_rl.data import Dataset
from quasimetric_rl.modules import QRLAgent, QRLConf


expr_checkpoint = '/xxx/xx/xx/xxxx.pth'  # FIXME


expr_dir = os.path.dirname(expr_checkpoint)
with open(expr_dir + '/config.yaml', 'r') as f:
    # load saved conf
    conf = OmegaConf.create(yaml.safe_load(f))


# 1. How to create env
dataset: Dataset = Dataset.Conf(kind=conf.env.kind, name=conf.env.name).make(dummy=True)  # dummy: don't load data
env = dataset.create_env()  # <-- you can use this now!
# episodes = list(dataset.load_episodes())  # if you want to load episodes for offline data


# 2. How to re-create QRL agent
agent_conf: QRLConf = OmegaConf.to_container(
  OmegaConf.merge(OmegaConf.structured(QRLConf()), conf.agent),  # overwrite with loaded conf
  structured_config_mode=SCMode.INSTANTIATE,  # create the object
)
agent: QRLAgent = agent_conf.make(env_spec=dataset.env_spec, total_optim_steps=1)[0]  # you can move to your fav device


# 3. Load checkpoint
agent.load_state_dict(torch.load(expr_checkpoint, map_location='cpu')['agent'])
```
</details>

> [!NOTE]
> 1. **We recommend monitoring experiments with tensorboard.**
> 2. **[Offline Only] if you do not want to train an actor** (e.g., because the action space is discrete and the code only implements policy training via backpropagating through quasimetric critics), add `agent.actor=null`.
> 3. **Environment flag `QRL_DEBUG=1`** will enable additional checks and automatic `pdb.post_mortem`. It is your debugging friend.
> 4. **Adding environments** can be done via `quasimetric_rl.data.register_(online|offline)_env`. See their docstrings for details. To construct an `quasimetric_rl.data.EpisodeData` from a  trajectory, see the `EpisodeData.from_simple_trajectory` helper constructor.

## FAQ

**Q:** How to run QRL where the goal is not a single state? <br>
**A:** If more than one state are considered as "reaching a goal", then we can think of the goal as _a set of states_. In this case, we can use the trick discussed in paper Appendix A: (1) Encode this goal as a tensor of the same format as states (but distinct from them, e.g., via an added indicator dimension). (2) Add transitions (<ins>state that reaches goal</ins> -> <ins>goal</ins>) whenever the agent reaches the goal. QRL can extend to such general goals in this way. This can be implemented by either modifying the dataset storage and sampling code [more flexible but involved], or changing the environment to append a transition when reaching the goal [simpler]. **Coming soon:** example code on the later approach.

**Q:** How to deal with variable-cost transitions? <br>
**A:** Current code assumes that each transition incurs a fixed cost:
  
  https://github.com/quasimetric-learning/quasimetric-rl/blob/2fd47ba8901d6b0c4713a9167bb2ba9cd615ee43/quasimetric_rl/modules/quasimetric_critic/losses/local_constraint.py#L59-L61

  To support variable-cost transitions, simply modify these lines to use `-data.rewards` as costs. However, you should make sure that your environment/dataset is set up to provide the expected non-positive rewards. We do not check that in current code.

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
