r"""
A base configuration class that can be useful for setting up both online and
offline experiments.
"""

from typing import *

import os
import abc

import attrs
import yaml
import logging
import socket
import subprocess
import tempfile

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import OmegaConf, DictConfig, SCMode

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

from . import data, modules, utils


@attrs.define(kw_only=True, auto_attribs=True)
class DeviceConfig:
    type: str = 'cuda'
    index: Optional[int] = 0

    def make(self):
        return torch.device(self.type, self.index)


@attrs.define(kw_only=True)
class BaseConf(abc.ABC):
    # Disable hydra working directory creation
    hydra: Dict = dict(
        output_subdir=None,
        job=dict(chdir=False),
        run=dict(dir=tempfile.TemporaryDirectory().name),  # can't disable https://github.com/facebookresearch/hydra/issues/1937
        mode=hydra.types.RunMode.RUN,  # sigh: https://github.com/facebookresearch/hydra/issues/2262
    )

    base_git_dir: str = subprocess.check_output(
        r'git rev-parse --show-toplevel'.split(),
        cwd=os.path.dirname(__file__), encoding='utf-8').strip()
    git_commit: str = subprocess.check_output(
        r'git rev-parse HEAD'.split(),
        cwd=os.path.dirname(__file__), encoding='utf-8').strip()
    git_status: Tuple[str] = tuple(
        l.strip() for l in subprocess.check_output(
            r'git status --short'.split(),
            cwd=os.path.dirname(__file__), encoding='utf-8').strip().split('\n')
        )

    overwrite_output: bool = False

    @property
    @abc.abstractmethod
    def output_base_dir(self) -> str:
        # should be an attribute, but abc doesn't support checking that
        # Subclass should overwrite this
        pass

    output_folder: Optional[str] = None
    output_folder_suffix: Optional[str] = None
    output_dir: Optional[str] = attrs.field(default=None, init=False)

    @property
    def completion_file(self) -> str:
        return os.path.join(self.output_dir, 'COMPLETE')

    device: DeviceConfig = DeviceConfig()

    # Seeding
    seed: int = 60912

    # Env
    @property
    @abc.abstractmethod
    def env(self) -> data.Dataset.Conf:
        # should be an attribute, but abc doesn't support checking that
        # Subclass should overwrite this, with either `data.Dataset.Conf` or `data.online.ReplayBuffer.Conf` (subclass of the former).
        pass

    # Agent
    agent: modules.QRLConf = modules.QRLConf()

    @classmethod
    def from_DictConfig(cls, cfg: DictConfig) -> 'BaseConf':
        return OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)

    def setup_for_expriment(self) -> SummaryWriter:
        r"""
        1. Finalize conf fields
        2. Do basic checks
        3. Setup logging, seeding, etc.
        4. Returns a tensorboard logger
        """

        if self.output_dir is not None:
            raise RuntimeError('setup_for_expriment() can only be called once')

        if self.output_folder is None:
            specs = [
                self.agent.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec,
                f'dyn={self.agent.quasimetric_critic.losses.latent_dynamics.weight:g}',
            ]
            if self.agent.num_critics > 1:
                specs.append(f'{self.agent.num_critics}critic')
            if self.agent.actor is not None:
                aspecs = []
                if self.agent.actor.losses.min_dist.add_goal_as_future_state:
                    aspecs.append('goal=Rand+Future')
                else:
                    aspecs.append('goal=Rand')
                if self.agent.actor.losses.min_dist.adaptive_entropy_regularizer:
                    aspecs.append('ent')
                if self.agent.actor.losses.behavior_cloning.weight > 0:
                    aspecs.append(f'BC={self.agent.actor.losses.behavior_cloning.weight:g}')
                specs.append('actor(' + ','.join(aspecs) + ')')
            specs.append(
                f'seed={self.seed}',
            )
            if self.output_folder_suffix is not None:
                specs.append(self.output_folder_suffix)
            self.output_folder = os.path.join(
                f'{self.env.kind}_{self.env.name}',
                '_'.join(specs),
            )
        assert os.path.exists(self.output_base_dir)
        self.output_dir = os.path.join(self.output_base_dir, self.output_folder)
        utils.mkdir(self.output_dir)

        if os.path.exists(self.completion_file):
            if self.overwrite_output:
                logging.warning(f'Overwriting output directory {self.output_dir}')
            else:
                raise RuntimeError(f'Output directory {self.output_dir} exists and is complete')

        writer = SummaryWriter(self.output_dir)  # tensorboard writer
        utils.logging.configure(os.path.join(self.output_dir, 'output.log'))

        # Log config
        logging.info('')
        logging.info(OmegaConf.to_yaml(self))
        logging.info('')
        logging.info(f'Running on {socket.getfqdn()}:')
        logging.info(f'\t{"PID":<30}{os.getpid()}')
        for var in ['CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID']:
            logging.info(f'\t{var:<30}{os.environ.get(var, None)}')
        logging.info('')
        logging.info(f'Output directory {self.output_dir}')
        logging.info('')

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self))
        writer.add_text('config', f"```\n{OmegaConf.to_yaml(self)}\n```")  # markdown

        logging.info('')
        logging.info(f'Base Git directory {self.base_git_dir}')
        logging.info(f'Git COMMIT: {self.git_commit}')
        logging.info(f'Git status:\n    ' + '\n    '.join(self.git_status))
        with open(os.path.join(self.output_dir, 'git_summary.yaml'), 'w') as f:
            f.write(yaml.safe_dump(dict(
                base_dir=self.base_git_dir,
                commit=self.git_commit,
                status=self.git_status,
            )))
        with open(os.path.join(self.output_dir, f'git_{self.git_commit}.patch'), 'w') as f:
            f.write(subprocess.getoutput(f'git diff {self.git_commit}'))
        logging.info('')

        # Seeding
        torch_seed, np_seed = utils.split_seed(cast(int, self.seed), 2)
        np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
        torch.manual_seed(np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31))

        # PyTorch setup
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(12)

        return writer
