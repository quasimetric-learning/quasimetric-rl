from typing import *

import os

import glob
import attrs
import yaml
import logging
import socket
import subprocess
import tempfile
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import OmegaConf, SCMode, DictConfig

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing

from tensorboardX import SummaryWriter

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from .trainer import Trainer



@attrs.define(kw_only=True, auto_attribs=True)
class DeviceConfig:
    type: str = 'cuda'
    index: Optional[int] = 0

    def make(self):
        return torch.device(self.type, self.index)


@utils.singleton
@attrs.define(kw_only=True)
class Conf:
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
    resume_if_possible: bool = True
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    @output_base_dir.validator
    def exists_path(self, attribute, value):
        assert os.path.exists(value)

    output_folder: Optional[str] = None

    @property
    def output_dir(self) -> str:
        return os.path.join(self.output_base_dir, self.output_folder)

    device: DeviceConfig = DeviceConfig()

    # Seeding
    seed: int = 60912

    # Env
    env: quasimetric_rl.data.Dataset.Conf = quasimetric_rl.data.Dataset.Conf()

    # Agent
    agent: quasimetric_rl.modules.QRLConf = quasimetric_rl.modules.QRLConf()

    batch_size: int = attrs.field(default=4096, validator=attrs.validators.gt(0))
    num_workers: int = attrs.field(default=8, validator=attrs.validators.ge(0))

    total_optim_steps: int = attrs.field(default=int(2e5), validator=attrs.validators.gt(0))
    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))


    def finalize(self):
        if self.output_folder is None:
            specs = [
                self.agent.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec,
                f'dyn={self.agent.quasimetric_critic.losses.latent_dynamics.weight:g}',
            ]
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
            self.output_folder = os.path.join(
                f'{self.env.kind}_{self.env.name}',
                '_'.join(specs),
            )
        utils.mkdir(self.output_dir)



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = OmegaConf.to_container(dict_cfg, structured_config_mode=SCMode.INSTANTIATE)
    cfg.finalize()

    utils.logging.configure(os.path.join(cfg.output_dir, 'output.log'))
    completion_file = os.path.join(cfg.output_dir, 'COMPLETE')
    if os.path.exists(completion_file):
        if cfg.overwrite_output:
            logging.warning(f'Overwriting output directory {cfg.output_dir}')
        else:
            raise RuntimeError(f'Output directory {cfg.output_dir} exists and is complete')

    torch.backends.cudnn.benchmark = True

    # Log config
    logging.info('')
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info('')
    logging.info(f'Running on {socket.getfqdn()}:')
    logging.info(f'\t{"PID":<30}{os.getpid()}')
    for var in ['CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID']:
        logging.info(f'\t{var:<30}{os.environ.get(var, None)}')
    logging.info('')
    logging.info(f'Output directory {cfg.output_dir}')
    logging.info('')

    with open(os.path.join(cfg.output_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    logging.info('')
    logging.info(f'Base Git directory {cfg.base_git_dir}')
    logging.info(f'Git COMMIT: {cfg.git_commit}')
    logging.info(f'Git status:\n    ' + '\n    '.join(cfg.git_status))
    with open(os.path.join(cfg.output_dir, 'git_summary.yaml'), 'w') as f:
        f.write(yaml.safe_dump(dict(
            base_dir=cfg.base_git_dir,
            commit=cfg.git_commit,
            status=cfg.git_status,
        )))
    with open(os.path.join(cfg.output_dir, f'git_{cfg.git_commit}.patch'), 'w') as f:
        f.write(subprocess.getoutput(f'git diff {cfg.git_commit}'))
    logging.info('')

    writer = SummaryWriter(cfg.output_dir)

    # Seeding
    torch_seed, np_seed = utils.split_seed(cast(int, cfg.seed), 2)
    np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
    torch.manual_seed(np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31))

    dataset = cfg.env.make()

    # trainer
    dataloader_kwargs = dict(shuffle=True, drop_last=True)
    if cfg.num_workers > 0:
        torch.multiprocessing.set_forkserver_preload(["torch", "quasimetric_rl"])
        dataloader_kwargs.update(
            num_workers=cfg.num_workers,
            multiprocessing_context=torch.multiprocessing.get_context('forkserver'),
            persistent_workers=True,
        )

    device = cfg.device.make()

    if cast(torch.device, device).type == 'cuda':
        pin_memory_device = 'cuda'  # DataLoader only allows string... lol
        if device.index is not None:
            pin_memory_device += f':{device.index}'
        dataloader_kwargs.update(
            pin_memory=True,
            pin_memory_device=pin_memory_device,
        )

    trainer = Trainer(
        agent_conf=cfg.agent,
        device=cfg.device.make(),
        dataset=dataset,
        batch_size=cfg.batch_size,
        total_optim_steps=cfg.total_optim_steps,
        dataloader_kwargs=dataloader_kwargs,
    )

    # save, load, and resume
    def save(epoch, it, *, suffix=None, extra=dict()):
        desc = f"{epoch:05d}_{it:05d}"
        if suffix is not None:
            desc += f'_{suffix}'
        utils.mkdir(cfg.output_dir)
        fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
        state_dicts = dict(
            epoch=epoch,
            it=it,
            agent=trainer.agent.state_dict(),
            losses=trainer.losses.state_dict(),
            **extra,
        )
        torch.save(state_dicts, fullpath)
        relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
        logging.info(f"Checkpointed to {relpath}")

    def load(ckpt):
        state_dicts = torch.load(ckpt, map_location='cpu')
        trainer.agent.load_state_dict(state_dicts['agent'])
        trainer.losses.load_state_dict(state_dicts['losses'])
        relpath = os.path.join('.', os.path.relpath(ckpt, os.path.dirname(__file__)))
        logging.info(f"Loaded from {relpath}")


    ckpts = {}  # (epoch, iter) -> path
    for ckpt in sorted(glob.glob(os.path.join(glob.escape(cfg.output_dir), 'checkpoint_*.pth'))):
        epoch, it = os.path.basename(ckpt).rsplit('.', 1)[0].split('_')[1:3]
        epoch, it = int(epoch), int(it)
        ckpts[epoch, it] = ckpt

    if cfg.resume_if_possible and len(ckpts) > 0:
        start_epoch, start_it = max(ckpts.keys())
        logging.info(f'Load from existing checkpoint: {ckpts[start_epoch, start_it]}')
        load(ckpts[start_epoch, start_it])
        logging.info(f'Fast forward to epoch={start_epoch} iter={start_it}')
    else:
        start_epoch, start_it = 0, 0


    # step counter to keep track of when to save
    step_counter = StepsCounter(
        alert_intervals=dict(
            log=cfg.log_steps,
            save=cfg.save_steps,
        ),
    )
    num_total_epochs = int(np.ceil(cfg.total_optim_steps / trainer.num_batches))

    # Training loop
    optim_steps = 0

    def log_tensorboard(optim_steps, info: InfoT, prefix: str):  # logging helper
        for k, v in info.items():
            if isinstance(v, Mapping):
                log_tensorboard(optim_steps, v, prefix=f"{prefix}{k}/")
                continue
            if isinstance(v, torch.Tensor):
                v = v.mean().item()
            writer.add_scalar(f"{prefix}{k}", v, optim_steps)

    save(0, 0)
    if start_epoch < num_total_epochs:
        for epoch in range(num_total_epochs):
            epoch_desc = f"Train epoch {epoch:05d}/{num_total_epochs:05d}"
            for it, (data, data_info) in enumerate(tqdm(trainer.iter_training_data(), total=trainer.num_batches, desc=epoch_desc)):
                step_counter.update_then_record_alerts()
                optim_steps += 1

                if (epoch, it) <= (start_epoch, start_it):
                    continue  # fast forward
                else:
                    iter_t0 = time.time()
                    train_info = trainer.train_step(data)
                    iter_time = time.time() - iter_t0

                if step_counter.alerts.save:
                    save(epoch, it)

                if step_counter.alerts.log:
                    log_tensorboard(optim_steps, data_info, 'data/')
                    log_tensorboard(optim_steps, train_info, 'train_')
                    writer.add_scalar("train/iter_time", iter_time, optim_steps)

    save(num_total_epochs, 0, suffix='final')
    open(completion_file, 'a').close()


if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()
