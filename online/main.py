from typing import *

import os

import attrs
import logging
import json
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

import torch
import torch.backends.cudnn
import torch.multiprocessing

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer, InteractionConf


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    env: quasimetric_rl.data.online.ReplayBuffer.Conf = quasimetric_rl.data.online.ReplayBuffer.Conf()

    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    interaction: InteractionConf = InteractionConf()

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    eval_steps: int = attrs.field(default=2000, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())


@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    writer = cfg.setup_for_expriment()  # checking & setup logging

    replay_buffer = cfg.env.make()

    trainer = Trainer(
        agent_conf=cfg.agent,
        device=cfg.device.make(),
        replay=replay_buffer,
        batch_size=cfg.batch_size,
        interaction_conf=cfg.interaction,
    )

    val_results: List[dict] = []
    val_summaries: List[dict] = []

    def save(env_steps, optim_steps, *, suffix=None, extra=dict()):
        desc = f"env{env_steps:08d}_opt{optim_steps:08d}"
        if suffix is not None:
            desc += f'_{suffix}'
        utils.mkdir(cfg.output_dir)
        fullpath = os.path.join(cfg.output_dir, f'checkpoint_{desc}.pth')
        state_dicts = dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            agent=trainer.agent.state_dict(),
            losses=trainer.losses.state_dict(),
            val_result=val_results[-1] if len(val_results) else None,
            val_summaries=val_summaries,
            **extra,
        )
        torch.save(state_dicts, fullpath)
        relpath = os.path.join('.', os.path.relpath(fullpath, os.path.dirname(__file__)))
        logging.info(f"Checkpointed to {relpath}")

    def eval(env_steps, optim_steps):
        val_result = trainer.evaluate()
        val_results.clear()
        val_results.append(dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            result=attrs.asdict(val_result),
        ))
        epi_return = val_result.episode_return
        succ_rate = val_result.is_success
        succ_rate_ts = val_result.timestep_is_success.mean(dtype=torch.float32, dim=-1)
        hitting_time = torch.where(
            val_result.hitting_time < 0, trainer.replay.episode_length + 1, val_result.hitting_time,
        )
        val_summaries.append(dict(
            env_steps=env_steps,
            optim_steps=optim_steps,
            epi_return=epi_return,
            succ_rate_ts=succ_rate_ts,
            succ_rate=succ_rate,
            hitting_time=hitting_time,
        ))
        for k, v in val_summaries[-1].items():
            if k == 'env_steps':
                continue
            if isinstance(v, torch.Tensor):
                v = v.to(torch.float64).mean().item()
            writer.add_scalar(f"eval/{k}", v, env_steps)
        with open(os.path.join(cfg.output_dir, 'eval.log'), 'a') as f:
            print(
                json.dumps({
                    k: (v.to(torch.float64).mean().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in val_summaries[-1].items()
                }),
                file=f,
            )
        logging.info(
            f"EVAL: " +
            "  ".join([
                f"env_steps={env_steps}",
                f"optim_steps={optim_steps}",
                f"succ_rate={succ_rate.to(torch.float64).mean().item():.2%}",
                f"epi_return={epi_return.to(torch.float64).mean().item():.4f}",
            ])
        )


    # step counter to keep track of when to save or eval
    step_counter = StepsCounter(
        alert_intervals=dict(
            log=cfg.log_steps,
            save=cfg.save_steps,
            eval=cfg.eval_steps,
        ),
    )

    def log_tensorboard(env_steps, info: InfoT, prefix: str):
        for k, v in info.items():
            if isinstance(v, Mapping):
                log_tensorboard(env_steps, v, prefix=f"{prefix}{k}/")
                continue
            if isinstance(v, torch.Tensor):
                v = v.mean().item()
            writer.add_scalar(f"{prefix}{k}", v, env_steps)

    # Training loop
    eval(0, 0); save(0, 0)
    for optim_steps, (env_steps, next_iter_new_env_step, data, data_info) in enumerate(trainer.iter_training_data(), start=1):

        iter_t0 = time.time()
        train_info = trainer.train_step(data)
        iter_time = time.time() - iter_t0

        # bookkeep
        if not next_iter_new_env_step:
            continue  # just train more, only eval/log/save right before new env step
        step_counter.update_to_then_record_alerts(env_steps)

        if step_counter.alerts.eval:
            eval(env_steps, optim_steps)

        if step_counter.alerts.save:
            save(env_steps, optim_steps)

        if step_counter.alerts.log:
            log_tensorboard(env_steps, data_info, 'data/')
            log_tensorboard(env_steps, train_info, 'train_')
            writer.add_scalar("train/iter_time", iter_time, env_steps)
            writer.add_scalar("train/optim_steps", optim_steps, env_steps)

    eval(trainer.total_env_steps, optim_steps)
    save(trainer.total_env_steps, optim_steps, suffix='final')
    open(cfg.completion_file, 'a').close()


if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()
