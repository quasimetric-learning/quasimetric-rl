from __future__ import annotations
from typing import *

import time
import logging

import torch
import torch.utils.data

from quasimetric_rl.modules import QRLConf, QRLAgent, QRLLosses, InfoT
from quasimetric_rl.data import BatchData, Dataset



class Trainer(object):
    agent: QRLAgent
    losses: QRLLosses
    device: torch.device
    dataset: Dataset
    batch_size: int
    dataloader: torch.utils.data.DataLoader

    def __init__(self, *,
                 agent_conf: QRLConf,
                 device: torch.device,
                 dataset: Dataset,
                 batch_size: int,
                 total_optim_steps: int,
                 dataloader_kwargs: Dict[str, Any] = {}):

        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device

        self.agent, self.losses = agent_conf.make(
            env_spec=dataset.env_spec,
            total_optim_steps=total_optim_steps)
        self.agent.to(device)
        self.losses.to(device)

        logging.info('Agent:\n\t' + str(self.agent).replace('\n', '\n\t') + '\n\n')
        logging.info('Losses:\n\t' + str(self.losses).replace('\n', '\n\t') + '\n\n')

        self.dataloader = dataset.get_dataloader(
            batch_size=batch_size,
            **dataloader_kwargs,
        )

    @property
    def num_batches(self):
        return len(self.dataloader)

    def iter_training_data(self) -> Iterator[Tuple[BatchData, InfoT]]:
        r"""
        Yield data to train on for each optimization iteration.

        yield (
            data,
            info,
        )
        """
        data_t0 = time.time()
        data: BatchData
        for data in self.dataloader:
            data = data.to(self.device)
            yield data, dict(data_time=time.time() - data_t0)
            data_t0 = time.time()

    def train_step(self, data: BatchData, *, optimize: bool = True) -> InfoT:
        return self.losses(self.agent, data, optimize=optimize).info
