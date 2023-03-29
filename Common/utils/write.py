from typing import Union

import os
import sys
import datetime

import torch
import logging
from termcolor import colored
from tensorboardX import SummaryWriter


class Write:
    def __init__(self, cfg, phase: str='train') -> None:
        self.cfg = cfg
        self.phase = phase
        self.save_dir = cfg.env.save_dir
        self.exper_name = cfg.env.exper_name
        self.base_writer_path = os.path.join(self.save_dir,
                                        self.cfg.exper_name,
                                        '_time-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                                        )
        os.makedirs(self.writer_path, exist_ok=True)

        _build()

    def _build(self):
        self.model_write_path = os.path.join(self.base_writer_path, 'model')
        os.makedirs(self.model_write_path, exist_ok=True)

        self.logger_write_path = os.path.join(self.logger_write_path, 'logs')
        os.makedirs(self.logger_write_path, exist_ok=True)
        self.logger_writer = SummaryWriter(log_dir=os.path.join(self.logger_write_path, self.phase))

    def plot_scalar(self, data, position: str, step: int):
        data: torch.Tensor = data.as_tensor() if hasattr(data, 'as_tensor') else data
        self.logger_writer.add_scalar(position, data, step)

    def plot_2d_images(self, data, position: str, step: int):
        data: torch.Tensor = data.as_tensor() if hasattr(data, 'as_tensor') else data
        data: torch.Tensor = self.max_min_norm(data)
        batch_size = data.size()[0]
        batch_size = batch_size if batch_size < 8 else 8
        self.logger_writer.add_image(position, data[:batch_size], step)

    def max_min_norm(self, data):
        return data if torch.max(data) > 1 else (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def create_logger(self, dist_rank=0, name=""):
        # create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # create formatter
        fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
        color_fmt = (
            colored("[%(asctime)s %(name)s]", "green")
            + colored("(%(filename)s %(lineno)d)", "yellow")
            + ": %(levelname)s %(message)s"
        )

        # create console handlers for master process
        if dist_rank == 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(
                # logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
                logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            )
            logger.addHandler(console_handler)

        # create file handlers
        file_handler = logging.FileHandler(
            os.path.join(self.base_writer_path, f"log_rank{dist_rank}.txt"), mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        # file_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        return logger
