import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer


class BaseTrainer(object):
    def __init__(
        self,
        device: Union[torch.device, str],
        max_epochs: int,
        data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        postprocessing: Optional[Callable] = None,
        key_metric: Optional[Dict[str, Metric]] = None,
        amp: bool = False,
        is_write_logger: bool = False,
    ):
        super().__init__(self._iteration)

        if isinstance(data_loader, DataLoader):
            sampler = data_loader.__dict__["sampler"]

            if epoch_length is None:
                epoch_length = len(data_loader)
        else:
            if epoch_length is None:
                raise ValueError(
                    "If data_loader is not PyTorch DataLoader, must specify the epoch_length."
                )

    def _iteration(self, engine, batchdata: Dict[str, torch.Tensor]):
        """
        Abstract callback function for the processing logic of 1 iteration in Ignite Engine.
        Need subclass to implement different logics, like SupervisedTrainer/Evaluator, GANTrainer, etc.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
            batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    def get_stats(self, *vars):
        """
        Get the statistics information of the workflow process.

        Args:
            vars: variables name in the `self.state`, will use the variable name as the key
                and the state content as the value. if the variable doesn't exist, default value is `None`.

        """
        return {k: getattr(self.state, k, None) for k in vars}
