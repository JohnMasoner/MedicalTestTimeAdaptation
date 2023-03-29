from typing import Union

import torch
from torch.nn import BCELoss
from monai.losses import DiceLoss, FocalLoss


class Criterion:
    """
    This class defines the criterion for the loss function.

    Args:
        cfg (object): The configuration object.

    Attributes:
        loss_method (Union[list, str]): The loss method.
        activation (str): The activation function.
        loss_method_list (list): The list of loss methods.

    Methods:
        __call__: The method to call the criterion.
        _criterion: The method to define the criterion.

    """

    def __init__(self, cfg):
        """
        The constructor for Criterion.

        Args:
            cfg (object): The configuration object.

        """
        self.cfg = cfg
        self.loss_method: Union[list, str] = cfg.loss.method
        self.activation: str = cfg.loss.activation

        self.loss_method_list: list = []

        self._criterion()

    def __call__(self, pred, label) -> dict:
        """
        This method calls the criterion.

        Args:
            pred (torch.Tensor): The predicted tensor.
            label (torch.Tensor): The label tensor.

        Returns:
            dict: The dictionary of losses.
            float: The sum of losses.

        """
        loss = {}

        pred = self.activate(pred)
        if len(self.loss_method) > 0:
            for idx, criterion in enumerate(self.loss_method_list):
                loss[f"{self.loss_method[idx]}"] = criterion(pred, label)

        return loss, sum(loss.values())

    def _criterion(self):
        """
        This method defines the criterion.

        Raises:
            NotImplementedError: If the loss method is not implemented.

        """
        criterion_method = ["dice", "bce"]
        if (
            self.loss_method not in criterion_method
            or len(set(criterion_method) & set(self.loss_method)) == 0
        ):
            raise NotImplementedError(f"{self.loss_method} is not implemented!")

        if isinstance(self.loss_method, str):
            self.loss_method = self.loss_method.split()

        if "dice" in self.loss_method:
            self.loss_method_list.append(DiceLoss())
        if "bce" in self.loss_method:
            self.loss_method_list.append(BCELoss())
        if "focal" in self.loss_method:
            self.loss_method_list.append(FocalLoss())

    def activate(self, pred) -> torch.Tensor:
        """
        This method activates the given tensor.

        Args:
            pred (torch.Tensor): The tensor to activate.

        Returns:
            torch.Tensor: The activated tensor.

        Raises:
            NotImplementedError: If the activation function is not implemented.

        """
        if self.activation == "sigmoid":
            pred = pred.sigmoid()
        elif self.activation == "softmax":
            pred = pred.softmax()
        else:
            raise NotImplementedError(f"{self.activation} is not implemented!")
        return pred
