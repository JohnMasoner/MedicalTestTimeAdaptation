from typing import Union

import torch
from monai.transforms import (
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    FillHoles,
)
from monai.metrics import HausdorffDistanceMetric, DiceMetric


class Metric:
    """
    A class to calculate metrics for segmentation models.

    Args:
        cfg (Config): The configuration object.

    Attributes:
        cfg (Config): The configuration object.
        metric_method (Union[list, str]): The metric method(s) to use.
        activation (str): The activation function to use.
        post_process (Union[list, str]): The post-processing method(s) to use.
        post_process_list (list): The list of post-processing methods.
        metric_method_list (list): The list of metric methods.

    Methods:
        __call__(self, pred, label) -> dict: Calculates the metrics for the given predictions and labels.
        _metric(self): Initializes the metric methods.
        activate(self, pred) -> torch.Tensor: Applies the activation function to the predictions.
        _post_process(self): Initializes the post-processing methods.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.metric_method: Union[list, str] = cfg.metric.method
        self.activation: str = cfg.metric.activation
        self.post_process: Union[list, str] = cfg.metric.post_process

        self.post_process_list: list = []
        self.metric_method_list: list = []

        self._post_process()
        self._metric()

    def __call__(self, pred, label) -> dict:
        """
        Calculates the metrics for the given predictions and labels.

        Args:
            pred (torch.Tensor): The predicted segmentation mask.
            label (torch.Tensor): The ground truth segmentation mask.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        metric = {}

        pred = self.activate(pred)
        if len(self.post_process_list) > 0:
            for post_process_method in self.post_process_list:
                pred = post_process_method(pred)

        if len(self.metric_method_list) > 0:
            for idx, metric_method in enumerate(self.metric_method_list):
                metric[f"{self.metric_method[idx]}"] = metric_method(pred, label)

        return metric

    def _metric(self):
        """
        This method performs metric calculation on the data.

        Raises:
            NotImplementedError: If the metric method is not implemented.
        """
        metric_method = ["dice", "hd95"]
        if (
            self.metric_method not in metric_method
            or len(set(metric_method) & set(self.metric_method)) == 0
        ):
            raise NotImplementedError(f"{self.metric_method} is not implemented!")

        if isinstance(self.metric_method, str):
            self.metric_method = self.metric_method.split()

        if "hd95" in self.metric_method:
            self.metric_method_list.append(HausdorffDistanceMetric())
        if "dice" in self.metric_method:
            self.metric_method_list.append(DiceMetric())

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
            pred = (pred.sigmoid() > 0.5).float()
        elif self.activation == "softmax":
            pred = (pred.softmax() > 0.5).float()
        else:
            raise NotImplementedError(f"{self.activation} is not implemented!")
        return pred

    def _post_process(self):
        """
        This method performs post-processing on the data.

        Raises:
            NotImplementedError: If the post-processing method is not implemented.

        """
        post_process = ["fill_hole", "keep_largest", "remove_small"]
        if (
            self.post_process not in post_process
            or len(set(post_process) & set(self.post_process)) == 0
        ):
            raise NotImplementedError(f"{self.post_process} is not implemented!")

        if isinstance(self.post_process, str):
            self.post_process = self.post_process.split()

        if "fill_hole" in self.post_process:
            self.post_process_list.append(FillHoles())
        if "keep_largest" in self.post_process:
            self.post_process_list.append(KeepLargestConnectedComponent())
        if "remove_small" in self.post_process:
            self.post_process_list.append(RemoveSmallObjects())
