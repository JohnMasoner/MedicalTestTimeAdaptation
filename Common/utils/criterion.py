from typing import Union

import monai
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from monai.losses import DiceLoss, FocalLoss

class Criterion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loss_method: Union[list, str] = cfg.loss.method
        self.activation: str = cfg.loss.activation

        self.loss_method_list: list = []

        self._criterion()

    def __call__(self, pred, label) -> dict:
        loss = {}

        pred = self.activate(pred)
        if len(self.loss_method) > 0:
            for idx, criterion in enumerate(self.loss_method_list):
                loss[f'{self.loss_method[idx]}'] = criterion(pred, label)

        return loss, sum(loss.values())

    def _criterion(self):
        criterion_method = ['dice', 'bce']
        if self.loss_method not in criterion_method or len(set(criterion_method) & set(self.loss_method)) == 0:
            raise NotImplementedError(f'{self.metric_method} is not implemented!')

        if isinstance(self.loss_method, str):
            self.loss_method = self.loss_method.split()

        if 'dice' in self.loss_method:
            self.loss_method_list.append(DiceLoss())
        if 'bce' in self.loss_method:
            self.loss_method_list.append(BCELoss())
        if 'focal' in self.loss_method:
            self.loss_method_list.append(FocalLoss())

    def activate(self, pred) -> torch.Tensor:
        if self.activation == 'sigmoid':
            pred = pred.sigmoid()
        elif self.activation == 'softmax':
            pred = pred.softmax()
        else:
            raise NotImplementedError(f'{self.activation} is not implemented!')
        return pred
