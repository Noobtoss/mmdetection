from collections import deque
from typing import Optional

import torch
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from mmengine.config import ConfigDict

from .class_losses_weighted import ClassLossWeighted as _ClassLossWeighted
from .class_weights import class_weights as _class_weights
from .cls_feat_loss import ClsFeatLoss as _ClsFeatLoss
from .cls_feat_proj_head import ClsFeatProjHead


@MODELS.register_module()
class ClassLossWeighted(_ClassLossWeighted):
    def __init__(self,
                 loss: ConfigType = dict(type='CrossEntropyLoss', use_sigmoid=False),
                 class_weights: Optional[str] = None,
                 *args, **kwargs,
                 ) -> None:
        if isinstance(loss, (dict, ConfigDict)):
            loss = MODELS.build(loss)
        loss.reduction = "none"  # class weighting requires unreduced intermediate values
        super().__init__(loss=loss, **_class_weights[class_weights])

@MODELS.register_module()
class ClsFeatLoss(_ClsFeatLoss):
    def __init__(self, *args, loss_weight: float = 1, log_window_size: int = 20, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight
        self.logging_cache = None
        self.log_window_size = log_window_size
        self._loss_history = deque(maxlen=log_window_size)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = super().forward(*args, **kwargs)
        detached = loss.detach()
        self._loss_history.append(detached)
        self.logging_cache = sum(self._loss_history) / len(self._loss_history)
        return loss * self.loss_weight

    def get_logging(self) -> torch.Tensor:
        logging = self.logging_cache
        self.logging_cache = None
        return logging

MODELS.register_module(name='ClsFeatProjHead', module=ClsFeatProjHead)
