from collections import deque
from typing import Optional

import torch
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.registry import MODELS
from mmdet.utils import ConfigType
from mmengine.config import ConfigDict

from .class_losses_weighted import ClassLossWeighted as _ClassLossWeighted
from .class_weights import class_weights as _class_weights
from .cls_feat_loss import ClsFeatLoss as _ClsFeatLoss
from .cls_feat_proj_head import ClsFeatProjHead


@MODELS.register_module()
class ClassLossWeighted(_ClassLossWeighted):
    """
    Acceptable issue: Custom callers passing one-hot targets are unsupported; current heads pass integer labels.
    Acceptable issue: avg_non_ignore=True with mean reduction and no avg_factor uses the wrong denominator; current heads supply avg_factor.
    """
    def __init__(self,
                 loss: ConfigType = dict(type='CrossEntropyLoss', use_sigmoid=False),
                 class_weights: Optional[str] = None,
                 *args, **kwargs,
                 ) -> None:
        if isinstance(loss, (dict, ConfigDict)):
            if kwargs.get('bg_cls_weight') is None:
                kwargs.pop('bg_cls_weight', None)
            loss = MODELS.build(dict(loss, **kwargs))
        if class_weights is not None and not loss.use_sigmoid:
            raise ValueError('Class weighting requires use_sigmoid=True.')
        self.reduction = loss.reduction
        loss.reduction = "none"  # class weighting requires unreduced intermediate values
        super().__init__(loss=loss, **_class_weights[class_weights])

    @property
    def use_sigmoid(self):
        return self.loss.use_sigmoid

    @property
    def custom_cls_channels(self):
        # Faster R-CNN uses this flag to request both channel count and prediction activation.
        return self.use_sigmoid

    def get_cls_channels(self, num_classes):
        # Required by that flag; keep Faster R-CNN's extra background output.
        return num_classes + 1

    def get_activation(self, cls_score):
        # Match the training loss; Faster R-CNN otherwise defaults to softmax prediction.
        if self.use_sigmoid:
            return cls_score.sigmoid()
        return cls_score.softmax(dim=-1)

    def forward(self,
                cls_score: torch.Tensor,
                label: torch.Tensor,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs) -> torch.Tensor:
        reduction = reduction_override or self.reduction
        if ignore_index is not None:
            kwargs["ignore_index"] = ignore_index
        loss = super().forward(cls_score, label, weight, **kwargs)
        return weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

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
