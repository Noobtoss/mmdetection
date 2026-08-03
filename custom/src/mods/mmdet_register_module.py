import torch
from mmdet.registry import MODELS

from .cls_feat_loss import ClsFeatLoss as _ClsFeatLoss
from .cls_feat_proj_head import ClsFeatProjHead


@MODELS.register_module()
class ClsFeatLoss(_ClsFeatLoss):
    def __init__(self, *args, loss_weight: float = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_weight = loss_weight
        self.logging_cache = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        loss = super().forward(*args, **kwargs)
        self.logging_cache = loss.detach()
        return loss * self.loss_weight

    def get_logging(self) -> torch.Tensor:
        logging = self.logging_cache
        self.logging_cache = None
        return logging

MODELS.register_module(name='ClsFeatProjHead', module=ClsFeatProjHead)
