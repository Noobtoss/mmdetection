import inspect
import os
import warnings
import torch
import torch.nn as nn
from pytorch_metric_learning import losses, reducers


def _strict_degenerate() -> bool:
    return os.getenv("CLS_FEAT_STRICT", os.getenv("DEBUG_NAN", "0")) == "1"


class UnpackReducer(reducers.BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        if loss_indices.dtype == torch.bool:
            return losses[loss_indices]
        return losses


class NormalizeEmbeddingsWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, embeddings, *args, **kwargs):
        embeddings = embeddings.float()
        norm = embeddings.norm(p=2, dim=1, keepdim=True).clamp_min(1e-4)
        return self.loss(embeddings / norm, *args, **kwargs)


class FeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None or loss == "None":
            return None
        elif loss == "sup_con_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss
            params = {
                "temperature": 0.07,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.SupConLoss(**params, reducer=UnpackReducer()))

        elif loss == "circle_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss
            params = {
                "m": 0.40,
                "gamma": 32,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.CircleLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.CircleLoss(**params, reducer=UnpackReducer()))

        elif loss == "multi_sim_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss
            params = {
                "alpha": 2.0,
                "beta": 12.0,
                "base": 0.5,
            }
            params.update(
                {k: v for k, v in kwargs.items() if k in inspect.signature(losses.MultiSimilarityLoss).parameters}
            )
            return NormalizeEmbeddingsWrapper(losses.MultiSimilarityLoss(**params, reducer=UnpackReducer()))

        else:
            raise ValueError(f"Unknown feat loss type: '{loss}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, **kwargs):
        super().__init__()
        self.loss = FeatLossFactory.get(loss, **kwargs)
        self._warned_degenerate = False

    def _degenerate_batch(self, cls_feats: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        n_pos = 0 if target_cls is None else int(target_cls.numel())
        n_classes = 0 if target_cls is None else int(torch.unique(target_cls).numel())
        msg = (f"[ClsFeatLoss] degenerate batch: {n_pos} selected ROIs, {n_classes} distinct classes. "
               f"Metric-learning losses need at least two samples sharing a class, otherwise the reduced "
               f"loss is empty and .mean() is 0/0 = NaN.")
        if _strict_degenerate():
            raise RuntimeError(msg)
        if not self._warned_degenerate:
            self._warned_degenerate = True
            warnings.warn(msg + " (further occurrences suppressed; returning zero loss)")
        return cls_feats.sum() * 0.0

    def forward(self,
                cls_feats: torch.Tensor, target_cls: torch.Tensor = None, target_scores: torch.Tensor = None
                ) -> torch.Tensor:
        if target_cls is None:
            target_cls = target_scores.max(-1).indices
        if cls_feats.numel() == 0 or target_cls.numel() < 2:
            return self._degenerate_batch(cls_feats, target_cls)
        loss_per_element = self.loss(cls_feats, target_cls).squeeze(-1)
        if loss_per_element.numel() == 0:
            return self._degenerate_batch(cls_feats, target_cls)
        return loss_per_element.mean()
