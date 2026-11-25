import torch
import torch.nn as nn


class ClassLossWeighted(nn.Module):
    def __init__(self,
                 loss: nn.Module = nn.BCEWithLogitsLoss(reduction="none"),
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ) -> None:
        super().__init__()
        self.loss = loss

        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred_scores: torch.Tensor, labels: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        loss = self.loss(pred_scores, labels, *args, **kwargs)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device=pred_scores.device)
            # Faster R-CNN adds a background output; weight foreground classes and leave background at 1.
            weight_per_class = self.class_weights.new_ones(loss.shape)
            weight_per_class[..., :self.class_weights.size(-1)] = self.class_weights.reshape(-1)
            loss = loss * weight_per_class
        if self.class_weights_matrix is not None:
            self.class_weights_matrix = self.class_weights_matrix.to(device=pred_scores.device)
            valid = (labels >= 0) & (labels < self.class_weights_matrix.size(0))
            weight_per_sample = self.class_weights_matrix.new_ones(loss.shape)
            weight_per_sample[valid, :self.class_weights_matrix.size(-1)] = self.class_weights_matrix[labels[valid]]
            loss = loss * weight_per_sample
        return loss
