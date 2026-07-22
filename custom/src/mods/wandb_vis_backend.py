from typing import Optional, Union
import torch
import numpy as np
from mmengine.visualization import WandbVisBackend as _WandbVisBackend
from mmengine.registry import VISBACKENDS


@VISBACKENDS.register_module(force=True)
class WandbVisBackend(_WandbVisBackend):
    train_prefix: str = 'train'
    train_scalars = ['loss', 'lr']  # substrings to match

    def _add_prefix(self, name):
        if any(name == kw or name.startswith(kw + '_') for kw in self.train_scalars):
            return f'{self.train_prefix}/{name}'
        return name

    def add_scalar(self,
                    name: str,
                    value: Union[int, float, torch.Tensor, np.ndarray],
                    step: int = 0,
                    **kwargs) -> None:
        name = self._add_prefix(name)
        super().add_scalar(name, value, step, **kwargs)

    def add_scalars(self,
                     scalar_dict: dict,
                     step: int = 0,
                     file_path: Optional[str] = None,
                     **kwargs) -> None:
        scalar_dict = {
            self._add_prefix(k): v
            for k, v in scalar_dict.items()
        }
        super().add_scalars(scalar_dict, step, file_path, **kwargs)
