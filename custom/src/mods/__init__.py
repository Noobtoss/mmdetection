# mods model eval
from .coco_metric import CocoMetric
from .wandb_vis_backend import WandbVisBackend
# mods model architectures
from .standard_roi_head import StandardRoIHead
from .shared2fc_bbox_head import Shared2FCBBoxHead
from .detr_head import DETRHead
from .deformable_detr_head import DeformableDETRHead
from .dino_head import DINOHead
# mods training loss
from .mmdet_register_module import ClsFeatLoss, ClsFeatProjHead
