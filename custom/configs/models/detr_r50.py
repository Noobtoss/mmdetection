_base_ = '../../../configs/detr/detr_r50_8xb2-150e_coco.py'

# num_classes = 37   # hard encoding bad
load_from = '../../../checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

model = dict(
    bbox_head = dict(
        type='DETRHead',
        num_classes=37
    )
)
