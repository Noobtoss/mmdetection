_base_ = '../../../configs/detr/detr_r50_8xb2-150e_coco.py'

# num_classes = 37   # hard encoding bad
load_from = '../../../checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

model = dict(
    bbox_head = dict(
        type='DETRHead',
        num_classes=37,
        cls_feat_loss = dict(
            type='ClsFeatLoss',
            loss='sup_con_loss',
            loss_weight=0.1  # loss_weight=1
        ),
        cls_feat_proj_head = dict(
            type='ClsFeatProjHead',
            proj_head='m',
            dim=256  # hard encoding bad
        )
    )
)
