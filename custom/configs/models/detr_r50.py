_base_ = '../../../configs/detr/detr_r50_8xb2-150e_coco.py'

# train ----------------------------------------------------------------------------------------------------------------

max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[70,90],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# num_classes = 37   # hard encoding bad
# load_from = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
load_from = '../../../checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

model = dict(
    bbox_head = dict(
        type='DETRHead',
        num_classes=37,
        cls_feat_loss = dict(
            type='ClsFeatLoss',
            loss='sup_con_loss',
            loss_weight=0  # loss_weight=1
        )
    )
)
