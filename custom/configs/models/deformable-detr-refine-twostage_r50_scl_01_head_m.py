_base_ = '../../../configs/deformable_detr/deformable-detr-refine-twostage_r50_16xb2-50e_coco.py'

# train ----------------------------------------------------------------------------------------------------------------

max_epochs = 50
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
        milestones=[40],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# num_classes = 37   # hard encoding bad
# load_from = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection/checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'
load_from = '../../../checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth'

model = dict(
    bbox_head=dict(
        type='DeformableDETRHead',
        num_classes=37,
        cls_feat_loss=dict(
            type='ClsFeatLoss',
            loss='sup_con_loss',
            loss_weight=0.1  # loss_weight=1
        ),
        cls_feat_proj_head=dict(
            type='ClsFeatProjHead',
            proj_head='m',
            dim=256  # hard encoding bad
        )
    )
)
