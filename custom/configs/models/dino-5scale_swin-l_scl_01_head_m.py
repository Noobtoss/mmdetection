_base_ = '../../../configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'

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
        milestones=[30,40],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# num_classes = 37   # hard encoding bad
# load_from = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
load_from = '../../../checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

model = dict(
    bbox_head=dict(
        type='DINOHead',
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
