_base_ = '../../../configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py'

# train ----------------------------------------------------------------------------------------------------------------

max_epochs = int('{{$MMDET_MAX_EPOCHS:50}}')
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
        milestones=[round(max_epochs * 0.6), round(max_epochs * 0.8)],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# build_cfg sets this before loading; head outputs must match the dataset classes.
num_classes = int('{{$MMDET_NUM_CLASSES:37}}')
# load_from = '/nfs/scratch/staff/schmittth/code_hub/mmdetection/checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
load_from = '../../../checkpoints/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'

model = dict(
    bbox_head=dict(
        type='DINOHead',
        num_classes=num_classes,
        cls_feat_loss=dict(
            type='ClsFeatLoss',
            loss='sup_con_loss',
            loss_weight=None
        )
    )
)

train_dataloader = dict(
    batch_size = 2 # 4
)
