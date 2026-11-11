_base_ = [
    '../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

# train ----------------------------------------------------------------------------------------------------------------

max_epochs = int('{{$MMDET_MAX_EPOCHS:100}}')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[round(max_epochs * 0.7), round(max_epochs * 0.9)],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# build_cfg sets this before loading; head outputs must match the dataset classes.
num_classes = int('{{$MMDET_NUM_CLASSES:37}}')
# load_from = '/nfs/scratch/staff/schmittth/code_hub/mmdetection/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
load_from = '../../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

model = dict(
    roi_head=dict(
        type="StandardRoIHead",
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=num_classes,
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='sup_con_loss',
                loss_weight=None
            )
        )
    )
)

train_dataloader = dict(
    batch_size = 2 # 4
)
