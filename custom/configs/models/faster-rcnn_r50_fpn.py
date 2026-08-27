_base_ = [
    '../../../configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../configs/_base_/datasets/coco_detection.py',
    '../../../configs/_base_/schedules/schedule_1x.py',
    '../../../configs/_base_/default_runtime.py'
]

# train ----------------------------------------------------------------------------------------------------------------

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,  # 12
        by_epoch=True,
        milestones=[70,90],
        gamma=0.1)
]

# model ----------------------------------------------------------------------------------------------------------------

# num_classes = 37   # hard encoding bad
# load_from = '/nfs/scratch/staff/schmittth/code_nexus/mmdetection/checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'
load_from = '../../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

model = dict(
    roi_head=dict(
        type="StandardRoIHead",
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=37,  # num_classes)
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='sup_con_loss',
                loss_weight=0
            )
        )
    )
)
