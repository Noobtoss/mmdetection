_base_ = [
    '_base_/models/faster-rcnn_r50_fpn.py',
    '_base_/datasets/05ACCV2026Plus.py',
    '_base_/schedule_1x.py',
    '_base_/runtime.py',
]
max_epochs = 10
batch_size=8
num_classes=37
load_from = 'checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes)
    )
)
train_dataloader = dict(
    batch_size=batch_size
)
train_cfg = dict(
    max_epochs=max_epochs
)
