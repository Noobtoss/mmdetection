_base_ = [
    'models/faster-rcnn_r50_fpn.py',
    'datasets/05ACCV2026Plus_local.py',
    'runtime.py',
]

max_epochs = 100
batch_size = 8
num_classes = 37
load_from = '../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

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
