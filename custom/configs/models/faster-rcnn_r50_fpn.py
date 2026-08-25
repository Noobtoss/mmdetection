_base_ = '../../../configs/_base_/models/faster-rcnn_r50_fpn.py'

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
                loss_weight=0  # loss_weight=1
            )
        )
    )
)
