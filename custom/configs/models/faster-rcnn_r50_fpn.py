_base_ = '../../../configs/_base_/models/faster-rcnn_r50_fpn.py'

# num_classes = 37   # hard encoding bad
load_from = '../../../checkpoints/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=37,  # num_classes)
            cls_feat_proj_head=dict(
                type='ClsFeatProjHead', proj_head='s', dim=1024,  # hard encoding bad
            ),
            cls_feat_loss = dict(
                type='ClsFeatLoss', loss='sup_con_loss', loss_weight=0  # loss_weight=1
            )
        )
    )
)
