_base_ = 'faster-rcnn_r50_fpn.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='sup_con_loss',
                loss_weight=0.1
            ),
            cls_feat_proj_head = dict(
                type='ClsFeatProjHead',
                proj_head='m',
                dim=1024  # hard encoding bad
            )
        )
    )
)
