_base_ = 'faster-rcnn_r50_fpn.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='sup_con_loss',
                loss_weight=0.1
            )
        )
    )
)
