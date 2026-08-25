_base_ = 'detr_r50.py'

model = dict(
    bbox_head = dict(
        cls_feat_loss = dict(
            type='ClsFeatLoss',
            loss='sup_con_loss',
            loss_weight=0.1  # loss_weight=1
        )
    )
)
