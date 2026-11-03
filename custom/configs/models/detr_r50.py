_base_ = 'detr_r50_base.py'

model = dict(
    bbox_head = dict(
        cls_feat_loss = dict(
            type='ClsFeatLoss',
            loss='{{$CLS_FEAT_LOSS:sup_con_loss}}',
            loss_weight=float('{{$CLS_FEAT:0.2}}')
        ),
        cls_feat_proj_head = dict(
            type='ClsFeatProjHead',
            proj_head='{{$CLS_FEAT_PROJ_HEAD:m}}',
            dim=256  # hard encoding bad
        )
    )
)
