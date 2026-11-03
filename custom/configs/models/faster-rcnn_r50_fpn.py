_base_ = 'faster-rcnn_r50_fpn_base.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='{{$CLS_FEAT_LOSS:sup_con_loss}}',
                loss_weight=float('{{$CLS_FEAT:0.2}}')
            ),
            cls_feat_proj_head = dict(
                type='ClsFeatProjHead',
                proj_head='{{$CLS_FEAT_PROJ_HEAD:m}}',
                dim=1024  # hard encoding bad
            )
        )
    )
)
