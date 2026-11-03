_base_ = 'dino-5scale_swin-l.py'

cls_feat = None if '{{$MMDET_CLS_FEAT:None}}' == 'None' else float('{{$MMDET_CLS_FEAT:None}}')
cls_feat_loss = '{{$MMDET_CLS_FEAT_LOSS:sup_con_loss}}'
cls_feat_proj_head = '{{$MMDET_CLS_FEAT_PROJ_HEAD:m}}'

model = dict(
    bbox_head = dict(
        cls_feat_loss = dict(
            type='ClsFeatLoss',
            loss=cls_feat_loss,
            loss_weight=cls_feat
        ),
        cls_feat_proj_head = dict(
            type='ClsFeatProjHead',
            proj_head=cls_feat_proj_head,
            dim=256  # hard encoding bad
        )
    )
)
