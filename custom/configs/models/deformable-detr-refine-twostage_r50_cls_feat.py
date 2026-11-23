_base_ = 'deformable-detr-refine-twostage_r50.py'

cls_feat = '{{$MMDET_CLS_FEAT:None}}'
cls_feat_loss = '{{$MMDET_CLS_FEAT_LOSS:sup_con_loss}}'
cls_feat_proj_head = '{{$MMDET_CLS_FEAT_PROJ_HEAD:m}}'

cls_feat = None if cls_feat == 'None' else float(cls_feat)
cls_feat_loss = None if cls_feat_loss == 'None' else cls_feat_loss
cls_feat_proj_head = None if cls_feat_proj_head == 'None' else cls_feat_proj_head

model = dict(
    bbox_head = dict(
        cls_feat_loss = None if cls_feat_loss is None else dict(
            type='ClsFeatLoss',
            loss=cls_feat_loss,
            loss_weight=cls_feat
        ),
        cls_feat_proj_head = None if cls_feat_proj_head is None else dict(
            type='ClsFeatProjHead',
            proj_head=cls_feat_proj_head,
            dim=256  # hard encoding bad
        )
    )
)
