_base_ = 'detr_r50.py'

cls_feat = '{{$MMDET_CLS_FEAT:None}}'
cls_feat_loss = '{{$MMDET_CLS_FEAT_LOSS:sup_con_loss}}'
cls_feat_proj_head = '{{$MMDET_CLS_FEAT_PROJ_HEAD:m}}'
class_weights = '{{$MMDET_CLASS_WEIGHTS:None}}'

cls_feat = None if cls_feat == 'None' else float(cls_feat)
cls_feat_loss = None if cls_feat_loss == 'None' else cls_feat_loss
cls_feat_proj_head = None if cls_feat_proj_head == 'None' else cls_feat_proj_head
class_weights = None if class_weights == 'None' else class_weights

model = dict(
    bbox_head=dict(
        loss_cls=dict(
            _delete_=True,
            type='ClassLossWeighted',
            class_weights=class_weights,
            loss=dict(
                type='CrossEntropyLoss',
                use_sigmoid=class_weights is not None,
                loss_weight=1.0
            ),
            class_weight=1.0 if class_weights is None else None,
            bg_cls_weight=0.1 if class_weights is None else None
        ),
        cls_feat_loss=None if cls_feat_loss is None else dict(
            type='ClsFeatLoss',
            loss=cls_feat_loss,
            loss_weight=cls_feat
        ),
        cls_feat_proj_head=None if cls_feat_proj_head is None else dict(
            type='ClsFeatProjHead',
            proj_head=cls_feat_proj_head,
            dim=256  # hard encoding bad
        )
    ),
    train_cfg=dict(assigner=dict(match_costs=[
        dict(type='ClassificationCost', weight=1.0)
        if class_weights is None else
        dict(type='FocalLossCost', gamma=0.0, alpha=0.5, weight=1.0),
        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
        dict(type='IoUCost', weight=2.0, iou_mode='giou')
    ]))
)
