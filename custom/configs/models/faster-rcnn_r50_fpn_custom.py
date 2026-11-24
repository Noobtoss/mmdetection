_base_ = 'faster-rcnn_r50_fpn.py'

cls_feat = '{{$MMDET_CLS_FEAT:None}}'
cls_feat_loss = '{{$MMDET_CLS_FEAT_LOSS:sup_con_loss}}'
cls_feat_proj_head = '{{$MMDET_CLS_FEAT_PROJ_HEAD:m}}'
class_weights = '{{$MMDET_CLASS_WEIGHTS:None}}'

cls_feat = None if cls_feat == 'None' else float(cls_feat)
cls_feat_loss = None if cls_feat_loss == 'None' else cls_feat_loss
cls_feat_proj_head = None if cls_feat_proj_head == 'None' else cls_feat_proj_head
class_weights = None if class_weights == 'None' else class_weights

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                _delete_=True,
                type='ClassLossWeighted',
                class_weights=class_weights,
                loss=dict(type='CrossEntropyLoss', use_sigmoid=class_weights is not None, loss_weight=1.0)
            ),
            cls_feat_loss = None if cls_feat_loss is None else dict(
                type='ClsFeatLoss', # 'ClsFeatLoss',
                loss=cls_feat_loss,
                loss_weight=cls_feat
            ),
            cls_feat_proj_head = None if cls_feat_proj_head is None else dict(
                type='ClsFeatProjHead',
                proj_head=cls_feat_proj_head,
                dim=1024  # hard encoding bad
            )
        )
    )
)
