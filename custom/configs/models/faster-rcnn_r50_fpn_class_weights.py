_base_ = 'faster-rcnn_r50_fpn.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_feat_loss = dict(
                type='ClsFeatLoss',
                loss='sup_con_loss',
                loss_weight=0.1
            ),
            loss_cls = dict(
                type='ClassLossWeighted',
                loss=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0
                ),
                class_weights='class_sim' # 'class_sim_matrix'
            )
        )
    )
)
