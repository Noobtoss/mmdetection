import torch
from torch import Tensor
from typing import Dict, List, Tuple
from mmengine.logging import MMLogger
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, OptMultiConfig, reduce_mean
from mmdet.registry import MODELS
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.detr_head import DETRHead as _DETRHead


@MODELS.register_module(force=True)
class DETRHead(_DETRHead):

    # >>> MOD
    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            train_cfg: ConfigType = dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg: ConfigType = dict(max_per_img=100),
            init_cfg: OptMultiConfig = None,
            cls_feat_loss: ConfigType = None,
            cls_feat_proj_head: ConfigType = None) -> None:
        MMLogger.get_current_instance().warning('[Modded] DETRHead')
        super(_DETRHead, self).__init__(init_cfg=init_cfg)
        # <<< MOD
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead):  # MOD: base class is not inheritance-safe
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()
        # >>> MOD
        assert not self.loss_cls.use_sigmoid, "Current implementation only supports loss_cls.use_sigmoid=False"
        self.cls_feat_loss = MODELS.build(cls_feat_loss) if cls_feat_loss is not None else None
        self.cls_feat_proj_head = MODELS.build(cls_feat_proj_head) if cls_feat_proj_head is not None else None
        # <<< MOD

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor]:
        """"Forward function.

        Args:
            hidden_states (Tensor): Features from transformer decoder. If
                `return_intermediate_dec` in detr.py is True output has shape
                (num_decoder_layers, bs, num_queries, dim), else has shape
                (1, bs, num_queries, dim) which only contains the last layer
                outputs.
        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """
        layers_cls_scores = self.fc_cls(hidden_states)
        layers_bbox_preds = self.fc_reg(
            self.activate(self.reg_ffn(hidden_states))).sigmoid()
        # >>> MOD
        layers_cls_feats = hidden_states
        if self.cls_feat_proj_head is not None:
            layers_cls_feats = self.cls_feat_proj_head(layers_cls_feats)
        return layers_cls_scores, layers_bbox_preds, layers_cls_feats
    # <<< MOD

    # >>> MOD
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_cls_feats: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # <<< MOD
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        # >>> MOD
        losses_cls, losses_bbox, losses_iou, losses_feats = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_cls_feats,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_cls_feats'] = losses_feats[-1]
        # assumes that loss last decoder layer is commuted last
        loss_dict['logging_cls_feats'] = self.cls_feat_loss.get_logging()
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_feat_i in \
                zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1], losses_feats[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_cls_feats'] = loss_feat_i
            num_dec_layer += 1
        # <<< MOD
        return loss_dict

    # >>> MOD
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor, cls_feats: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        # <<< MOD
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # >>> MOD
        loss_feats = loss_cls.new_zeros(())
        if self.cls_feat_loss is not None:
            cls_feats = cls_feats.reshape(-1, cls_feats.shape[-1])
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            loss_feats = self.cls_feat_loss(cls_feats=cls_feats[pos_inds],
                                            target_cls=labels[pos_inds],
                                            pred_scores=cls_scores[pos_inds]
                                            )
        # <<< MOD

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        # >>> MOD
        return loss_cls, loss_bbox, loss_iou, loss_feats
    # <<< MOD

    def predict_by_feat(self,
                        layer_cls_scores: Tensor,
                        layer_bbox_preds: Tensor,
                        layer_cls_feats: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        return super().predict_by_feat(layer_cls_scores, layer_bbox_preds, batch_img_metas, rescale)
