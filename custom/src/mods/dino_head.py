from typing import Dict, List, Tuple
import torch
from torch import Tensor
from mmengine.logging import MMLogger
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmdet.models.utils import multi_apply
from mmdet.models.dense_heads.dino_head import DINOHead as _DINOHead

from .detr_head import DETRHead
from .deformable_detr_head import DeformableDETRHead


_DINOHead.__bases__ = (DeformableDETRHead,)


@MODELS.register_module(force=True)
class DINOHead(_DINOHead):

    def __init__(self, *args, **kwargs):
        MMLogger.get_current_instance().warning('[Modded] DINOHead')
        super().__init__(*args, **kwargs)

    # >>> MOD
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_cls_feats: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # <<< MOD
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # >>> MOD
        # extract denoising and matching part of outputs
        (all_layers_matching_cls_scores,
         all_layers_matching_bbox_preds,
         all_layers_matching_cls_feats,
         all_layers_denoising_cls_scores,
         all_layers_denoising_bbox_preds,
         all_layers_denoising_cls_feats) = self.split_outputs(all_layers_cls_scores,
                                                              all_layers_bbox_preds,
                                                              all_layers_cls_feats,
                                                              dn_meta)
        loss_dict = DETRHead.loss_by_feat(self,
            all_layers_matching_cls_scores, all_layers_matching_bbox_preds, all_layers_matching_cls_feats,
            batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        # <<< MOD
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            # >>> MOD
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, _ = \
                self.loss_by_feat_single(
                    cls_scores=enc_cls_scores,
                    bbox_preds=enc_bbox_preds,
                    cls_feats=None,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas
                )
            # <<< MOD
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            # >>> MOD
            dn_losses_cls, dn_losses_bbox, dn_losses_iou, dn_losses_feat = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_cls_feats,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
            if self.cls_feat_loss is not None:
                loss_dict['dn_loss_cls_feats'] = dn_losses_feat[-1]
                # loss_dict['logging_dn_cls_feats'] = self.cls_feat_loss.get_logging()
            # <<< MOD
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in \
                    enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1],
                                  dn_losses_iou[:-1])):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
        return loss_dict

    # >>> MOD
    def loss_dn(self,
                all_layers_denoising_cls_scores: Tensor,
                all_layers_denoising_bbox_preds: Tensor,
                all_layers_denoising_cls_feats: Tensor,
                batch_gt_instances: InstanceList, batch_img_metas: List[dict],
                dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        # <<< MOD
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            all_layers_denoising_cls_feats,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    # >>> MOD
    def _loss_dn_single(self,
                        dn_cls_scores: Tensor,
                        dn_bbox_preds: Tensor,
                        dn_cls_feats: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        # <<< MOD
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)
        # >>> MOD
        loss_feats = loss_cls.new_zeros(())
        if dn_cls_feats is not None and self.cls_feat_loss is not None:
            dn_cls_feats = dn_cls_feats.reshape(-1, dn_cls_feats.shape[-1])
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            loss_feats = self.cls_feat_loss(cls_feats=dn_cls_feats[pos_inds],
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
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
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

    # >>> MOD
    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      all_layers_cls_feats: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        # <<< MOD
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        num_denoising_queries = dn_meta['num_denoising_queries']
        if dn_meta is not None:
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
            # >>> MOD
            all_layers_denoising_cls_feats = \
                all_layers_cls_feats[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_feats = \
                all_layers_cls_feats[:, :, num_denoising_queries:, :]
            # <<< MOD
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
            # >>> MOD
            all_layers_denoising_cls_feats = None
            all_layers_matching_cls_feats = all_layers_cls_feats
            # <<< MOD
            # >>> MOD
        return (all_layers_matching_cls_scores,
                all_layers_matching_bbox_preds,
                all_layers_matching_cls_feats,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                all_layers_denoising_cls_feats)
        # <<< MOD
