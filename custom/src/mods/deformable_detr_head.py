import copy
from typing import Dict, List, Tuple
import torch
from torch import Tensor
from mmengine.logging import MMLogger
from mmdet.utils import ConfigType
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, OptInstanceList
from mmdet.models.utils import multi_apply
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead as _DeformableDETRHead

# TODO: tmp
# from mmdet.models.dense_heads.detr_head import DETRHead

from .detr_head import DETRHead

_DeformableDETRHead.__bases__ = (DETRHead,)


@MODELS.register_module(force=True)
class DeformableDETRHead(_DeformableDETRHead):

    def __init__(self,
                 *args,
                 cls_feat_loss: ConfigType = None,
                 cls_feat_proj_head: ConfigType = None,
                 **kwargs) -> None:
        MMLogger.get_current_instance().warning('[Modded] DeformableDETRHead')
        super().__init__(*args, **kwargs)
        self.cls_feat_loss = MODELS.build(cls_feat_loss) if cls_feat_loss is not None else None
        self.cls_feat_proj_head = MODELS.build(cls_feat_proj_head) if cls_feat_proj_head is not None else None

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        # >>> MOD
        all_layers_cls_feats = hidden_states
        if self.cls_feat_proj_head is not None:
            all_layers_cls_feats = self.cls_feat_proj_head(all_layers_cls_feats)
        return all_layers_outputs_classes, all_layers_outputs_coords, all_layers_cls_feats
    # <<< MOD

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
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        # <<< MOD
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
                Only when `as_two_stage` is `True` it would be passes in,
                otherwise, it would be `None`.
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h). Only when `as_two_stage`
                is `True` it would be passed in, otherwise it would be `None`.
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
        # >>> MOD
        loss_dict = DETRHead.loss_by_feat(self,
                                          all_layers_cls_scores,
                                          all_layers_bbox_preds,
                                          all_layers_cls_feats,
                                          batch_gt_instances, batch_img_metas,
                                          batch_gt_instances_ignore)
        # <<< MOD

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            proposal_gt_instances = copy.deepcopy(batch_gt_instances)
            for i in range(len(proposal_gt_instances)):
                proposal_gt_instances[i].labels = torch.zeros_like(
                    proposal_gt_instances[i].labels)
            # >>> MOD
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, _ = \
                self.loss_by_feat_single(
                    cls_scores=enc_cls_scores,
                    bbox_preds=enc_bbox_preds,
                    cls_feats=None,
                    batch_gt_instances=proposal_gt_instances,
                    batch_img_metas=batch_img_metas)
            # <<< MOD
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        return loss_dict

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        all_layer_cls_feats: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        return super().predict_by_feat(all_layers_cls_scores, all_layers_bbox_preds, batch_img_metas, rescale)
