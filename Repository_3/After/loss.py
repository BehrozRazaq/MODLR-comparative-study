import torch

from torch.autograd import Variable
import torch.nn.functional as F

from model.utils.net_utils import _smooth_l1_loss

from models import _fasterRCNN


class FasterRCNNLoss:
    def __init__(self, batch_size, class_agnostic):
        self.batch_size = batch_size
        self.class_agnostic = class_agnostic
        pass

    def __call__(self, model: _fasterRCNN, im_data, im_info, gt_boxes, num_boxes):

        model.zero_grad()

        (
            rois,
            rpn_data,
            rpn_cls_score_reshape,
            rpn_bbox_pred,
            (rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws),
            bbox_pred,
            cls_score,
        ) = model(im_data, im_info, gt_boxes, num_boxes)

        #################
        # RPN loss
        #################
        rpn_cls_score = (
            rpn_cls_score_reshape.permute(0, 2, 3, 1)
            .contiguous()
            .view(self.batch_size, -1, 2)
        )
        rpn_label = rpn_data[0].view(self.batch_size, -1)

        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())
        rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[
            1:
        ]

        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        rpn_loss_box = _smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=3,
            dim=[1, 2, 3],
        )
        ###################
        # RPN End         #
        ###################

        ###################
        # fasterRCNN loss #
        ###################
        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        if not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4
            )

            bbox_pred_select = torch.gather(
                bbox_pred_view,
                1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4
                ),
            )

            bbox_pred = bbox_pred_select.squeeze(1)

        # classification loss
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(
            bbox_pred, rois_target, rois_inside_ws, rois_outside_ws
        )

        loss = (
            rpn_loss_cls.mean()
            + rpn_loss_box.mean()
            + RCNN_loss_cls.mean()
            + RCNN_loss_bbox.mean()
        )

        return loss, (
            rpn_loss_cls,
            rpn_loss_box,
            RCNN_loss_cls,
            RCNN_loss_bbox,
            rois_label,
        )
