import tensorflow as tf
import numpy as np


class RPNLoss:
    def __init__(self, lambda_scale):
        self.lambda_scale = lambda_scale

    def compute_loss(
        self, target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes
    ):
        """
        target_scores shape: [1, 45, 60, 9, 2],  pred_scores shape: [1, 45, 60, 9, 2]
        target_bboxes shape: [1, 45, 60, 9, 4],  pred_bboxes shape: [1, 45, 60, 9, 4]
        target_masks  shape: [1, 45, 60, 9]
        """
        score_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_scores, logits=pred_scores
        )
        foreground_background_mask = (np.abs(target_masks) == 1).astype(np.int)
        score_loss = tf.reduce_sum(
            score_loss * foreground_background_mask, axis=[1, 2, 3]
        ) / np.sum(foreground_background_mask)
        score_loss = tf.reduce_mean(score_loss)

        boxes_loss = tf.abs(target_bboxes - pred_bboxes)
        boxes_loss = 0.5 * tf.pow(boxes_loss, 2) * tf.cast(
            boxes_loss < 1, tf.float32
        ) + (boxes_loss - 0.5) * tf.cast(boxes_loss >= 1, tf.float32)
        boxes_loss = tf.reduce_sum(boxes_loss, axis=-1)
        foreground_mask = (target_masks > 0).astype(np.float32)
        boxes_loss = tf.reduce_sum(
            boxes_loss * foreground_mask, axis=[1, 2, 3]
        ) / np.sum(foreground_mask)
        boxes_loss = tf.reduce_mean(boxes_loss)

        return score_loss, boxes_loss

    def __call__(self, model, image_data, target_scores, target_bboxes, target_masks):
        pred_scores, pred_bboxes = model(image_data)
        score_loss, boxes_loss = self.compute_loss(
            target_scores, target_bboxes, target_masks, pred_scores, pred_bboxes
        )
        return score_loss + self.lambda_scale * boxes_loss, score_loss, boxes_loss
