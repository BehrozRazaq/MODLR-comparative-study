import tensorflow as tf
import cv2
import random
import numpy as np
import os

from utils import compute_iou, load_gt_boxes, wandhG, compute_regression


class Dataset:
    def __init__(
        self,
        synthetic_dataset_path,
        batch_size,
        image_height,
        image_width,
        pos_threshold,
        neg_threshold,
        grid_width,
        grid_height,
    ):
        self.synthetic_dataset_path = synthetic_dataset_path
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.grid_width = grid_width
        self.grid_height = grid_height

    def create_image_label_path_generator(self):
        image_num = 8000
        image_label_paths = [
            (
                os.path.join(self.synthetic_dataset_path, "image/%d.jpg" % (idx + 1)),
                os.path.join(
                    self.synthetic_dataset_path, "imageAno/%d.txt" % (idx + 1)
                ),
            )
            for idx in range(image_num)
        ]
        while True:
            random.shuffle(image_label_paths)
            for i in range(image_num):
                yield image_label_paths[i]

    def encode_label(self, gt_boxes):
        target_scores = np.zeros(
            shape=[45, 60, 9, 2]
        )  # 0: background, 1: foreground, ,
        target_bboxes = np.zeros(shape=[45, 60, 9, 4])  # t_x, t_y, t_w, t_h
        target_masks = np.zeros(
            shape=[45, 60, 9]
        )  # negative_samples: -1, positive_samples: 1
        for i in range(45):  # y: height
            for j in range(60):  # x: width
                for k in range(9):
                    center_x = j * self.grid_width + self.grid_width * 0.5
                    center_y = i * self.grid_height + self.grid_height * 0.5
                    xmin = center_x - wandhG[k][0] * 0.5
                    ymin = center_y - wandhG[k][1] * 0.5
                    xmax = center_x + wandhG[k][0] * 0.5
                    ymax = center_y + wandhG[k][1] * 0.5
                    # print(xmin, ymin, xmax, ymax)
                    # ignore cross-boundary anchors
                    if (
                        (xmin > -5)
                        & (ymin > -5)
                        & (xmax < (self.image_width + 5))
                        & (ymax < (self.image_height + 5))
                    ):
                        anchor_boxes = np.array([xmin, ymin, xmax, ymax])
                        anchor_boxes = np.expand_dims(anchor_boxes, axis=0)
                        # compute iou between this anchor and all ground-truth boxes in image.
                        ious = compute_iou(anchor_boxes, gt_boxes)
                        positive_masks = ious >= self.pos_threshold
                        negative_masks = ious <= self.neg_threshold

                        if np.any(positive_masks):
                            target_scores[i, j, k, 1] = 1.0
                            target_masks[i, j, k] = 1  # labeled as a positive sample
                            # find out which ground-truth box matches this anchor
                            max_iou_idx = np.argmax(ious)
                            selected_gt_boxes = gt_boxes[max_iou_idx]
                            target_bboxes[i, j, k] = compute_regression(
                                selected_gt_boxes, anchor_boxes[0]
                            )

                        if np.all(negative_masks):
                            target_scores[i, j, k, 0] = 1.0
                            target_masks[i, j, k] = -1  # labeled as a negative sample
        return target_scores, target_bboxes, target_masks

    def process_image_label(self, image_path, label_path):
        raw_image = cv2.imread(image_path)
        gt_boxes = load_gt_boxes(label_path)
        target = self.encode_label(gt_boxes)
        return raw_image / 255.0, target

    def __iter__(self):
        self.image_label_path_generator = self.create_image_label_path_generator()
        return self

    def __next__(self):
        images = np.zeros(
            shape=[self.batch_size, self.image_height, self.image_width, 3],
            dtype=np.float,
        )

        target_scores = np.zeros(shape=[self.batch_size, 45, 60, 9, 2], dtype=np.float)
        target_bboxes = np.zeros(shape=[self.batch_size, 45, 60, 9, 4], dtype=np.float)
        target_masks = np.zeros(shape=[self.batch_size, 45, 60, 9], dtype=np.int)

        for i in range(self.batch_size):
            image_path, label_path = next(self.image_label_path_generator)
            image, target = self.process_image_label(image_path, label_path)
            images[i] = image
            target_scores[i] = target[0]
            target_bboxes[i] = target[1]
            target_masks[i] = target[2]
        return images, target_scores, target_bboxes, target_masks
