#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-10-17 15:00:25
#   Description :
#
# ================================================================


import tensorflow as tf

from dataset import Dataset
from rpn import RPNplus
from loss import RPNLoss
from runner import Runner

pos_thresh = 0.5
neg_thresh = 0.1
grid_width = grid_height = 16
image_height, image_width = 720, 960

EPOCHS = 10
STEPS = 4000
batch_size = 2
lambda_scale = 1.0
synthetic_dataset_path = "./synthetic_dataset"
writer_save_path = "./logs"

dataset = Dataset(
    synthetic_dataset_path,
    batch_size,
    image_height,
    image_width,
    pos_thresh,
    neg_thresh,
    grid_width,
    grid_height,
)

model = RPNplus()
loss = RPNLoss(lambda_scale)
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
runner = Runner(model, loss, optimizer, dataset, writer_save_path, EPOCHS, STEPS)

runner.train()
