# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
"""
import tensorflow as tf

from modules import noam_scheme  # type: ignore
from utils import save_hparams  # type: ignore
from model import Transformer

from hparams import Hparams  # type: ignore
import logging

from dataset import Dataset
from loss import Loss
from runner import Runner

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
train_set = Dataset(
    hp.train1, hp.train2, hp.maxlen1, hp.maxlen2, hp.vocab, hp.batch_size, shuffle=True
)
eval_set = Dataset(
    hp.eval1, hp.eval2, 100000, 100000, hp.vocab, hp.batch_size, shuffle=False
)

# create a iterator of the correct shape and type

logging.info("# Load model")
m = Transformer(hp, train_set.token2idx, train_set.idx2token)

loss = Loss(hp, train_set.token2idx)

global_step = tf.train.get_or_create_global_step()
lr = noam_scheme(hp.lr, global_step, hp.warmup_steps)
optimizer = tf.train.AdamOptimizer(lr)

runner = Runner(train_set, eval_set, m, loss, optimizer, global_step, lr, hp, logging)


logging.info("# Session")

runner.train()
