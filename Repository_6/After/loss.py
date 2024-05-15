from modules import label_smoothing  # type: ignore
import tensorflow as tf


class Loss:
    def __init__(self, hp, token2idx):
        self.hp = hp
        self.token2idx = token2idx

    def __call__(self, model, xs, ys):
        memory, sents1, src_masks = model.encode(xs)
        logits, preds, y, sents2 = model.decode(ys, memory, src_masks)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        return tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)
