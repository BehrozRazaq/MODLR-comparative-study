import tensorflow as tf
from gan_model import _GanModel


class Discriminator(_GanModel):
    def __init__(self):
        initializer = tf.random_normal_initializer(0.0, 0.02)
        inputs = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
        target = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")

        x = tf.keras.layers.concatenate([inputs, target])  # (bs, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(
            512, 4, strides=1, kernel_initializer=initializer, use_bias=False
        )(
            zero_pad1
        )  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
            zero_pad2
        )  # (bs, 30, 30, 1)

        super().__init__(inputs=[inputs, target], outputs=last)
