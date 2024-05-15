import tensorflow as tf


class GANLoss:
    def __init__(, d_loss_function):
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # type: ignore

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(
            tf.ones_like(disc_generated_output), disc_generated_output
        )
        # mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gan_loss = gan_loss + (100 * l1_loss)
        return total_gan_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(
            tf.zeros_like(disc_generated_output), disc_generated_output
        )
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def __call__(self, generator, discriminator, input_image, target):
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gan_loss = self.generator_loss(disc_generated_output, gen_output, target)

        disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        return gan_loss, disc_loss
