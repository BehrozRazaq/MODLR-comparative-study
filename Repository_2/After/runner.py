import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Runner:
    def __init__(
        self,
        dataset,
        generator,
        discriminator,
        loss,
        generator_optimizer,
        discriminator_optimizer,
        epochs,
        checkpoint_dir="./training_checkpoints",
    ):
        self.dataset = dataset
        self.generator = generator
        self.discriminator = discriminator
        self.loss = loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.epochs = epochs

        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ["Input Image", "Ground Truth", "Predicted Image"]

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # Getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")
        plt.show()

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_loss, disc_loss = self.loss(
                self.generator, self.discriminator, input_image, target
            )

        generator_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            for input_image, target in self.dataset.train.take(1):
                self.train_step(input_image, target)

            clear_output(wait=True)
            for inp, tar in self.dataset.test.take(1):
                self.generate_images(self.generator, inp, tar)

            if (epoch + 1) % 20 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            print(
                "=> Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, time.time() - start
                )
            )
