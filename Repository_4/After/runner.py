import tensorflow as tf


class Runner:
    def __init__(
        self,
        model,
        loss,
        optimizer,
        dataset,
        writer_save_path,
        epochs,
        steps,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.writer = tf.summary.create_file_writer(writer_save_path)
        self.global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.epochs = epochs
        self.steps = steps

    def train(self):
        for epoch in range(self.epochs):
            for step in range(self.steps):
                self.global_steps.assign_add(1)
                image_data, target_scores, target_bboxes, target_masks = next(
                    self.dataset
                )
                with tf.GradientTape() as tape:
                    total_loss, score_loss, boxes_loss = self.loss(
                        self.model,
                        image_data,
                        target_scores,
                        target_bboxes,
                        target_masks,
                    )
                    gradients = tape.gradient(
                        total_loss, self.model.trainable_variables
                    )
                    self.optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )
                    print(
                        "=> epoch %d  step %d  total_loss: %.6f  score_loss: %.6f  boxes_loss: %.6f"
                        % (
                            epoch + 1,
                            step + 1,
                            total_loss.numpy(),
                            score_loss.numpy(),
                            boxes_loss.numpy(),
                        )
                    )
                # writing summary data
                with self.writer.as_default():
                    tf.summary.scalar("total_loss", total_loss, step=self.global_steps)
                    tf.summary.scalar("score_loss", score_loss, step=self.global_steps)
                    tf.summary.scalar("boxes_loss", boxes_loss, step=self.global_steps)
                self.writer.flush()
            self.model.save_weights("RPN.h5")
