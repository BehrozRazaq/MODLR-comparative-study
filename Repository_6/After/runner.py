from matplotlib.pylab import f
import tensorflow as tf
from utils import get_hypotheses, calc_bleu  # type: ignore
import os
from tqdm import tqdm
import math


class Runner:
    def __init__(
        self,
        train_set,
        eval_set,
        model,
        loss,
        optimizer,
        global_step,
        lr,
        hp,
        logging,
    ):
        self.first1, self.first2 = train_set.iterator.get_next()
        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model
        self.loss = loss(model, self.first1, self.first2)
        self.optimizer = optimizer
        self.global_step = global_step
        self.lr = lr
        self.hp = hp
        self.logging = logging

        self.train_init_op = train_set.iterator.make_initializer(train_set.batches)
        self.eval_init_op = train_set.iterator.make_initializer(eval_set.batches)

        self.train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar("lr", lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        self.train_summaries = tf.summary.merge_all()

        self.y_hat, self.eval_summaries = self.model.eval(self.first1, self.first2)

        self.saver = tf.train.Saver(max_to_keep=hp.num_epochs)

    def train(self):
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.hp.logdir)
            if ckpt is None:
                self.logging.info("Initializing from scratch")
                sess.run(tf.global_variables_initializer())
                self.save_variable_specs(os.path.join(self.hp.logdir, "specs"))
            else:
                self.saver.restore(sess, ckpt)

            summary_writer = tf.summary.FileWriter(self.hp.logdir, sess.graph)

            sess.run(self.train_init_op)
            total_steps = self.hp.num_epochs * self.train_set.num_batches
            _gs = sess.run(self.global_step)
            for i in tqdm(range(_gs, total_steps + 1)):
                _, _gs, _summary = sess.run(
                    [self.train_op, self.global_step, self.train_summaries]
                )
                epoch = math.ceil(_gs / self.train_set.num_batches)
                summary_writer.add_summary(_summary, _gs)

                if _gs and _gs % self.train_set.num_batches == 0:
                    self.logging.info("epoch {} is done".format(epoch))
                    _loss = sess.run(self.loss)  # train loss

                    self.logging.info("# test evaluation")
                    _, _eval_summaries = sess.run(
                        [self.eval_init_op, self.eval_summaries]
                    )
                    summary_writer.add_summary(_eval_summaries, _gs)

                    self.logging.info("# get hypotheses")
                    hypotheses = get_hypotheses(
                        self.eval_set.num_batches,
                        self.eval_set.num_samples,
                        sess,
                        self.y_hat,
                        self.train_set.idx2token,
                    )

                    self.logging.info("# write results")
                    model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
                    if not os.path.exists(self.hp.evaldir):
                        os.makedirs(self.hp.evaldir)
                    translation = os.path.join(self.hp.evaldir, model_output)
                    with open(translation, "w") as fout:
                        fout.write("\n".join(hypotheses))

                    self.logging.info("# calc bleu score and append it to translation")
                    calc_bleu(self.hp.eval3, translation)

                    self.logging.info("# save models")
                    ckpt_name = os.path.join(self.hp.logdir, model_output)
                    self.saver.save(sess, ckpt_name, global_step=_gs)
                    self.logging.info(
                        "after training of {} epochs, {} has been saved.".format(
                            epoch, ckpt_name
                        )
                    )

                    self.logging.info("# fall back to train mode")
                    sess.run(self.train_init_op)
            summary_writer.close()

        self.logging.info("Done")

    def save_variable_specs(self, fpath):
        """Saves information about variables such as
        their name, shape, and total parameter number
        fpath: string. output file path

        Writes
        a text file named fpath.
        """

        def _get_size(shp):
            """Gets size of tensor shape
            shp: TensorShape

            Returns
            size
            """
            size = 1
            for d in range(len(shp)):
                size *= shp[d]
            return size

        params, num_params = [], 0
        for v in tf.global_variables():
            params.append("{}==={}".format(v.name, v.shape))
            num_params += _get_size(v.shape)
        print("num_params: ", num_params)
        with open(fpath, "w") as fout:
            fout.write("num_params: {}\n".format(num_params))
            fout.write("\n".join(params))
        self.logging.info("Variables info has been saved.")
