import os
import time
import math
import torch

from tqdm import tqdm


class Runner:

    def __init__(
        self, train_dataloader, val_dataloader, model, loss, optimizer, opt, device
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.opt = opt
        self.device = device

    def patch_src(self, src, pad_idx):
        src = src.transpose(0, 1)
        return src

    def patch_trg(self, trg, pad_idx):
        trg = trg.transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        return trg, gold

    def train_epoch(self):
        """Epoch operation in training phase"""

        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = "  - (Training)   "
        for batch in tqdm(self.train_dataloader, mininterval=2, desc=desc, leave=False):

            # prepare data
            src_seq = self.patch_src(batch.src, self.opt.src_pad_idx).to(self.device)
            trg_seq, gold = map(
                lambda x: x.to(self.device),
                self.patch_trg(batch.trg, self.opt.trg_pad_idx),
            )

            # forward
            self.optimizer.zero_grad()

            # backward and update parameters
            loss, n_correct, n_word = self.loss(
                self.model, src_seq, trg_seq, gold, smoothing=self.opt.smoothing
            )
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def eval_epoch(self):
        """Epoch operation in evaluation phase"""

        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0

        desc = "  - (Validation) "
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader, mininterval=2, desc=desc, leave=False
            ):

                # prepare data
                src_seq = self.patch_src(batch.src, self.opt.src_pad_idx).to(
                    self.device
                )
                trg_seq, gold = map(
                    lambda x: x.to(self.device),
                    self.patch_trg(batch.trg, self.opt.trg_pad_idx),
                )

                # forward
                pred = self.model(src_seq, trg_seq)
                loss, n_correct, n_word = self.loss(
                    pred, gold, self.opt.trg_pad_idx, smoothing=False
                )

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def train(self):
        """Start training"""

        # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
        if self.opt.use_tb:
            print("[Info] Use Tensorboard")
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tb_writer = SummaryWriter(
                log_dir=os.path.join(self.opt.output_dir, "tensorboard")
            )

        log_train_file = os.path.join(self.opt.output_dir, "train.log")
        log_valid_file = os.path.join(self.opt.output_dir, "valid.log")

        print(
            "[Info] Training performance will be written to file: {} and {}".format(
                log_train_file, log_valid_file
            )
        )

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,loss,ppl,accuracy\n")
            log_vf.write("epoch,loss,ppl,accuracy\n")

        def print_performances(header, ppl, accu, start_time, lr):
            print(
                "  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, "
                "elapse: {elapse:3.3f} min".format(
                    header=f"({header})",
                    ppl=ppl,
                    accu=100 * accu,
                    elapse=(time.time() - start_time) / 60,
                    lr=lr,
                )
            )

        # valid_accus = []
        valid_losses = []
        for epoch_i in range(self.opt.epoch):
            print("[ Epoch", epoch_i, "]")

            start = time.time()
            train_loss, train_accu = self.train_epoch()
            train_ppl = math.exp(min(train_loss, 100))
            # Current learning rate
            lr = self.optimizer._optimizer.param_groups[0]["lr"]
            print_performances("Training", train_ppl, train_accu, start, lr)

            start = time.time()
            valid_loss, valid_accu = self.eval_epoch()
            valid_ppl = math.exp(min(valid_loss, 100))
            print_performances("Validation", valid_ppl, valid_accu, start, lr)

            valid_losses += [valid_loss]

            checkpoint = {
                "epoch": epoch_i,
                "settings": self.opt,
                "model": self.model.state_dict(),
            }

            if self.opt.save_mode == "all":
                model_name = "model_accu_{accu:3.3f}.chkpt".format(
                    accu=100 * valid_accu
                )
                torch.save(checkpoint, model_name)
            elif self.opt.save_mode == "best":
                model_name = "model.chkpt"
                if valid_loss <= min(valid_losses):
                    torch.save(
                        checkpoint, os.path.join(self.opt.output_dir, model_name)
                    )
                    print("    - [Info] The checkpoint file has been updated.")

            with open(log_train_file, "a") as log_tf, open(
                log_valid_file, "a"
            ) as log_vf:
                log_tf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=train_loss,
                        ppl=train_ppl,
                        accu=100 * train_accu,
                    )
                )
                log_vf.write(
                    "{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n".format(
                        epoch=epoch_i,
                        loss=valid_loss,
                        ppl=valid_ppl,
                        accu=100 * valid_accu,
                    )
                )

            if self.opt.use_tb:
                tb_writer.add_scalars(
                    "ppl", {"train": train_ppl, "val": valid_ppl}, epoch_i
                )
                tb_writer.add_scalars(
                    "accuracy",
                    {"train": train_accu * 100, "val": valid_accu * 100},
                    epoch_i,
                )
                tb_writer.add_scalar("learning_rate", lr, epoch_i)
