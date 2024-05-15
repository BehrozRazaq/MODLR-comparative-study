import torch
import time
import os
from torch.autograd import Variable

from models.utils.net_utils import (
    adjust_learning_rate,
    save_checkpoint,
    clip_gradient,
)

from loss import FasterRCNNLoss


class Runner:
    def __init__(
        self,
        dataloader,
        model,
        loss,
        optimizer,
        cuda,
        max_epochs,
        lr_decay_step,
        lr_decay_gamma,
        mGPUs,
        iters_per_epoch,
        disp_interval,
        output_dir,
        use_tfboard,
        session,
        net,
        POOLING_MODE,
        class_agnostic,
        lr,
    ):
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.max_epochs = max_epochs
        self.lr_decay_step = lr_decay_step
        self.lr_decay_gamma = lr_decay_gamma
        self.mGPUs = mGPUs
        self.iters_per_epoch = iters_per_epoch
        self.disp_interval = disp_interval
        self.output_dir = output_dir
        self.use_tfboard = use_tfboard
        self.session = session
        self.net = net
        self.POOLING_MODE = POOLING_MODE
        self.class_agnostic = class_agnostic
        self.lr = lr

        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        self.im_data = Variable(self.im_data)
        self.im_info = Variable(self.im_info)
        self.num_boxes = Variable(self.num_boxes)
        self.gt_boxes = Variable(self.gt_boxes)

        if use_tfboard:
            from tensorboardX import SummaryWriter

            self.logger = SummaryWriter("logs")

    def train(self, start_epoch):
        for epoch in range(start_epoch, self.max_epochs + 1):
            # setting to train mode
            self.model.train()
            loss_temp = 0
            start = time.time()

            if epoch % (self.lr_decay_step + 1) == 0:
                adjust_learning_rate(self.optimizer, self.lr_decay_gamma)
                self.lr *= self.lr_decay_gamma

            data_iter = iter(self.dataloader)
            for step in range(self.iters_per_epoch):
                data = next(data_iter)
                self.im_data.data.resize_(data[0].size()).copy_(data[0])
                self.im_info.data.resize_(data[1].size()).copy_(data[1])
                self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

                loss, (
                    rpn_loss_cls,
                    rpn_loss_box,
                    RCNN_loss_cls,
                    RCNN_loss_bbox,
                    rois_label,
                ) = self.loss(
                    self.model,
                    self.im_data,
                    self.im_info,
                    self.gt_boxes,
                    self.num_boxes,
                )
                loss_temp += loss.item()

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                if self.net == "vgg16":
                    clip_gradient(self.model, 10.0)
                self.optimizer.step()

                if step % self.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= self.disp_interval + 1

                    if self.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print(
                        "[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e"
                        % (
                            self.session,
                            epoch,
                            step,
                            self.iters_per_epoch,
                            loss_temp,
                            self.lr,
                        )
                    )
                    print(
                        "\t\t\tfg/bg=(%d/%d), time cost: %f"
                        % (fg_cnt, bg_cnt, end - start)
                    )
                    print(
                        "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                        % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box)
                    )
                    if self.use_tfboard:
                        info = {
                            "loss": loss_temp,
                            "loss_rpn_cls": loss_rpn_cls,
                            "loss_rpn_box": loss_rpn_box,
                            "loss_rcnn_cls": loss_rcnn_cls,
                            "loss_rcnn_box": loss_rcnn_box,
                        }
                        self.logger.add_scalars(
                            "logs_s_{}/losses".format(self.session),
                            info,
                            (epoch - 1) * self.iters_per_epoch + step,
                        )

                    loss_temp = 0
                    start = time.time()

            save_name = os.path.join(
                self.output_dir,
                "faster_rcnn_{}_{}_{}.pth".format(self.session, epoch, step),
            )
            save_checkpoint(
                {
                    "session": self.session,
                    "epoch": epoch + 1,
                    "model": (
                        self.model.module.state_dict()
                        if self.mGPUs
                        else self.model.state_dict()
                    ),
                    "optimizer": self.optimizer.state_dict(),
                    "pooling_mode": self.POOLING_MODE,
                    "class_agnostic": self.class_agnostic,
                },
                save_name,
            )
            print("save model: {}".format(save_name))

        if self.use_tfboard:
            self.logger.close()
