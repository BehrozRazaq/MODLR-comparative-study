# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import argparse
import pprint
import pdb

import torch
import torch.nn as nn


from roi_data_layer.roidb import combined_roidb
from model.utils.config import cfg, cfg_from_file, cfg_from_list


from dataset import Dataset
from models import vgg16, resnet
from runner import Runner
from loss import FasterRCNNLoss


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument(
        "--dataset",
        dest="dataset",
        help="training dataset",
        default="pascal_voc",
        type=str,
    )
    parser.add_argument(
        "--net", dest="net", help="vgg16, res101", default="vgg16", type=str
    )
    parser.add_argument(
        "--start_epoch", dest="start_epoch", help="starting epoch", default=1, type=int
    )
    parser.add_argument(
        "--epochs",
        dest="max_epochs",
        help="number of epochs to train",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--disp_interval",
        dest="disp_interval",
        help="number of iterations to display",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--checkpoint_interval",
        dest="checkpoint_interval",
        help="number of iterations to display",
        default=10000,
        type=int,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="directory to save models",
        default="models",
        type=str,
    )
    parser.add_argument(
        "--nw",
        dest="num_workers",
        help="number of workers to load data",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--cuda", dest="cuda", help="whether use CUDA", action="store_true"
    )
    parser.add_argument(
        "--ls",
        dest="large_scale",
        help="whether use large imag scale",
        action="store_true",
    )
    parser.add_argument(
        "--mGPUs", dest="mGPUs", help="whether use multiple GPUs", action="store_true"
    )
    parser.add_argument(
        "--bs", dest="batch_size", help="batch_size", default=1, type=int
    )
    parser.add_argument(
        "--cag",
        dest="class_agnostic",
        help="whether to perform class_agnostic bbox regression",
        action="store_true",
    )

    # config optimization
    parser.add_argument(
        "--o", dest="optimizer", help="training optimizer", default="sgd", type=str
    )
    parser.add_argument(
        "--lr", dest="lr", help="starting learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--lr_decay_step",
        dest="lr_decay_step",
        help="step to do learning rate decay, unit is epoch",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--lr_decay_gamma",
        dest="lr_decay_gamma",
        help="learning rate decay ratio",
        default=0.1,
        type=float,
    )

    # set training session
    parser.add_argument(
        "--s", dest="session", help="training session", default=1, type=int
    )

    # resume trained model
    parser.add_argument(
        "--r", dest="resume", help="resume checkpoint or not", default=False, type=bool
    )
    parser.add_argument(
        "--checksession",
        dest="checksession",
        help="checksession to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkepoch",
        dest="checkepoch",
        help="checkepoch to load model",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="checkpoint to load model",
        default=0,
        type=int,
    )
    # log and display
    parser.add_argument(
        "--use_tfb",
        dest="use_tfboard",
        help="whether use tensorboard",
        action="store_true",
    )

    args = parser.parse_args()

    print("Called with args:")
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = [
            "ANCHOR_SCALES",
            "[4, 8, 16, 32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "50",
        ]

    args.cfg_file = (
        "cfgs/{}_ls.yml".format(args.net)
        if args.large_scale
        else "cfgs/{}.yml".format(args.net)
    )

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    return args


if __name__ == "__main__":
    args = parse_args()

    print("Using config:")
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.cuda:
        cfg.CUDA = True

    dataset = Dataset(
        roidb,
        ratio_list,
        ratio_index,
        train_size,
        args.batch_size,
        imdb.num_classes,
        training=True,
        num_workers=args.num_workers,
    )

    # initilize the network here.
    if args.net == "vgg16":
        model = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == "res101":
        model = resnet(
            imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res50":
        model = resnet(
            imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
    elif args.net == "res152":
        model = resnet(
            imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    model.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]

    loss = FasterRCNNLoss(args.batch_size, args.class_agnostic)

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.cuda:
        model.cuda()

    if args.resume:
        load_name = os.path.join(
            output_dir,
            "faster_rcnn_{}_{}_{}.pth".format(
                args.checksession, args.checkepoch, args.checkpoint
            ),
        )
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint["session"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        if "pooling_mode" in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        model = nn.DataParallel(model)

    iters_per_epoch = int(train_size / args.batch_size)

    runner = Runner(
        dataset,
        model,
        loss,
        optimizer,
        args.cuda,
        args.max_epochs,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.mGPUs,
        iters_per_epoch,
        args.disp_interval,
        args.output_dir,
        args.use_tfboard,
        args.session,
        args.net,
        cfg.POOLING_MODE,
        args.class_agnostic,
        lr,
    )

    runner.train(args.start_epoch)
