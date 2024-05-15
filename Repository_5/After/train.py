"""
This script handles the training process.
"""

import argparse

import numpy as np
import random
import os

import torch
import torch.optim as optim

from dataset import DataIteratorHelper
from model import Transformer
from loss import CELoss
from optimizer import ScheduledOptim
from runner import Runner

__author__ = "Yu-Hsiang Huang"


def main():
    """
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -output_dir output -b 256 -warmup 128000
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-data_pkl", default=None)  # all-in-1 data pickle or bpe field

    parser.add_argument("-train_path", default=None)  # bpe encoded data
    parser.add_argument("-val_path", default=None)  # bpe encoded data

    parser.add_argument("-epoch", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=2048)

    parser.add_argument("-d_model", type=int, default=512)
    parser.add_argument("-d_inner_hid", type=int, default=2048)
    parser.add_argument("-d_k", type=int, default=64)
    parser.add_argument("-d_v", type=int, default=64)

    parser.add_argument("-n_head", type=int, default=8)
    parser.add_argument("-n_layers", type=int, default=6)
    parser.add_argument("-warmup", "--n_warmup_steps", type=int, default=4000)
    parser.add_argument("-lr_mul", type=float, default=2.0)
    parser.add_argument("-seed", type=int, default=None)

    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-embs_share_weight", action="store_true")
    parser.add_argument("-proj_share_weight", action="store_true")
    parser.add_argument("-scale_emb_or_prj", type=str, default="prj")

    parser.add_argument("-output_dir", type=str, default=None)
    parser.add_argument("-use_tb", action="store_true")
    parser.add_argument("-save_mode", type=str, choices=["all", "best"], default="best")

    parser.add_argument("-no_cuda", action="store_true")
    parser.add_argument("-label_smoothing", action="store_true")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # https://pytorch.org/docs/stable/notes/randomness.html
    # For reproducibility
    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    if not opt.output_dir:
        print("No experiment result will be saved.")
        raise

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    if opt.batch_size < 2048 and opt.n_warmup_steps <= 4000:
        print(
            "[Warning] The warmup steps may be not enough.\n"
            "(sz_b, warmup) = (2048, 4000) is the official setting.\n"
            "Using smaller batch w/o longer warmup may cause "
            "the warmup stage ends with only little data trained."
        )

    device = torch.device("cuda" if opt.cuda else "cpu")

    # ========= Loading Dataset =========#

    data_helper = DataIteratorHelper(opt, device)

    if all((opt.train_path, opt.val_path)):
        training_data, validation_data = (
            data_helper.prepare_dataloaders_from_bpe_files()
        )
    elif opt.data_pkl:
        training_data, validation_data = data_helper.prepare_dataloaders()
    else:
        raise

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.trg_vocab_size,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_trg_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout,
        scale_emb_or_prj=opt.scale_emb_or_prj,
    ).to(device)

    loss = CELoss(opt.trg_pad_idx)

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul,
        opt.d_model,
        opt.n_warmup_steps,
    )

    runner = Runner(
        training_data, validation_data, transformer, loss, optimizer, device, opt
    )

    runner.train()


if __name__ == "__main__":
    main()
