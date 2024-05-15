# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
"""
import tensorflow as tf


class Dataset:

    def __init__(
        self,
        fpath1,
        fpath2,
        maxlen1,
        maxlen2,
        vocab_fpath,
        batch_size,
        shuffle=False,
    ):
        self.fpath1 = fpath1
        self.fpath2 = fpath2
        self.maxlen1 = maxlen1
        self.maxlen2 = maxlen2
        self.vocab_fpath = vocab_fpath
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batches, self.num_batches, self.num_samples = self.get_batches(
            fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle
        )

        self.iterator = tf.data.Iterator.from_structure(
            self.batches.output_types, self.batches.output_shapes
        )

        self.token2idx, self.idx2token = self.load_vocab(vocab_fpath)

    def calc_num_batches(self, total_num, batch_size):
        """Calculates the number of batches.
        total_num: total sample number
        batch_size

        Returns
        number of batches, allowing for remainders."""
        return total_num // batch_size + int(total_num % batch_size != 0)

    @classmethod
    def load_vocab(cls, vocab_fpath):
        """Loads vocabulary file and returns idx<->token maps
        vocab_fpath: string. vocabulary file path.
        Note that these are reserved
        0: <pad>, 1: <unk>, 2: <s>, 3: </s>

        Returns
        two dictionaries.
        """
        vocab = [line.split()[0] for line in open(vocab_fpath, "r").read().splitlines()]
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    @classmethod
    def load_data(cls, fpath1, fpath2, maxlen1, maxlen2):
        """Loads source and target data and filters out too lengthy samples.
        fpath1: source file path. string.
        fpath2: target file path. string.
        maxlen1: source sent maximum length. scalar.
        maxlen2: target sent maximum length. scalar.

        Returns
        sents1: list of source sents
        sents2: list of target sents
        """
        sents1, sents2 = [], []
        with open(fpath1, "r") as f1, open(fpath2, "r") as f2:
            for sent1, sent2 in zip(f1, f2):
                if len(sent1.split()) + 1 > maxlen1:
                    continue  # 1: </s>
                if len(sent2.split()) + 1 > maxlen2:
                    continue  # 1: </s>
                sents1.append(sent1.strip())
                sents2.append(sent2.strip())
        return sents1, sents2

    @classmethod
    def encode(cls, inp, type, dict):
        """Converts string to number. Used for `generator_fn`.
        inp: 1d byte array.
        type: "x" (source side) or "y" (target side)
        dict: token2idx dictionary

        Returns
        list of numbers
        """
        inp_str = inp.decode("utf-8")
        if type == "x":
            tokens = inp_str.split() + ["</s>"]
        else:
            tokens = ["<s>"] + inp_str.split() + ["</s>"]

        x = [dict.get(t, dict["<unk>"]) for t in tokens]
        return x

    @classmethod
    def generator_fn(cls, sents1, sents2, vocab_fpath):
        """Generates training / evaluation data
        sents1: list of source sents
        sents2: list of target sents
        vocab_fpath: string. vocabulary file path.

        yields
        xs: tuple of
            x: list of source token ids in a sent
            x_seqlen: int. sequence length of x
            sent1: str. raw source (=input) sentence
        labels: tuple of
            decoder_input: decoder_input: list of encoded decoder inputs
            y: list of target token ids in a sent
            y_seqlen: int. sequence length of y
            sent2: str. target sentence
        """
        token2idx, _ = cls.load_vocab(vocab_fpath)
        for sent1, sent2 in zip(sents1, sents2):
            x = cls.encode(sent1, "x", token2idx)
            y = cls.encode(sent2, "y", token2idx)
            decoder_input, y = y[:-1], y[1:]

            x_seqlen, y_seqlen = len(x), len(y)
            yield (x, x_seqlen, sent1), (decoder_input, y, y_seqlen, sent2)

    @classmethod
    def input_fn(cls, sents1, sents2, vocab_fpath, batch_size, shuffle=False):
        """Batchify data
        sents1: list of source sents
        sents2: list of target sents
        vocab_fpath: string. vocabulary file path.
        batch_size: scalar
        shuffle: boolean

        Returns
        xs: tuple of
            x: int32 tensor. (N, T1)
            x_seqlens: int32 tensor. (N,)
            sents1: str tensor. (N,)
        ys: tuple of
            decoder_input: int32 tensor. (N, T2)
            y: int32 tensor. (N, T2)
            y_seqlen: int32 tensor. (N, )
            sents2: str tensor. (N,)
        """
        shapes = (([None], (), ()), ([None], [None], (), ()))
        types = (
            (tf.int32, tf.int32, tf.string),
            (tf.int32, tf.int32, tf.int32, tf.string),
        )
        paddings = ((0, 0, ""), (0, 0, 0, ""))

        dataset = tf.data.Dataset.from_generator(
            cls.generator_fn,
            output_shapes=shapes,
            output_types=types,
            args=(sents1, sents2, vocab_fpath),
        )  # <- arguments for generator_fn. converted to np string arrays

        if shuffle:  # for training
            dataset = dataset.shuffle(128 * batch_size)

        dataset = dataset.repeat()  # iterate forever
        dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

        return dataset

    def get_batches(
        self, fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False
    ):
        """Gets training / evaluation mini-batches
        fpath1: source file path. string.
        fpath2: target file path. string.
        maxlen1: source sent maximum length. scalar.
        maxlen2: target sent maximum length. scalar.
        vocab_fpath: string. vocabulary file path.
        batch_size: scalar
        shuffle: boolean

        Returns
        batches
        num_batches: number of mini-batches
        num_samples
        """
        sents1, sents2 = self.__class__.load_data(fpath1, fpath2, maxlen1, maxlen2)
        batches = self.__class__.input_fn(
            sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle
        )
        num_batches = self.calc_num_batches(len(sents1), batch_size)
        return batches, num_batches, len(sents1)
