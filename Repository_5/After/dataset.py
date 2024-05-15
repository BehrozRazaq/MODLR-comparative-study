import dill as pickle  # type: ignore
from torchtext.data import Field, Dataset, BucketIterator  # type: ignore
from torchtext.datasets import TranslationDataset  # type: ignore
import transformer.Constants as Constants  # type: ignore


class DataIteratorHelper:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device

    def prepare_dataloaders_from_bpe_files(self):
        batch_size = self.opt.batch_size
        MIN_FREQ = 2
        if not self.opt.embs_share_weight:
            raise

        data = pickle.load(open(self.opt.data_pkl, "rb"))
        MAX_LEN = data["settings"].max_len
        field = data["vocab"]
        fields = (field, field)

        def filter_examples_with_length(x):
            return len(vars(x)["src"]) <= MAX_LEN and len(vars(x)["trg"]) <= MAX_LEN

        train = TranslationDataset(
            fields=fields,
            path=self.opt.train_path,
            exts=(".src", ".trg"),
            filter_pred=filter_examples_with_length,
        )

        val = TranslationDataset(
            fields=fields,
            path=self.opt.val_path,
            exts=(".src", ".trg"),
            filter_pred=filter_examples_with_length,
        )

        self.opt.max_token_seq_len = MAX_LEN + 2
        self.opt.src_pad_idx = self.opt.trg_pad_idx = field.vocab.stoi[
            Constants.PAD_WORD
        ]
        self.opt.src_vocab_size = self.opt.trg_vocab_size = len(field.vocab)

        train_iterator = BucketIterator(
            train, batch_size=batch_size, device=self.device, train=True
        )
        val_iterator = BucketIterator(val, batch_size=batch_size, device=self.device)
        return train_iterator, val_iterator

    def prepare_dataloaders(self):
        batch_size = self.opt.batch_size
        data = pickle.load(open(self.opt.data_pkl, "rb"))

        self.opt.max_token_seq_len = data["settings"].max_len
        self.opt.src_pad_idx = data["vocab"]["src"].vocab.stoi[Constants.PAD_WORD]
        self.opt.trg_pad_idx = data["vocab"]["trg"].vocab.stoi[Constants.PAD_WORD]

        self.opt.src_vocab_size = len(data["vocab"]["src"].vocab)
        self.opt.trg_vocab_size = len(data["vocab"]["trg"].vocab)

        # ========= Preparing Model =========#
        if self.opt.embs_share_weight:
            assert (
                data["vocab"]["src"].vocab.stoi == data["vocab"]["trg"].vocab.stoi
            ), "To sharing word embedding the src/trg word2idx table shall be the same."

        fields = {"src": data["vocab"]["src"], "trg": data["vocab"]["trg"]}

        train = Dataset(examples=data["train"], fields=fields)
        val = Dataset(examples=data["valid"], fields=fields)

        train_iterator = BucketIterator(
            train, batch_size=batch_size, device=self.device, train=True
        )
        val_iterator = BucketIterator(val, batch_size=batch_size, device=self.device)

        return train_iterator, val_iterator
