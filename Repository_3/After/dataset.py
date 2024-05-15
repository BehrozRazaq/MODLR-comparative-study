from altair import sample
import torch
from torch.utils.data import DataLoader, Sampler

from roidata_layer import roibatchLoader


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(
                self.num_per_batch * batch_size, train_size
            ).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = (
            rand_num.expand(self.num_per_batch, self.batch_size) + self.range
        )

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


class Dataset(DataLoader):
    def __init__(
        self,
        roidb,
        ratio_list,
        ratio_index,
        train_size,
        batch_size,
        num_classes,
        training,
        num_workers,
    ):
        sampler_batch = sampler(train_size, batch_size)

        dataset = roibatchLoader(
            roidb, ratio_list, ratio_index, batch_size, num_classes, training=training
        )

        dataset = dataset

        super(Dataset, self).__init__(
            dataset,
            batch_size=batch_size,
            sampler=sampler_batch,
            num_workers=num_workers,
        )
