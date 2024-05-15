import numpy as np
import torch

from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from torchvision.utils import save_image  # type: ignore

from .loss import GANLoss


class GANRunner:
    def __init__(
        self,
        dataloader: DataLoader,
        generator: Module,
        discriminator: Module,
        optimizer_G: Optimizer,
        optimizer_D: Optimizer,
        loss: GANLoss,
        cuda: bool,
        latent_dim: int,
        n_epochs: int,
        sample_interval: int,
    ):
        self.dataloader = dataloader
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.loss = loss
        self.cuda = cuda
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.sample_interval = sample_interval

        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    def sample_image(self, n_row: int, batches_done: int) -> None:
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(
            self.FloatTensor(np.random.normal(0, 1, (n_row**2, self.latent_dim)))
        )
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(self.LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_image(
            gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True
        )

    def calculate_metrics(self, labels: Tensor) -> float:
        cached_values = self.loss.cached_values
        real_aux, fake_aux, gen_labels = (
            cached_values["real_aux"],
            cached_values["fake_aux"],
            cached_values["gen_labels"],
        )
        pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        gt = np.concatenate(
            [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
        )
        return np.mean(np.argmax(pred, axis=1) == gt)

    def train(self):
        for epoch in range(self.n_epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                self.optimizer_G.zero_grad()
                self.optimizer_D.zero_grad()

                d_loss, g_loss = self.loss(
                    self.generator, self.discriminator, imgs, labels
                )

                g_loss.backward()
                self.optimizer_G.step()

                d_acc = self.calculate_metrics(labels)
                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (
                        epoch,
                        self.n_epochs,
                        i,
                        len(self.dataloader),
                        d_loss.item(),
                        100 * d_acc,
                        g_loss.item(),
                    )
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    self.sample_image(5, batches_done)
