import numpy as np
import torch

from torch.autograd import Variable
from torch import Tensor
from torch.nn import Module


class GANLoss:
    def __init__(self, latent_dim: int, n_classes: int, cuda: bool) -> None:

        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.CrossEntropyLoss()

        self.cached_values = {}

        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        if bool(torch.cuda.is_available()):
            self.adversarial_loss.cuda()
            self.auxiliary_loss.cuda()

    def __call__(
        self,
        generator: Module,
        discriminator: Module,
        features: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:

        batch_size = features.shape[0]

        valid = Variable(
            self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False
        )
        fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(features.type(self.FloatTensor))

        z = Variable(
            self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim)))
        )
        gen_labels = Variable(
            self.LongTensor(np.random.randint(0, self.n_classes, batch_size))
        )

        gen_imgs = generator(z, gen_labels)
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (
            self.adversarial_loss(validity, valid)
            + self.auxiliary_loss(pred_label, gen_labels)
        )

        real_pred, real_aux = discriminator(real_imgs)
        real_pred = real_pred.to("cuda")
        valid = valid.to("cuda")
        real_aux = real_aux.to("cuda")
        labels = labels.to("cuda")
        d_real_loss = (
            self.adversarial_loss(real_pred, valid)
            + self.auxiliary_loss(real_aux, labels)
        ) / 2

        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (
            self.adversarial_loss(fake_pred, fake)
            + self.auxiliary_loss(fake_aux, gen_labels)
        ) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2

        self.cached_values = {
            "real_aux": real_aux,
            "fake_aux": fake_aux,
            "gen_labels": gen_labels,
        }

        return d_loss, g_loss
