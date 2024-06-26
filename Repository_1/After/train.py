import argparse
import os
import torch

import torchvision.transforms as transforms  # type: ignore
from torchvision import datasets  # type: ignore

from .runner import GANRunner
from .loss import GANLoss
from .models import Discriminator, Generator

DATA_PATH = "./data/mnist"

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=200, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--n_classes", type=int, default=10, help="number of classes for dataset"
)
parser.add_argument(
    "--img_size", type=int, default=32, help="size of each image dimension"
)
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval between image sampling"
)
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Configure data loader
os.makedirs(DATA_PATH, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        DATA_PATH,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Initialize generator and discriminator
generator = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.img_size)
discriminator = Discriminator(opt.channels, opt.n_classes, opt.img_size)

if cuda:
    generator.cuda()
    discriminator.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

loss = GANLoss(cuda, opt.latent_dim, opt.n_classes)

runner = GANRunner(
    dataloader,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    loss,
    cuda,
    opt.latent_dim,
    opt.n_epochs,
    opt.sample_interval,
)

runner.train()
