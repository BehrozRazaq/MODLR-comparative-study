import os
import tensorflow as tf

from dataset import Dataset
from models import Generator, Discriminator
from loss import GANLoss
from runner import Runner

_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz"

path_to_zip = tf.keras.utils.get_file("facades.tar.gz", origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), "facades/")


BUFFER_SIZE = 400
EPOCHS = 200
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3


dataset = Dataset(
    PATH + "train/*.jpg",
    PATH + "test/*.jpg",
    IMG_HEIGHT,
    IMG_WIDTH,
    BUFFER_SIZE,
    BATCH_SIZE,
)

generator = Generator()
discriminator = Discriminator()

loss = GANLoss()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

runner = Runner(
    dataset,
    generator,
    discriminator,
    loss,
    generator_optimizer,
    discriminator_optimizer,
    EPOCHS,
    checkpoint_dir="./training_checkpoints",
)

runner.train()
