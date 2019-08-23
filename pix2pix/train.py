import tensorflow as tf
from pix2pix.utils.model import Pix2Pix
from pix2pix.utils.dataset import train_pipeline, test_pipeline

BUFFER_SIZE = 100
BATCH_SIZE = 1
IMG_WIDTH = 416
IMG_HEIGHT = 416
LAMBDA = 10
EPOCHS = 100
PATH_training = '../input/training/prueba/'
PATH_testing = '../input/testing/prueba/'
checkpoint_dir = '../pix2pix/checkpoint/'

train_dataset = train_pipeline(PATH_training, BUFFER_SIZE)
test_dataset = test_pipeline(PATH_testing)

p2p = Pix2Pix(train_dataset, test_dataset, LAMBDA, EPOCHS, checkpoint_dir)