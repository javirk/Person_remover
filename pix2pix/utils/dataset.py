import tensorflow as tf
from random import triangular, randint
import matplotlib.pyplot as plt
import numpy as np

def load(image_file):
    real_image = tf.io.read_file(image_file)
    real_image = tf.image.decode_jpeg(real_image)

    real_image = tf.cast(real_image, tf.float32)

    return real_image

def resize(real_image, height, width):
    real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return real_image

# normalizing the images to [-1, 1]
def normalize(real_image):
    real_image = (real_image / 127.5) - 1
    # input_image = (input_image / 127.5) - 1

    return real_image

@tf.function
def remove_portion(image_file, x_dim, y_dim):
    x_0 = int(triangular(x_dim / 3, x_dim * 4/5))
    y_0 = int(triangular(0, y_dim * 4/5))
    w =  int(min(y_dim / 6, randint(20, y_dim - 1 - y_0)))
    h =  int(max(w, randint(0, x_dim - 1 - x_0)))

    a = [[[False if (i < y_0 or i > y_0 + w) or (j < x_0 or j > x_0 + h) else True for _ in range(0,3)] for i in range(0, y_dim)] for j in range(0, x_dim)]
    a_t = tf.constant(a, dtype=tf.double)

    indices = tf.where(a_t)
    pixels = np.sum(np.array(a))
    update = tf.zeros((pixels))
    image_file = tf.tensor_scatter_nd_update(image_file, indices, update)

    return image_file

def load_image(image_file, height=256, width=256):
    real_image = load(image_file)
    real_image = resize(real_image, height, width)
    real_image = normalize(real_image)
    input_image = remove_portion(real_image, height, width)

    return input_image, real_image

def train_pipeline(PATH, BUFFER_SIZE):
    train_dataset = tf.data.Dataset.list_files(PATH + '*.jpg')
    train_dataset = train_dataset.map(load_image,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.map(load_image)
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(1)

    return train_dataset

def test_pipeline(PATH):
    test_dataset = tf.data.Dataset.list_files(PATH + '*.jpg')
    test_dataset = test_dataset.map(load_image)
    test_dataset = test_dataset.batch(1)

    return test_dataset

if __name__ == '__main__':
    image = '../../input/training/images/_-6wpIfarPnFg-9RN4Y1mA.jpg'
    input_im, real_image = load_image(image)