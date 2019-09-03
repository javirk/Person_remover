import tensorflow as tf
from random import triangular, randint
import numpy as np

def load(image_file):
    real_image = tf.io.read_file(image_file)
    try:
        real_image = tf.image.decode_png(real_image, channels=3)
    except:
        real_image = tf.image.decode_jpeg(real_image, channels=3)

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
def remove_portion(image_file, x_dim, y_dim, fancy_portion=False):
    if fancy_portion:
        x_0 = int(triangular(x_dim / 3, x_dim * 4/5))
        y_0 = int(triangular(0, y_dim * 4/5))
        w =  int(min(y_dim / 6, randint(20, y_dim - 1 - y_0)))
        h =  int(max(w, randint(0, x_dim - 1 - x_0)))
    else:
        h = int(x_dim / 2)
        w = int(y_dim / 2)
        x_0 = int(x_dim / 4)
        y_0 = int(y_dim / 4)

    a = [[[False if (i < y_0 or i > y_0 + w) or (j < x_0 or j > x_0 + h) else True for _ in range(0,3)] for i in range(0, y_dim)] for j in range(0, x_dim)]
    a_t = tf.constant(a, dtype=tf.double)

    indices = tf.where(a_t)
    pixels = np.sum(np.array(a))
    # update = tf.zeros((pixels))
    update = tf.ones((pixels))
    image_file = tf.tensor_scatter_nd_update(image_file, indices, update)

    return image_file

def random_crop(image, height, width):
  cropped_image = tf.image.random_crop(image, size=[height, width, 3])

  return cropped_image

@tf.function()
def random_jitter(image, height, width):
  # resizing to 286 x 286 x 3
  image = resize(image, 286, 286)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image, height, width)

  if tf.random.uniform(()) > 0.5:
        # random mirroring
        image = tf.image.flip_left_right(image)

  return image

def load_image(image_file, width=256, height=256, jitter=False):
    real_image = load(image_file)
    real_image = resize(real_image, height, width)
    if jitter:
        real_image = random_jitter(real_image, height, width)
    real_image = normalize(real_image)
    input_image = remove_portion(real_image, height, width)

    return input_image, real_image

def train_pipeline(PATH, BUFFER_SIZE, WIDTH, HEIGHT, n, BATCH_SIZE):
    train_dataset = tf.data.Dataset.list_files(PATH + '*.JPG')
    train_dataset = train_dataset.take(n)
    train_dataset = train_dataset.map(lambda x: load_image(x, HEIGHT, WIDTH, False),
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset

def test_pipeline(PATH, WIDTH, HEIGHT, n):
    test_dataset = tf.data.Dataset.list_files(PATH + '*.jpg')
    if n != -1:
        test_dataset = test_dataset.take(int(n * 0.25))
    else:
        test_dataset = test_dataset.take(n)
    test_dataset = test_dataset.map(lambda x: load_image(x, HEIGHT, WIDTH, False))
    # test_dataset = test_dataset.map(load_image)
    test_dataset = test_dataset.batch(1)

    return test_dataset