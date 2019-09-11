# Common functions
import tensorflow as tf
from pix2pix.utils.dataset import remove_portion, normalize
from yolo.yolov3.dataset import transform_images
import numpy as np
import math

def read_image(input_file):
    img = tf.io.read_file(input_file)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)

    return img


def prepare_image_yolo(img, size=416):
    img = tf.expand_dims(img, 0)
    img = transform_images(img, size)

    return img

def prepare_frame_p2p(frame):
    frame = tf.cast(frame, tf.float32)

    return frame


def cut_result(output):
    '''
    This function reshapes the tf tensors.
    '''
    boxes, scores, classes, nums = output[0], output[1], output[2], output[3]

    amount = nums.numpy()[0]
    boxes = boxes[0, :amount, :]
    scores = scores[0, :amount]
    classes = classes[0, :amount]

    return [boxes, scores, classes, nums]


def expand_box(x1y1, x2y2, coeff_area=1.05):
    coeff_length = math.sqrt(coeff_area)

    x1y1 = (int(x1y1[0] * (2 - coeff_length)), int(x1y1[1] * (2 - coeff_length)))
    x2y2 = (int(x2y2[0] * coeff_length), int(x2y2[1] * coeff_length))

    return x1y1, x2y2


def get_person_image(img, box, expand=False, HEIGHT=256, WIDTH=256):
    wh = np.flip(img.shape[0:2])
    # Posiciones de la caja. x1y1 es abajo a la izquierda y x2y2 es arriba a la derecha
    x1y1 = tuple((np.array(box[0:2]) * wh).astype(np.int32))
    x2y2 = tuple((np.array(box[2:4]) * wh).astype(np.int32))

    if expand:
        x1y1, x2y2 = expand_box(x1y1, x2y2)

    box_width = x2y2[1] - x1y1[1]
    box_height = x2y2[0] - x1y1[0]

    box_border_1 = (x1y1[0] - box_height // 2, x1y1[1] - box_width // 2)
    box_border_2 = (x2y2[0] + box_height // 2, x2y2[1] + box_width // 2)

    # im_cut = img[box_border_1[1]:box_border_2[1], box_border_1[0]:box_border_2[0], :]

    im_cut = img.numpy().take(range(box_border_1[1], box_border_2[1]), mode='wrap', axis=0).take(
                        range(box_border_1[0], box_border_2[0]), mode='wrap', axis=1)

    im_cut = tf.image.resize(im_cut, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # return im_cut, box_border_1, box_border_2
    return im_cut, x1y1, x2y2


def generate_fake(img, model, HEIGHT=256, WIDTH=256):
    # img = tf.image.resize(img, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = normalize(img)
    fake = remove_portion(img, HEIGHT, WIDTH)
    fake = tf.expand_dims(fake, 0)

    return model.generator(fake)[0]


def insert_into_image(img_real, img_fake, x1y1, x2y2, HEIGHT=256, WIDTH=256):
    img_fake = img_fake[WIDTH // 4:WIDTH * 3 // 4, HEIGHT // 4:HEIGHT * 3 // 4, :]
    x1y1 = (max(0, x1y1[0]), max(0, x1y1[1]))
    x2y2 = (min(img_real.shape[1], x2y2[0]), min(img_real.shape[0], x2y2[1]))

    box_width = x2y2[1] - x1y1[1]
    box_height = x2y2[0] - x1y1[0]
    img_fake_resized = tf.image.resize(img_fake, [box_width, box_height], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    img_real[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0], :] = img_fake_resized[:]

    return img_real


def insert_blocks(img_real, blocks, coords):
    img_numpy = img_real.numpy()
    for block, coord in zip(blocks, coords):
        x1y1 = coord[0]
        x2y2 = coord[1]
        img_numpy = insert_into_image(img_numpy, block, x1y1, x2y2)

    return img_numpy


def create_new_image(img_real, output_yolo, p2p, objects):
    img_fake = normalize(img_real)
    boxes, scores, classes, nums = output_yolo[0], output_yolo[1], output_yolo[2], output_yolo[3]
    blocks = []
    coords = []

    for i in range(nums[0]):
        if classes[i].numpy() in objects:
            im_cut, x1y1, x2y2 = get_person_image(img_real, boxes[i])
            fake_block = generate_fake(im_cut, p2p)

            blocks.append(fake_block)
            coords.append((x1y1, x2y2))

    new = insert_blocks(img_fake, blocks, coords)
    return new

def frame_to_int(frame):
    frame = (frame * 0.5 + 0.5) * 255.
    frame = frame.astype(np.uint8)

    return frame