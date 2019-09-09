import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from yolo.detect import prepare_detector
from yolo.yolov3.utils import draw_outputs
import cv2
from yolo.yolov3.dataset import transform_images

image = 'input/training/images/_-6wpIfarPnFg-9RN4Y1mA.jpg'
image = 'input/remove_people/2.png'
classes = './yolo/data/coco.names'
size = 416
class_names = [c.strip() for c in open(classes).readlines()]

img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
img = tf.expand_dims(img, 0)
img = transform_images(img, size)

yolo = prepare_detector()

boxes, scores, classes, nums = yolo(img) # detect(image, debug=True) would be fine too, but only for debug purposes

img_2 = cv2.imread(image)
img = draw_outputs(img_2, (boxes, scores, classes, nums), class_names)
