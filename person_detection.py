from yolo.detect import prepare_detector
from yolo.yolov3.utils import draw_outputs
import cv2
import tensorflow as tf
from yolo.yolov3.dataset import transform_images

image = 'input/training/images/_-6wpIfarPnFg-9RN4Y1mA.jpg'
classes='./yolo/data/coco.names'
size = 416
class_names = [c.strip() for c in open(classes).readlines()]

img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
img = tf.expand_dims(img, 0)
img = transform_images(img, size)

yolo = prepare_detector()

boxes, scores, classes, nums = yolo(img) # detect(image, debug=True) would be fine too, but only for debug purposes

# img = cv2.imread(image)
# img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
