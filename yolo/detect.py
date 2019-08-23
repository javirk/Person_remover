## ALL CODE OF YOLOV3 COMES FROM: https://github.com/zzh8829/yolov3-tf2

import time
from absl import app, flags, logging
import cv2
import numpy as np
from yolo.yolov3.models import (
    YoloV3, YoloV3Tiny
)
from yolo.yolov3.utils import draw_outputs

def prepare_detector(num_classes=80, tiny=False, weights='./yolo/checkpoints/yolov3.tf'):
    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights)
    logging.info('weights loaded')

    return yolo

def detect(image, model, class_names='./yolo/data/coco.names', debug=False):
    if not isinstance(class_names, list):
        class_names = [c.strip() for c in open(class_names).readlines()]
    logging.info('classes loaded')

    if debug:
        t1 = time.time()
        boxes, scores, classes, nums = model(image)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.imread(image)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    else:
        return model(image)