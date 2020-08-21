import tensorflow as tf
from ..pix2pix.utils.model import Pix2Pix
import cv2
from pix2pix.utils.dataset import normalize

test_image = cv2.imread('../TensorFlow Lite models/Testing images/p2p_test_a.jpg')

test_tensor = tf.expand_dims(tf.cast(test_image, tf.float32), axis=0)
test_tensor_res = normalize(tf.image.resize(test_tensor, [256, 256]))

p2p = Pix2Pix(mode='try', checkpoint_dir='../pix2pix/checkpoint/', for_tflite=True)

with tf.device('/cpu:0'):
    print('Running with CPU')
    tf_out_cpu = p2p.generator(test_tensor_res)
    try:
        tf.debugging.check_numerics(tf_out_cpu, 'CPU')
        print('\tThe model works fine.')
    except Exception as e:
        print('\t' + str(e))

with tf.device('/GPU:0'):
    print('Running with GPU')
    tf_out_gpu = p2p.generator(test_tensor_res)
    try:
        tf.debugging.check_numerics(tf_out_gpu, 'GPU')
        print('\tThe model works fine.')
    except Exception as e:
        print('\t' + str(e))