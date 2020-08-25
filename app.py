from flask import Flask, request, send_file
from flask_cors import CORS, cross_origin
from yolo.detect import prepare_detector
from pix2pix.utils.model import Pix2Pix
from utils import prepare_image_yolo, cut_result, create_new_image
import numpy as np
import io
import cv2
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    original_image_bytes = request.files['image']

    # Decode array
    in_memory_file = io.BytesIO()
    original_image_bytes.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    original_image = cv2.imdecode(data, 1) # This decodes in BGR
    original_image = original_image[:,:,::-1] # This sorts channels to RGB (faster than cv2.cvtColor)

    # Prepare image for YOLO
    original_image = tf.cast(original_image, tf.float32)

    image_for_yolo = prepare_image_yolo(original_image)
    output_yolo = yolo(image_for_yolo)  # Detect objects
    output_yolo = cut_result(output_yolo)  # Prepare YOLO outputs

    # Rewrite people
    final_image = create_new_image(original_image, output_yolo, p2p, objects)
    final_image = (final_image * 0.5 + 0.5) * 255

    # Send the new array
    flipped_image = Image.fromarray(final_image.astype('uint8'))
    in_memory_file = io.BytesIO()
    flipped_image.save(in_memory_file, 'PNG')
    in_memory_file.seek(0)
    return send_file(in_memory_file, mimetype='image/png')


if __name__ == '__main__':
    yolo = prepare_detector()
    p2p = Pix2Pix(mode='try', checkpoint_dir='pix2pix/checkpoint/')
    objects = [0, 24, 26]
    # app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='127.0.0.1', port=5000, debug=True)
