import os
from yolo.detect import prepare_detector
from pix2pix.utils.model import Pix2Pix
from utils import read_image, prepare_image_yolo, cut_result, create_new_image, prepare_frame_p2p, frame_to_int
import matplotlib.pyplot as plt
import argparse
import cv2
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image-path',
                        default='input/remove_people/',
                        type=str,
                        help='The path to the directory where images are saved')

    parser.add_argument('-v', '--video-path',type=str)

    parser.add_argument('-vo', '--video-output-path',
                        type=str,
                        default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-io', '--image-output-path',
                        type=str,
                        default='./output/',
                        help='The path of the output photos')

    parser.add_argument('-ob', '--objects', nargs='+', type=int, default=[0, 24, 26])

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.image_path == FLAGS.image_output_path or FLAGS.video_path == FLAGS.video_output_path:
        raise Exception('Input and output directories cannot be the same')

    yolo = prepare_detector()
    p2p = Pix2Pix(mode='try', checkpoint_dir='pix2pix/checkpoint/')

    if FLAGS.image_path:
        input_folder = FLAGS.image_path
        for file in os.listdir(input_folder):
            input = input_folder + file
            image = read_image(input)

            image_yolo = prepare_image_yolo(image)
            output_yolo = yolo(image_yolo)
            output_yolo = cut_result(output_yolo)
            final_image = create_new_image(image, output_yolo, p2p, FLAGS.objects)
            if os.path.isfile(FLAGS.image_output_path + '/' + file):
                os.remove(FLAGS.image_output_path + '/' + file)
            plt.imsave(f'{FLAGS.image_output_path}/{file}', final_image * 0.5 + 0.5)

    elif FLAGS.video_path:
        print(FLAGS.video_path)
        height, width = None, None
        writer = None
        try:
            vid = cv2.VideoCapture(FLAGS.video_path)
        except:
            raise Exception('Video cannot be loaded!\n\
                                       Please check the path provided!')
        finally:
            while True:
                start = time.time()
                grabbed, frame = vid.read()

                # Checking if the complete video is read
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                frame_yolo = prepare_image_yolo(frame)
                frame_p2p = prepare_frame_p2p(frame)
                output_yolo = yolo(frame_yolo)
                output_yolo = cut_result(output_yolo)
                final_frame = create_new_image(frame_p2p, output_yolo, p2p, FLAGS.objects)

                if writer is None:
                    # Initialize the video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(FLAGS.video_output_path, fourcc, 30,
                                             (frame.shape[1], frame.shape[0]), True)

                writer.write(frame_to_int(final_frame))
                print(f'This frame took {time.time() - start} seconds')

            print("[INFO] Cleaning up...")
            writer.release()
            vid.release()

    else:
        print('Neither video nor image loaded.')