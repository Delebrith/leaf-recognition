import os

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

from src.file_utils import get_files_and_classes

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data_src', 'String: Your dataset directory')

flags.DEFINE_string('data_dest', './../data_filtered', 'String: Directory where results of filtration will be stored')

flags.DEFINE_string('colors', 'gray', 'String: Filter image in rgb->grayscale or rgb->hsv')

flags.DEFINE_string('rich_keypoints', 'no', 'String: "yes" or "no", if filtering result should be flagged as '
                                            'DRAW_RICH_KEYPOINTS')

flags.DEFINE_string('output', 'jpg', 'String: "jpg" or "csv')

FLAGS = flags.FLAGS


def main():
    if not FLAGS.data_dest:
        raise ValueError('data_dest not given')
    if not FLAGS.data_src:
        raise ValueError('data_src not given')

    if not os.path.exists(FLAGS.data_src):
        raise ValueError('data_src directory does not exist')
    if not os.path.isdir(FLAGS.data_src):
        raise ValueError('data_src is not a directory')

    if not os.path.exists(FLAGS.data_dest):
        os.makedirs(FLAGS.data_dest)

    if FLAGS.colors not in ['gray', 'hsv', 'rgb'] or FLAGS.rich_keypoints not in ['yes', 'no'] \
            or FLAGS.output not in ['jpg', 'csv']:
        raise ValueError('invalid arguments in --colors, --output or --rich_keypoints')

    images, classes = get_files_and_classes(FLAGS.data_src)
    sift_extraction(images, classes, FLAGS.data_dest)


def sift_extraction(images, classes, dest_dir):
    inner_dir = os.listdir(FLAGS.data_src)[0]
    inner_dir = os.path.join(dest_dir, inner_dir)

    for classname in classes:
        os.makedirs(os.path.join(inner_dir, classname))

    for i in range(len(images)):
        FEATUTRES = 1000
        print('Extracting features from image %d/%d' % (i+1, len(images)))

        image_path = images[i]
        image = cv2.imread(image_path)

        if FLAGS.colors == 'gray':
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif FLAGS.colors == 'rgb':
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=FEATUTRES)
        kp = sift.detect(processed, None)


        if FLAGS.output == 'csv':
            classname = os.path.split(os.path.split(image_path)[0])[1]
            filename = os.path.basename(image_path)
            filename = os.path.splitext(filename)[0]
            filename = filename + '.csv'
            output_filename = os.path.join(inner_dir,  os.path.join(classname, filename))
            df = pd.DataFrame([], columns=['angle', 'octave', 'x', 'y', 'response', 'size'])

            for key_point in kp:
                df = df.append({'angle': key_point.angle,
                                'octave': key_point.octave,
                                'x': key_point.pt[0],
                                'y': key_point.pt[1],
                                'response': key_point.response,
                                'size': key_point.size},
                               ignore_index=True)

            df = df.sort_values(by='response', ascending=False)

            if df['angle'].size > FEATUTRES:
                df = df[:FEATUTRES]

            if df['angle'].size < FEATUTRES:
                for _ in range(FEATUTRES - df['angle'].size):
                    df = df.append({'angle': 0.0,
                                    'octave': 0.0,
                                    'x': 0.0,
                                    'y': 0.0,
                                    'response': 0.0,
                                    'size': 0.0},
                                   ignore_index=True)

            df.to_csv(output_filename)

        else:
            blank_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

            if FLAGS.rich_keypoints == 'yes':
                features = cv2.drawKeypoints(blank_image, kp, blank_image,
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            else:
                features = cv2.drawKeypoints(blank_image, kp, blank_image, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

            classname = os.path.split(os.path.split(image_path)[0])[1]
            features_path = os.path.join(os.path.join(inner_dir, classname), os.path.basename(image_path))
            cv2.imwrite(features_path, features)


if __name__ == "__main__":
    main()
