import os

import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from src.file_utils import get_files_in_classes

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data_src', 'String: Your dataset directory')

flags.DEFINE_string('data_dest', './../data_filtered', 'String: Directory where results of filtration will be stored')

flags.DEFINE_string('colors', 'gray', 'String: Filter image in rgb->grayscale or rgb->hsv')

flags.DEFINE_string('rich_keypoints', 'no', 'String: "yes" or "no", if filtering result should be flagged as '
                                            'DRAW_RICH_KEYPOINTS')

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

    if FLAGS.colors not in ['gray', 'hsv', 'rgb'] or FLAGS.rich_keypoints not in ['yes', 'no']:
        raise ValueError('invalid arguments in --colors or --rich_keypoints')

    images, classes = get_files_in_classes(FLAGS.data_src)
    sift_extraction(images, classes, FLAGS.data_dest)


def sift_extraction(images, classes, dest_dir):
    inner_dir = os.listdir(FLAGS.data_src)[0]
    inner_dir = os.path.join(dest_dir, inner_dir)

    for classname in classes:
        os.makedirs(os.path.join(inner_dir, classname))

    for i in range(len(images)):
        print('Extracting features from image %d/%d' % (i+1, len(images)))

        image_path = images[i]
        image = cv2.imread(image_path)

        if FLAGS.colors == 'gray':
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif FLAGS.colors == 'rgb':
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)

        sift = cv2.xfeatures2d.SIFT_create()
        kps = sift.detect(processed)
        decoded_kps = np.asarray([(kp.response, kp.pt[0], kp.pt[1], kp.angle, kp.size, kp.octave) for kp in kps])
        kps_as_df = pd.DataFrame(decoded_kps, columns=["response", "x_pos", "y_pos", "angle", "size", "octave"])

        blank_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        if FLAGS.rich_keypoints == 'yes':
            features = cv2.drawKeypoints(blank_image, kps, blank_image,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            features = cv2.drawKeypoints(blank_image, kps, blank_image, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        classname = os.path.split(os.path.split(image_path)[0])[1]
        features_path = os.path.join(os.path.join(inner_dir, classname), os.path.basename(image_path))
        cv2.imwrite(features_path, features)
        csv_path = features_path.split(".")[0] + ".csv"
        kps_as_df.to_csv(csv_path)


if __name__ == "__main__":
    main()
