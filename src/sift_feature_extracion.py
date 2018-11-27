import os

import cv2
import tensorflow as tf
import numpy as np
from src.file_utils import get_files_in_classes

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data_src', 'String: Your dataset directory')

flags.DEFINE_string('data_dest', './../data_filtered', 'String: Directory where results of filtration will be stored')

flags.DEFINE_string('features_only', 'no', 'String: "yes" or "no", if filtering result should overlap original image'
                                           'or be drawn on black bitmap')

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)

        blank_image = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        if FLAGS.features_only == 'yes':
            features = cv2.drawKeypoints(blank_image, kp, blank_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            features = cv2.drawKeypoints(image, kp, blank_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        classname = os.path.split(os.path.split(image_path)[0])[1]
        features_path = os.path.join(os.path.join(inner_dir, classname), os.path.basename(image_path))
        cv2.imwrite(features_path, features)


if __name__ == "__main__":
    main()
