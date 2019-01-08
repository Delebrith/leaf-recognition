import random
import os
import sys
import shutil

import tensorflow as tf
import cv2
import pandas as pd
from src.file_utils import get_files_and_classes, get_images_in_classes

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data', 'String: Your dataset directory')

flags.DEFINE_float('validation_set_size', 0.2,
                   'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_float('test_set_size', 0.2, 'Float: The proportion of examples in the dataset to be used for testing')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.data_src:
        raise ValueError('data_src not given')

    if not os.path.exists(FLAGS.data_src):
        raise ValueError('data_src directory does not exist')
    if not os.path.isdir(FLAGS.data_src):
        raise ValueError('data_src is not a directory')

    image_files, classes = get_files_and_classes(FLAGS.data_src)
    print('Total images: %d' % len(image_files))

    validation_index = int(FLAGS.validation_set_size * len(image_files))
    test_index = int(FLAGS.test_set_size * len(image_files)) + validation_index

    random.seed(FLAGS.random_seed)
    random.shuffle(image_files)
    validation_files = image_files[0:validation_index]
    test_files = image_files[validation_index:test_index]
    training_files = image_files[test_index:]

    data_dir = FLAGS.data_src
    os.makedirs(os.path.join(data_dir, "sets", "training"))
    os.makedirs(os.path.join(data_dir, "sets",  "validation"))
    os.makedirs(os.path.join(data_dir, "sets",  "test"))

    for c in classes:
        os.makedirs(os.path.join(data_dir, "sets",  "training", c))
        os.makedirs(os.path.join(data_dir, "sets",  "validation", c))
        os.makedirs(os.path.join(data_dir, "sets",  "test", c))

    images_in_classes = get_images_in_classes(os.path.join(data_dir, "Folio"))

    for c in classes:
        for file in training_files:
            if os.path.basename(file) in images_in_classes[c]:
                shutil.copyfile(file, os.path.join(data_dir, "sets", "training", c, os.path.basename(file)))

        for file in validation_files:
            if os.path.basename(file) in images_in_classes[c]:
                shutil.copyfile(file, os.path.join(data_dir, "sets", "validation", c, os.path.basename(file)))

        for file in test_files:
            if os.path.basename(file) in images_in_classes[c]:
                shutil.copyfile(file, os.path.join(data_dir, "sets", "test", c, os.path.basename(file)))


if __name__ == '__main__':
    main()