import random
import os
import math

import tensorflow as tf
import cv2
import numpy as np

#===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data', 'String: Your dataset directory')

flags.DEFINE_float('validation_set_size', 0.2,
                   'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_float('test_set_size', 0.2, 'Float: The proportion of examples in the dataset to be used for testing')

flags.DEFINE_integer('shards', 1, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

flags.DEFINE_string('tfrecord_file', 'data.tfrecord', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.tfrecord_file:
        raise ValueError('tfrecord_file not given')
    if not FLAGS.data_src:
        raise ValueError('data_src not given')

    image_files, classes = _get_files_in_classes(FLAGS.data_src)
    class_ids_dict = dict(zip(classes, range(len(classes))))

    validation_index = int(FLAGS.validation_set_size * len(image_files))
    test_index = validation_index + int(FLAGS.test_set_size * len(image_files))

    random.seed(FLAGS.random_seed)
    random.shuffle(image_files)
    training_files = image_files[validation_index:]
    validation_files = image_files[validation_index:test_index]
    test_files = image_files[:test_index]

    _prepare_tfrecord('training', training_files, class_ids_dict, FLAGS.shards, FLAGS.data_src, FLAGS.tfrecord_file)
    _prepare_tfrecord('test', test_files, class_ids_dict, FLAGS.shards, FLAGS.data_src, FLAGS.tfrecord_file)
    _prepare_tfrecord('validation', validation_files, class_ids_dict, FLAGS.shards, FLAGS.data_src, FLAGS.tfrecord_file)


def _get_files_in_classes(data_src):
    directories = []
    classes = []
    for element in os.listdir(data_src):
        path = os.path.join(data_src, element)
        if os.path.isdir(path):
            directories.append(path)
            classes.append(element)

    images = []
    for directory in directories:
        for element in os.listdir(directory):
            path = os.path.join(directory, element)
            if os.path.isfile(path):
                images.append(path)

    print(classes)
    print(images)

    return images, classes


def _prepare_tfrecord(type, images, class_dict, shards, data_src, tfrecord_filename):
    assert type in ['training', 'validation', 'test']
    images_per_shard = int(math.ceil(len(images)/float(shards)))

    filename = os.path.join(data_src, type + '.' + tfrecord_filename)

    writer = tf.python_io.TFRecordWriter(filename)
    


    return 0


def _read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    main()
