import random
import os
import datetime
import sys

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from src.file_utils import get_files_in_classes

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data', 'String: Your dataset directory')

flags.DEFINE_float('validation_set_size', 0.2,
                   'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_float('test_set_size', 0.2, 'Float: The proportion of examples in the dataset to be used for testing')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

flags.DEFINE_string('tfrecord_file', 'data', 'String: The base of output filename to name your TFRecord file. '
                                             'You\'ll find your datasets in data_src directory.')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.tfrecord_file:
        raise ValueError('tfrecord filename not given')
    if not FLAGS.data_src:
        raise ValueError('data_src not given')

    if not os.path.exists(FLAGS.data_src):
        raise ValueError('data_src directory does not exist')
    if not os.path.isdir(FLAGS.data_src):
        raise ValueError('data_src is not a directory')

    image_files, classes = get_files_in_classes(FLAGS.data_src)
    class_ids_dict = dict(zip(classes, range(len(classes))))
    print('Total images: %d' % len(image_files))

    validation_index = int(FLAGS.validation_set_size * len(image_files))
    test_index = int(FLAGS.test_set_size * len(image_files)) + validation_index

    random.seed(FLAGS.random_seed)
    random.shuffle(image_files)
    validation_files = image_files[0:validation_index]
    test_files = image_files[validation_index:test_index]
    training_files = image_files[test_index:]

    timestamp = datetime.datetime.now().isoformat()
    # _prepare_tfrecord('training', training_files, class_ids_dict, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)
    # _prepare_tfrecord('test', test_files, class_ids_dict, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)
    # _prepare_tfrecord('validation', validation_files, class_ids_dict, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)
    _prepare_dataframe('training', training_files, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)
    _prepare_dataframe('test', test_files, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)
    _prepare_dataframe('validation', validation_files, FLAGS.data_src, FLAGS.tfrecord_file, timestamp)

def _read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _prepare_tfrecord(type, files, class_dict, data_src, tfrecord_filename, timestamp):
    assert type in ['training', 'validation', 'test']

    filename = os.path.join(data_src, '%s-%s.tfrecord' % (
        type, tfrecord_filename
    ))
    writer = tf.python_io.TFRecordWriter(filename)

    for i in range(0, len(files)):
        print('Including image %d/%d into %s dataset' % (i+1, len(files), type))
        sys.stdout.flush()

        image = _read_image(files[i])
        classname = os.path.basename(os.path.dirname(files[i]))
        feature = {'class_id': _int64_feature(class_dict[classname]),
                   'data': _bytes_feature(tf.compat.as_bytes(np.array2string(image)))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


def _prepare_dataframe(type, files, data_src, tfrecord_filename, timestamp):
    assert type in ['training', 'validation', 'test']

    output_filename = os.path.join(data_src, '%s-%s.csv' % (
        type, tfrecord_filename
    ))
    df = pd.DataFrame([], columns=['data', 'class'])

    os.makedirs(os.path.join(data_src, type))

    for i in range(0, len(files)):
        print('Including image %d/%d into %s dataset' % (i+1, len(files), type))
        sys.stdout.flush()

        classname = os.path.basename(os.path.dirname(files[i]))

        filename = os.path.basename(files[i])
        filename = os.path.splitext(filename)[0]
        filename = filename + '.jpg'

        image = cv2.imread(files[i])
        cv2.imwrite(filename=os.path.join(data_src, os.path.join(type, filename)), img=image)

        df = df.append({'data': filename,
                        'class': classname}, ignore_index=True)

    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    main()