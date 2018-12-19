import random
import os
import sys

import tensorflow as tf
import cv2
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

flags.DEFINE_string('output_filename', 'data', 'String: The base of output filename to name your TFRecord file. '
                                             'You\'ll find your datasets in data_src directory.')

flags.DEFINE_string('type', 'jpg', 'String: What files are splitted into datasets, jpg or csv')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.output_filename:
        raise ValueError('output_filename filename not given')
    if not FLAGS.data_src:
        raise ValueError('data_src not given')

    if not os.path.exists(FLAGS.data_src):
        raise ValueError('data_src directory does not exist')
    if not os.path.isdir(FLAGS.data_src):
        raise ValueError('data_src is not a directory')

    if FLAGS.type not in ['csv', 'jpg']:
        raise ValueError('type has incorrect value')

    image_files, classes = get_files_in_classes(FLAGS.data_src)
    print('Total images: %d' % len(image_files))

    validation_index = int(FLAGS.validation_set_size * len(image_files))
    test_index = int(FLAGS.test_set_size * len(image_files)) + validation_index

    random.seed(FLAGS.random_seed)
    random.shuffle(image_files)
    validation_files = image_files[0:validation_index]
    test_files = image_files[validation_index:test_index]
    training_files = image_files[test_index:]

    if FLAGS.type == 'jpg':
        _prepare_dataframe('training', training_files, FLAGS.data_src, FLAGS.output_filename)
        _prepare_dataframe('test', test_files, FLAGS.data_src, FLAGS.output_filename)
        _prepare_dataframe('validation', validation_files, FLAGS.data_src, FLAGS.output_filename)
    else:
        _prepare_csv_dataframe('training', training_files, FLAGS.data_src, FLAGS.output_filename)
        _prepare_csv_dataframe('test', test_files, FLAGS.data_src, FLAGS.output_filename)
        _prepare_csv_dataframe('validation', validation_files, FLAGS.data_src, FLAGS.output_filename)


def _prepare_dataframe(type, files, data_src, output_filename):
    assert type in ['training', 'validation', 'test']

    output_filename = os.path.join(data_src, '%s-%s.csv' % (
        type, output_filename
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


def _prepare_csv_dataframe(type, files, data_src, output_filename):
    assert type in ['training', 'validation', 'test']

    output_filename = os.path.join(data_src, '%s-%s.csv' % (
        type, output_filename
    ))
    df = pd.DataFrame([], columns=['file', 'class'])


    inner_dir = os.listdir(FLAGS.data_src)[0]
    inner_dir = os.path.join(data_src, inner_dir)
    classes = os.listdir(inner_dir)

    for i in range(0, len(files)):
        print('Including image %d/%d into %s dataset' % (i+1, len(files), type))
        sys.stdout.flush()

        classname = os.path.basename(os.path.dirname(files[i]))

        filename = os.path.join(classname, os.path.basename(files[i]))

        class_id = classes.index(classname)

        df = df.append({'file': filename,
                        'class': classname,
                        'class_id': class_id},
                       ignore_index=True)

    df.to_csv(output_filename, index=False)


if __name__ == "__main__":
    main()
