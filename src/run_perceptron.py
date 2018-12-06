import tensorflow as tf
from src.Perceptron import Perceptron
import pandas as pd

import os

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('mode', 'eval', 'String: Training (training and validation set), '
                                    'testing (test set) or usage(single picture) mode')

flags.DEFINE_string('training_data_set', './../training-data.csv', 'String: Your training dataset')

flags.DEFINE_string('validation_data_set', './../validation-data.csv', 'String: Your validation dataset')

flags.DEFINE_string('test_data_set', './../test-data.csv', 'String: Your test dataset')

flags.DEFINE_string('data_dir', './../data', 'String: Directory with your images')

flags.DEFINE_integer('epochs', 1, 'Int: Number of epochs')

flags.DEFINE_integer('classes', 1, 'Int: Number of classes')

flags.DEFINE_integer('first_layer', 1024, 'Int: Number of neurons in 1st hidden layer')

flags.DEFINE_integer('second_layer', 128, 'Int: Number of neurons in 2nd hidden layer')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.mode:
        raise ValueError("Please specify mode")
    if FLAGS.mode not in ['eval', 'train', 'test']:
        raise ValueError('Mode should be specified as one of [use, train, test]')

    perceptron = Perceptron(196, 196, FLAGS.first_layer, FLAGS.second_layer, FLAGS.classes)

    training_df = pd.read_csv(FLAGS.training_data_set)
    validation_df = pd.read_csv(FLAGS.validation_data_set)

    perceptron.train(training_frame=training_df, validation_frame=validation_df, batch_size=32, epochs=FLAGS.epochs,
                     data_dir=FLAGS.data_dir)

    perceptron.save(os.path.join(FLAGS.data_dir, 'perceptron-model-%s-%s.hdf5' %
                                 (FLAGS.first_layer, FLAGS.second_layer)),
                    os.path.join(FLAGS.data_dir,  'perceptron-history-%s-%s.pickle' %
                                 (FLAGS.first_layer, FLAGS.second_layer)))

if __name__ == "__main__":
    main()

