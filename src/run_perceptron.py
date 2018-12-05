import tensorflow as tf
from src.Perceptron import Perceptron
import pandas as pd

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('mode', 'eval', 'String: Training (training and validation set), '
                                    'testing (test set) or usage(single picture) mode')

flags.DEFINE_string('training_data_set', './../training-data.csv', 'String: Your training dataset')

flags.DEFINE_string('validation_data_set', './../validation-data.csv', 'String: Your validation dataset')

flags.DEFINE_string('test_data_set', './../test-data.csv', 'String: Your test dataset')

flags.DEFINE_string('data_dir', './../data', 'String: Directory with your images')

flags.DEFINE_string('model', './../model/perceptron', 'String: Directory where your model data is located '
                                                      '(or where to create it)')

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

    perceptron = Perceptron(128, 128, FLAGS.first_layer, FLAGS.second_layer, FLAGS.classes)

    training_df = pd.read_csv(FLAGS.training_data_set)
    validation_df = pd.read_csv(FLAGS.validation_data_set)

    perceptron.train(training_frame=training_df, validation_frame=validation_df, batch_size=32, epochs=50,
                     data_dir=FLAGS.data_dir)


if __name__ == "__main__":
    main()

