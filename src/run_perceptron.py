import tensorflow as tf
from src.perceptron import Perceptron

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('mode', 'eval', 'String: Training (training and validation set), testing (test set) or usage(single picture) mode')

flags.DEFINE_string('training_data_set', './../training-data.tfrecord', 'String: Your training dataset')

flags.DEFINE_string('validation_data_set', './../validation-data.tfrecord', 'String: Your validation dataset')

flags.DEFINE_string('test_data_set', './../test-data.tfrecord', 'String: Your test dataset')

flags.DEFINE_string('model', './../model/perceptron', 'String: Directory where your model data is located (or where to create it)')

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

    perceptron.train(FLAGS.training_data_set, FLAGS.validation_data_set, epochs=FLAGS.epochs, batchsize=32)


if __name__ == "__main__":
    main()

