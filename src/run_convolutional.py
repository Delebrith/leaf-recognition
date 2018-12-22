import tensorflow as tf
from src.VGG16 import VGG16
from src.LeNet5 import LeNet5
import pandas as pd
from keras import backend

import os


# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('mode', 'eval', 'String: Training (training and validation set), '
                                    'testing (test set) or usage(single picture) mode')

flags.DEFINE_string('data_dir', './../data', 'String: Directory with your images')

flags.DEFINE_integer('epochs', 1, 'Int: Number of epochs')

flags.DEFINE_integer('classes', 32, 'Int: Number of classes')

flags.DEFINE_integer('filters', 64, 'Int: Number of filters in first block')

flags.DEFINE_integer('filter_size', 3, 'Int: Size of filter')

flags.DEFINE_integer('batch_size', 32, 'Int: Number of images in batch')

flags.DEFINE_integer('input_width', 128, 'Int: Number of neurons in 2nd hidden layer')

flags.DEFINE_integer('input_height', 256, 'Int: Number of neurons in 3nd hidden layer')

flags.DEFINE_string('regularization', 'None', 'String: method of regularization')

flags.DEFINE_string('augmentation', 'No', 'String: is data augmentation on')

flags.DEFINE_string('type', 'lenet', 'String: typr of neural network')

flags.DEFINE_float('lr', 0.001, 'Float: learning rate')

FLAGS = flags.FLAGS


def main():
    print(backend.tensorflow_backend._get_available_gpus())
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
    sess = tf.Session(config=config)
    backend.set_session(sess)

    if FLAGS.type == 'lenet':
        conv = LeNet5(FLAGS.input_width, FLAGS.input_height, FLAGS.classes, FLAGS.filter_size, FLAGS.filters,
                      None if FLAGS.regularization == 'None' else FLAGS.regularization, FLAGS.lr)
    elif FLAGS.type == 'vgg16':
        conv = VGG16(FLAGS.input_width, FLAGS.input_height, FLAGS.classes, FLAGS.filter_size, FLAGS.filters,
                      None if FLAGS.regularization == 'None' else FLAGS.regularization, FLAGS.lr)
    else:
        raise ValueError('Invalid network type!')

    if FLAGS.mode == 'train':
        training_frame = pd.read_csv(os.path.join(FLAGS.data_dir, "training-leafs.csv"))
        validation_frame = pd.read_csv(os.path.join(FLAGS.data_dir, "validation-leafs.csv"))

        augmentation = True if FLAGS.augmentation == 'Yes' else False
        conv.train(training_frame=training_frame, validation_frame=validation_frame, batch_size=FLAGS.batch_size,
                   epochs=FLAGS.epochs, data_dir=FLAGS.data_dir, augmentation=augmentation)
        conv.save(os.path.join(FLAGS.data_dir, "%s-model-%d-%d-%d-%d-%d-%s-%9.8f.hdf5") %
                  (FLAGS.type, FLAGS.input_width, FLAGS.input_height, FLAGS.filter_size, FLAGS.filters,
                   FLAGS.batch_size, FLAGS.regularization, FLAGS.lr),
                  os.path.join(FLAGS.data_dir, "%s-history-%d-%d-%d-%d-%d-%s-%9.8f.csv") %
                  (FLAGS.type, FLAGS.input_width, FLAGS.input_height, FLAGS.filter_size, FLAGS.filters,
                   FLAGS.batch_size, FLAGS.regularization, FLAGS.regularization, FLAGS.lr))

    test_frame = pd.read_csv(os.path.join(FLAGS.data_dir, "test-leafs.csv"))
    conv.evaluate(test_frame=test_frame, data_dir=FLAGS.data_dir, batch_size=FLAGS.batch_size)
    conv.top5(test_frame, FLAGS.data_dir, FLAGS.batch_size)


if __name__ == "__main__":
    main()
