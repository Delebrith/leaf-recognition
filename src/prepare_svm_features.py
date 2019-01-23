import tensorflow as tf
import cv2
import numpy as np

from src.LeNet5 import LeNet5
from src.file_utils import get_images_in_classes

import os

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_dir', './../data/sets', 'String: Directory with your images')

flags.DEFINE_integer('classes', 32, 'Int: Number of classes')

flags.DEFINE_integer('filters', 64, 'Int: Number of filters in first block')

flags.DEFINE_integer('filter_size', 3, 'Int: Size of filter')

flags.DEFINE_integer('batch_size', 32, 'Int: Number of images in batch')

flags.DEFINE_integer('input_width', 128, 'Int: Number of neurons in 2nd hidden layer')

flags.DEFINE_integer('input_height', 256, 'Int: Number of neurons in 3nd hidden layer')

flags.DEFINE_string('regularization', 'None', 'String: method of regularization')

flags.DEFINE_string('augmentation', 'No', 'String: is data augmentation on')

flags.DEFINE_string('type', 'lenet', 'String: type of neural network')

flags.DEFINE_float('lr', 0.001, 'Float: learning rate')

flags.DEFINE_string('model', '../../data/model.hdf5', 'Trained model for testing')

FLAGS = flags.FLAGS


def main():

    cnn = LeNet5(FLAGS.input_width, FLAGS.input_height, FLAGS.classes, FLAGS.filter_size, FLAGS.filters,
                  None if FLAGS.regularization == 'None' else FLAGS.regularization, 0.001)
    cnn.load(FLAGS.model)
    cnn.pop_last_layers()

    images_in_classes = get_images_in_classes(FLAGS.data_dir)
    prepare_data(images_in_classes=images_in_classes, cnn=cnn, data_dir=FLAGS.data_dir)

    return 0


def prepare_data(images_in_classes, cnn, data_dir):
    data_dest = os.path.join(os.path.dirname(data_dir), 'svm')
    os.makedirs(data_dest)
    for c in images_in_classes.keys():
        os.makedirs(os.path.join(data_dest, c))
        for file in os.listdir(os.path.join(data_dir, c)):
            print("Processing {}/{}".format(c, file))
            image = cv2.imread(filename=(os.path.join(data_dir, c, file)))
            features = cnn.predict(image)
            result = np.asarray(features)
            file = os.path.splitext(file)[0] + '.csv'
            np.savetxt(os.path.join(data_dest, c, os.path.basename(file)), result, delimiter=',')


if __name__ == '__main__':
    main()
