import tensorflow as tf
import numpy as np
import cv2

from src.NeuralNetwork import NeuralNetwork


class Perceptron(NeuralNetwork):

    def __init__(self, input_width, input_height, first_hidden_size, second_hidden_size,
                 classes, restore=False):
        self.input_width = input_width
        self.input_height = input_height
        self.first_hidden_size = first_hidden_size
        self.second_hidden_size = second_hidden_size
        self.classes = classes
        NeuralNetwork.__init__(self, input_width, input_height, restore)

    def _build(self):
        initializer = tf.random_normal_initializer(0.0, 1.0)

        image = cv2.resize(tf.cast(tf.decode_raw(self.image_content, tf.uint8), tf.float32),
                           (self.input_width, self.input_height))

        image = np.array(image) / 255

        input_layer = tf.reshape(image, [-1, self.input_width, self.input_height, 3])

        first_hidden_layer = tf.layers.dense(input_layer, self.first_hidden_size,
                                             activation=tf.nn.sigmoid,
                                             kernel_initializer=initializer,
                                             bias_initializer=initializer)

        second_hidden_layer = tf.layers.dense(first_hidden_layer, self.second_hidden_size,
                                              activation=tf.nn.sigmoid,
                                              kernel_initializer=initializer,
                                              bias_initializer=initializer)

        result = tf.layers.dense(second_hidden_layer, self.classes,
                                 activation=tf.nn.softmax,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)

        return result

    def _restore(self):
        raise NotImplementedError
