import tensorflow as tf
import numpy as np


class NeuralNetwork:
    def __init__(self, input_width, input_height, classes, restore=False):
        self.input_width = input_width
        self.input_height = input_height
        self.image_content = tf.placeholder(tf.string)
        self.expected_result = tf.placeholder(tf.float32, [None, classes])

        if restore:
            self.neural_network = self._restore()
        else:
            self.neural_network = self._build()

        self.loss_function = tf.reduce_sum(tf.squared_difference(self.neural_network, self.expected_result), 1)

        optimizer = tf.train.AdamOptimizer()
        self.training_operation = optimizer.minimize(self.loss_function)

    def _build(self):
        raise NotImplementedError

    def _restore(self):
        raise NotImplementedError

    def train(self, training_set, validation_set, epochs, batchsize):
        feature = {'data': tf.FixedLenFeature([], tf.string),
                   'class_id': tf.FixedLenFeature([], tf.int64)}

        training_dataset = tf.data.TFRecordDataset(training_set).shuffle(1000).repeat()\
            .batch(batchsize, drop_remainder=True)
        validation_set = tf.data.TFRecordDataset(validation_set).shuffle(1000).repeat()\
            .batch(batchsize, drop_remainder=True)

        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_set.make_initializable_iterator()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(training_iterator.initializer)
            example = sess.run(tf.parse_example(training_iterator.get_next(), feature))
            for e in range(epochs):
                r = sess.run(self.neural_network, feed_dict={self.image_content: example['data'],
                                                             self.expected_result:
                                                                 self.__parse_expected(example['class_id'])})
                print('epoch ' + str(e) + ' : result ' + str(r))

    def __parse_expected(self, expected):
        result = []
        for i in range(len(expected)):
            result.append([0.0] * self.classes)
            result[i][expected[i]] = 1.0
        return result
