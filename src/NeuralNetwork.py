import tensorflow as tf


class NeuralNetwork:
    def __init__(self, input_width, input_height, classes, restore=False):
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

    def train(self, training_set, validation_set, epochs):
        feature = {'data': tf.FixedLenFeature([], tf.string),
                   'class_id': tf.FixedLenFeature([], tf.int64)}

        training_dataset = tf.data.TFRecordDataset(training_set).shuffle(1000).repeat().batch(64, drop_remainder=True)
        validation_set = tf.data.TFRecordDataset(validation_set).shuffle(1000).repeat().batch(64, drop_remainder=True)

        training_iterator = training_dataset.make_initializable_iterator()
        validation_iterator = validation_set.make_initializable_iterator()

        with tf.Session() as sess:
            # tbc
            raise NotImplementedError
