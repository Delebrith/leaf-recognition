import tensorflow as tf
from src.Perceptron import Perceptron
from src.DfPerceptron import DfPerceptron
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

flags.DEFINE_integer('first_layer', 256, 'Int: Number of neurons in 1st hidden layer')

flags.DEFINE_integer('second_layer', 128, 'Int: Number of neurons in 2nd hidden layer')

flags.DEFINE_integer('third_layer', 64, 'Int: Number of neurons in 3nd hidden layer')

flags.DEFINE_integer('input_width', 64, 'Int: Number of neurons in 2nd hidden layer')

flags.DEFINE_integer('input_height', 128, 'Int: Number of neurons in 3nd hidden layer')

flags.DEFINE_string('type', 'image', 'String: "image" or "features"')


FLAGS = flags.FLAGS


def main():

    if not FLAGS.mode:
        raise ValueError("Please specify mode")
    if FLAGS.mode not in ['eval', 'train', 'test']:
        raise ValueError('Mode should be specified as one of [use, train, test]')
    if FLAGS.type not in ['image', 'features']:
        raise ValueError('type should be specified as one of [image, features]')

    if FLAGS.type == 'image':
        perceptron = Perceptron(FLAGS.input_width, FLAGS.input_height, FLAGS.first_layer, FLAGS.second_layer, 
                                FLAGS.third_layer, FLAGS.classes)
    
        training_df = pd.read_csv(FLAGS.training_data_set)
        validation_df = pd.read_csv(FLAGS.validation_data_set)
        test_df = pd.read_csv(FLAGS.test_data_set)
    
        perceptron.train(training_frame=training_df, validation_frame=validation_df, batch_size=32, epochs=FLAGS.epochs,
                         data_dir=FLAGS.data_dir)
    
        perceptron.evaluate(test_frame=test_df, data_dir=FLAGS.data_dir, batch_size=32)
    
        perceptron.save(os.path.join(FLAGS.data_dir, 'perceptron-model-%s-%s-%s-input-%d-%d.hdf5' %
                                     (FLAGS.first_layer, FLAGS.second_layer, FLAGS.third_layer, FLAGS.input_width,
                                      FLAGS.input_height)),
                        os.path.join(FLAGS.data_dir,  'perceptron-history-%s-%s-%s-input-%d-%d.csv' %
                                     (FLAGS.first_layer, FLAGS.second_layer, FLAGS.third_layer, FLAGS.input_width,
                                      FLAGS.input_height)))
        
    if FLAGS.type == 'features':
        perceptron = DfPerceptron(FLAGS.first_layer, FLAGS.second_layer, FLAGS.third_layer, FLAGS.classes)

        training_df = pd.read_csv(FLAGS.training_data_set)
        validation_df = pd.read_csv(FLAGS.validation_data_set)
        test_df = pd.read_csv(FLAGS.test_data_set)

        perceptron.train(training_frame=training_df, validation_frame=validation_df, batch_size=32, epochs=FLAGS.epochs,
                         data_dir=FLAGS.data_dir)

        perceptron.evaluate(test_frame=test_df, data_dir=FLAGS.data_dir, batch_size=32)

        perceptron.save(os.path.join(FLAGS.data_dir, 'perceptron-model-%s-%s-%s-input-%d-%d.hdf5' %
                                     (FLAGS.first_layer, FLAGS.second_layer, FLAGS.third_layer, FLAGS.input_width,
                                      FLAGS.input_height)),
                        os.path.join(FLAGS.data_dir, 'perceptron-history-%s-%s-%s-input-%d-%d.csv' %
                                     (FLAGS.first_layer, FLAGS.second_layer, FLAGS.third_layer, FLAGS.input_width,
                                      FLAGS.input_height)))

if __name__ == "__main__":
    main()

