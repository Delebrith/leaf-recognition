import random
import tensorflow as tf

#===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_src', './../data', 'String: Your dataset directory')

flags.DEFINE_float('validation_set_size', 0.2, 'Float: The proportion of examples in the dataset to be used for validation')

flags.DEFINE_float('test_set_size', 0.2, 'Float: The proportion of examples in the dataset to be used for testing')

flags.DEFINE_integer('shards', 1, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

flags.DEFINE_string('tfrecord_file', 'data.tfrecord', 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS


def main():

    if not FLAGS.tfrecord_file:
        raise ValueError('tfrecord_file not given')
    if not FLAGS.dataset_dir:
        raise ValueError('data_src not given')

    image_files, classes = _get_files_in_classes(FLAGS.data_src)
    class_ids = dict(zip(classes, range(len(classes))))

    validation_index = int(FLAGS.validation_set_size * len(image_files))
    test_index = validation_index + int(FLAGS.test_set_size * len(image_files))

    random.seed(FLAGS.random_seed)
    random.shuffle(image_files)
    training_files = image_files[validation_index:]
    validation_index = image_files[:validation_index, test_index:]
    test_index = image_files[:test_index]


def _get_files_in_classes(data_src):
    print("Not implemented yet")
    return [0], [0]


if __name__ == "__main__":
    main()
