import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

from src.file_utils import get_images_in_classes

import os

# ===============DEFINE ARGUMENTS==============
flags = tf.app.flags

flags.DEFINE_string('data_dir', './../data/sets', 'String: Directory with your images')

flags.DEFINE_string('kernel', 'poly', 'String: Type of kernel')

flags.DEFINE_float('penalty', 1.0, 'Float: penalty coefficent (c)')

flags.DEFINE_integer('random_seed', 1, 'Int: Random seed to use for repeatability.')


FLAGS = flags.FLAGS


def main():
    files_dict = get_images_in_classes(FLAGS.data_dir)
    data = get_train_data(files_in_classes=files_dict, data_dir=FLAGS.data_dir)
    X, Y = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        shuffle=True, random_state=FLAGS.random_seed)

    svm = SVC(kernel=FLAGS.kernel, C=FLAGS.penalty, class_weight='balanced', random_state=FLAGS.random_seed, verbose=2)
    cross_val_scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=4, verbose=2, n_jobs=4)

    print('Cross-Validation scores: ', cross_val_scores)
    print("Accuracy: {} (+/- {})".format(cross_val_scores.mean(), cross_val_scores.std() * 2))


    return 0


def get_train_data(files_in_classes, data_dir):
    data = list()
    classes = list(files_in_classes.keys())
    for c in files_in_classes:
        for file in os.listdir(os.path.join(data_dir, c)):
            example = np.genfromtxt(os.path.join(data_dir, c, file), delimiter=',')
            data.append((example, classes.index(c)))

    return data


if __name__ == '__main__':
    main()
