import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve
from sklearn.svm import SVC
from matplotlib import pyplot as plt

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

    svm = SVC(kernel=FLAGS.kernel, degree=2, C=FLAGS.penalty, class_weight='balanced', random_state=FLAGS.random_seed, verbose=2)
    cross_val_scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=4, verbose=1, n_jobs=4, scoring="accuracy")

    param_range = np.logspace(-6, -1, 5)
    train_scores, valid_scores = validation_curve(svm, X_train, y_train, "gamma", cv=12,
                                                  param_range=param_range,
                                                  scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)

    plt.title("Validation Curve with SVM kernel: {}, c: {}".format(FLAGS.kernel, FLAGS.penalty))
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

    fit_result = svm.fit(X_train, y_train)
    test_result = svm.score(X_test, y_test)

    print("[RESULTS]: fit: {} test: {} ".format(fit_result, test_result))
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
