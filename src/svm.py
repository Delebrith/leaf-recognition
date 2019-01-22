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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def main():
    files_dict = get_images_in_classes(FLAGS.data_dir)
    data = get_train_data(files_in_classes=files_dict, data_dir=FLAGS.data_dir)
    X, Y = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        shuffle=True, random_state=FLAGS.random_seed)

    svm = SVC(kernel=FLAGS.kernel, degree=2, C=FLAGS.penalty, class_weight='balanced', random_state=FLAGS.random_seed, verbose=2)
    cross_val_scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=12, verbose=1, n_jobs=4, scoring="accuracy")

    print('Cross-Validation scores: ', cross_val_scores)
    print("Accuracy: {} (+/- {})".format(cross_val_scores.mean(), cross_val_scores.std() * 2))

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


    title = "Learning Curves (SVM, {} kernel, c={})".format(FLAGS.kernel, FLAGS.penalty)
    plot_learning_curve(svm, title, X_train, y_train, (0.0, 1.01), cv=12, n_jobs=4)

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
