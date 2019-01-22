from src.LeNet5 import LeNet5
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from src.file_utils import get_images_in_classes
from src.svm import get_train_data, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import numpy as np


def plot_cnn(width, height, filter_size, filters, batch_size, regularization, color):
    lenet = LeNet5(width, height, 32, filter_size, filters, None if regularization == 'None' else regularization, 0.001)
    lenet.load("../../data/sets/lenet-model-{}-{}-{}-{}-{}-{}-0.001-adam.hdf5"
               .format(width, height, filter_size, filters, batch_size, regularization))
    fpr, tpr = lenet.draw_roc("../../data/sets")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='LeNet5 model ({}x{}) filter size: {} filters: {} batch size: {} regularization: {} '
                   'ROC curve (area = {})'
             .format(width, height, filter_size, filters, batch_size, regularization,
                     roc_auc),
             color=color, linewidth=1)


def plot_svm(kernel, C, color):
    files_dict = get_images_in_classes("../../data/svm/")
    data = get_train_data(files_in_classes=files_dict, data_dir="../../data/svm/")
    X, Y = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        shuffle=True, random_state=1)

    svm = SVC(kernel=kernel, degree=2, C=C, class_weight='balanced', random_state=1,
              verbose=2)
    y_score = svm.fit(X_train, y_train).decision_function(X_test).ravel()

    def to_catogrical(vector):
        result = []
        for _ in range(len(vector)):
            result.append([0] * 32)

        for elem in range(len(result)):
            result[elem][vector[elem]] = 1

        return result
    y_test = np.asarray(to_catogrical(y_test)).ravel()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    lw = 2
    plt.plot(fpr, tpr, color=color,
             lw=lw, label='ROC curve SVM kernel: {} C: {} (area = {})'.format(kernel, C, roc_auc))


def main():
    plt.figure(1)

    plot_cnn(32, 64, 3, 20, 32, 'None', '#9999FF')
    plot_svm('poly', 0.1, '#FF0000')
    plot_svm('poly', 1, '#FF9999')
    plot_svm('poly', 10, '#FF00FF')
    plot_svm('rbf', 0.1, '#0000FF')
    plot_svm('rbf', 1, '#9999FF')
    plot_svm('rbf', 10, '#00FFFF')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


if __name__ == "__main__":
    main()