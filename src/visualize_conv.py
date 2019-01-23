from src.DfPerceptron import DfPerceptron
from src.LeNet5 import LeNet5
from matplotlib import pyplot as plt
from sklearn.metrics import auc

import pandas as pd


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

def main():
    plt.figure(1)

    plot_cnn(128, 256, 3, 20, 32, 'None', '#0000FF')
    plot_cnn(64, 128, 3, 20, 32, 'None', '#00FFFF')
    plot_cnn(32, 64, 3, 20, 32, 'None', '#9999FF')
    plot_cnn(32, 64, 5, 20, 32, 'None', '#99FF99')
    plot_cnn(32, 64, 7, 20, 32, 'None', '#00FF00')
    plot_cnn(32, 64, 3, 16, 32, 'None', '#FF99FF')
    plot_cnn(32, 64, 3, 24, 32, 'None', '#FF00FF')
    plot_cnn(32, 64, 3, 20, 16, 'None', '#FF0000')
    plot_cnn(32, 64, 3, 20, 24, 'None', '#990000')
    plot_cnn(32, 64, 3, 20, 40, 'None', '#FF9999')
    plot_cnn(32, 64, 3, 20, 48, 'None', '#FF9900')
    plot_cnn(32, 64, 3, 20, 32, 'l1', '#FFFF00')
    plot_cnn(32, 64, 3, 20, 32, 'l2', '#449900')

    lenet = LeNet5(128, 256, 32, 3, 20, None, 0.001)
    lenet.load("../../data/sets/lenet-model-128-256-3-20-32-None-0.001-adam-No.hdf5")
    fpr, tpr = lenet.draw_roc("../../data/sets")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='LeNet5 model ({}x{}) filter size: {} filters: {} batch size: {} regularization: {} No augmentation '
                   'ROC curve (area = {})'
             .format(128, 256, 3, 20, 32, None,
                     roc_auc),
             color='#666666', linewidth=1)

    lenet = LeNet5(128, 256, 32, 3, 20, None, 0.001)
    fpr, tpr = lenet.draw_roc("../../data/sets")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='No training LeNet5 model ({}x{}) filter size: {} filters: {} batch size: {} regularization: {} '
                   'ROC curve (area = {})'
             .format(128, 256, 3, 20, 32, None,
                     roc_auc),
             color='#666666', linewidth=1)

    perceptron = DfPerceptron(512, 256, 128, 32)
    perceptron.load("../../rgb-to-csv2/perceptron-model-512-256-128-input-6-1000.hdf5")
    test_df = pd.read_csv("../../rgb-to-csv2/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb-to-csv2")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='perceptron (256 x 128 x 64) (6 x 1000) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='pink', linewidth=1)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


if __name__ == "__main__":
    main()