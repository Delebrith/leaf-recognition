from src.DfPerceptron import DfPerceptron
from src.Perceptron import Perceptron
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc


def main():
    plt.figure(1)

    perceptron = DfPerceptron(1024, 512, 128, 32)
    perceptron.load("../../rgb-to-csv2/perceptron-model-1024-512-128-input-6-1000.hdf5")
    test_df = pd.read_csv("../../rgb-to-csv2/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb-to-csv2")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(1024 x 512 x 128) (6 x 1000) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='deeppink', linewidth=2)

    perceptron = DfPerceptron(512, 256, 128, 32)
    perceptron.load("../../rgb-to-csv2/perceptron-model-512-256-128-input-6-1000.hdf5")
    test_df = pd.read_csv("../../rgb-to-csv2/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb-to-csv2")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(512 x 256 x 128) (6 x 1000) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='red', linewidth=2)

    perceptron = DfPerceptron(256, 128, 64, 32)
    perceptron.load("../../rgb-to-csv2/perceptron-model-256-128-64-input-6-1000.hdf5")
    test_df = pd.read_csv("../../rgb-to-csv2/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb-to-csv2")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(256 x 128 x 64) (6 x 1000) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='pink', linewidth=2)

    perceptron = DfPerceptron(128, 96, 64, 32)
    perceptron.load("../../rgb-to-csv2/perceptron-model-128-96-64-input-6-1000.hdf5")
    test_df = pd.read_csv("../../rgb-to-csv2/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb-to-csv2")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(128 x 96 x 64) (6 x 1000) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='purple', linewidth=2)

    perceptron = Perceptron(128, 256, 1024, 512, 128, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-1024-512-128-input-128-256.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(1024 x 512 x 128) (128 x 256) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='blue', linewidth=2)

    perceptron = Perceptron(128, 256, 512, 256, 128, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-512-256-128-input-128-256.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(512 x 256 x 128) (128 x 256) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='lightblue', linewidth=2)

    perceptron = Perceptron(128, 256, 256, 128, 64, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-256-128-64-input-128-256.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr,
             label='(256 x 128 x 64) (128 x 256) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='aquamarine', linewidth=2)

    perceptron = Perceptron(64, 128, 1024, 512, 128, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-1024-512-128-input-64-128.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(1024 x 512 x 128) (64 x 128) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='lime', linewidth=2)

    perceptron = Perceptron(64, 128, 512, 256, 128, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-512-256-128-input-64-128.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(512 x 256 x 128) (64 x 128) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='springgreen', linewidth=2)

    perceptron = Perceptron(64, 128, 256, 128, 64, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-256-128-64-input-64-128.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(256 x 128 x 64) (64 x 128) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='olivedrab', linewidth=2)

    perceptron = Perceptron(64, 128, 128, 96, 64, 32)
    perceptron.load("../../rgb/augmented/perceptron-model-128-96-64-input-64-128.hdf5")
    test_df = pd.read_csv("../../rgb/test-leafs.csv")
    fpr, tpr = perceptron.draw_roc(test_df, "../../rgb")
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             label='(128 x 96 x 64) (64 x 128) ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc),
             color='green', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


if __name__ == "__main__":
    main()