import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("../../filtered_features_rich/perceptron-history-1024-128.csv")
    df.plot(y=['loss', 'val_loss'], use_index=True)
    plt.show()
    df.plot(y=['acc', 'val_acc'], use_index=True)
    plt.show()


if __name__ == "__main__":
    main()
