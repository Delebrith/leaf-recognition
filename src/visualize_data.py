import pandas as pd
import matplotlib.pyplot as plt

import sys


def main():
    df = pd.read_csv(sys.argv[1])
    df.plot(y=['loss', 'val_loss'], use_index=True)
    plt.xlabel('epochs')
    plt.ylabel('loss function')
    plt.show()
    df.plot(y=['acc', 'val_acc'], use_index=True)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    main()
