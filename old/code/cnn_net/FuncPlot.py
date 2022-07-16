"""
FuncPlot.py
函数来plot神经网络训练过程和验证集技巧
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt


def trainPlot(FName):
    File = open("TrainRes/" + FName + ".pickle", "rb")
    Dic = pickle.load(File)
    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(1, 24), Dic["ACCList"], "-o", label="CNN")
    ax1.hlines(0.5, 0.5, 23.5)
    ax1.set_xlim(0.5, 23.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("lead time (month)")
    ax1.set_ylabel("ACC")
    plt.legend()
    plt.xticks(np.arange(1, 24, 1))
    plt.title(Dic["train_time"], loc="right")
    plt.title("ExpName:" + Dic["trainName"], loc="left")
    ax2 = fig.add_subplot(212)
    ax2.plot(Dic["lossList"], label="Train Loss")
    ax2.hlines(Dic["LossVal"], 0, len(Dic["lossList"]), label="Val Loss", colors="red")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("batch_number")
    plt.legend()
    plt.savefig("./TrainRes/%s.png" % FName, dpi=300)
    plt.show()


if __name__ == '__main__':
    trainPlot("try")
