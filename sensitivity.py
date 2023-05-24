import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import smoothness
import random as rnd

if __name__ == '__main__':
    trainingGood = pd.read_csv("assets/140_1.csv")
    trainingBad = pd.read_csv("assets/140_2.csv")

    fakeGood = []
    fakeLowFreqHighAmp = []
    fakeHighFreqHighAmp = []
    fakeLowFreqLowAmp = []
    fakeHighFreqLowAmp = []
    fakeNoisy = []

    for i in range(84):
        fakeGood.append(math.sin(3.14159/85 * i))
        fakeLowFreqHighAmp.append(math.sin(3.14159/85 * i) + 0.5 * math.sin(3.14159/10 * i))
        fakeHighFreqHighAmp.append(math.sin(3.14159 / 85 * i) + 0.5 * math.sin(3.14159 / 5 * i))
        fakeLowFreqLowAmp.append(math.sin(3.14159 / 85 * i) + 0.15 * math.sin(3.14159 / 10 * i))
        fakeHighFreqLowAmp.append(math.sin(3.14159 / 85 * i) + 0.15 * math.sin(3.14159 / 5 * i))
        fakeNoisy.append(math.sin(3.14159/85 * i)+rnd.random()*0.15)


    plt.figure()
    plt.plot(range(len(trainingGood)), trainingGood,linestyle="",marker=".", label = f"Training data with smoothness score: {-1.708}")
    plt.plot(range(len(trainingBad)), trainingBad,linestyle="",marker=".", label = f"Training data with smoothness score: {-2.477}")
    plt.legend(loc="upper left")
    plt.title(f"Speed profile (m/s) for two movements by a test subject")

    plt.figure()
    plt.plot(range(len(trainingGood)), fakeGood, linestyle="", marker=".",label=f"Sine wave representing training data: {np.round(smoothness.sparc(fakeGood, 12)[0],3)}")
    plt.plot(range(len(trainingGood)), fakeLowFreqLowAmp,label=f"Sine wave with low amplitude and low frequency tremor: {np.round(smoothness.sparc(fakeLowFreqLowAmp, 12)[0],3)}")
    plt.plot(range(len(trainingGood)), fakeHighFreqLowAmp,label=f"Sine wave with low amplitude and high frequency tremor: {np.round(smoothness.sparc(fakeHighFreqLowAmp, 12)[0],3)}")
    plt.plot(range(len(trainingGood)), fakeLowFreqHighAmp,label=f"Sine wave with high amplitude and low frequency tremor: {np.round(smoothness.sparc(fakeLowFreqHighAmp, 12)[0],3)}")
    plt.plot(range(len(trainingGood)), fakeHighFreqHighAmp,label=f"Sine wave with high amplitude and high frequency tremor: {np.round(smoothness.sparc(fakeHighFreqHighAmp, 12)[0],3)}")
    plt.plot(range(len(trainingGood)), fakeNoisy,label=f"Sine wave with random noise added: {np.round(smoothness.sparc(fakeNoisy, 12)[0], 3)}")
    plt.title(f"Smoothness scores for different disturbance in movement")
    plt.legend(loc="upper left")

    plt.show()