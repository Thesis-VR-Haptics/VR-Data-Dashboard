import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from Visualizer import Visualizer

# TODO: transform.velocity.magnitude functie niet meer gebruiken bij unity data

if __name__ == '__main__':
    visualizer = Visualizer("data\QuestController_still.csv")
    #visualizer.applyMovingAvg()
    visualizer.visualize3D()
    visualizer.visualizeHand(True) # True als ge rechts wilt visualizeren
    plt.show()
    visualizer.fft(visualizer.time_vector,visualizer.speed_vector)
    plt.show()