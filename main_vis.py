import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import smoothness

from Visualizer import Visualizer

if __name__ == '__main__':
    #visualizer = Visualizer("data\QuestController_notJittery.csv")
   # visualizer = Visualizer("data\QuestController_jittery.csv")
   # visualizer = Visualizer("data\QuestController_vJittery.csv")
    visualizer = Visualizer()
    visualizer.setArrays("data\QApplesFromDatabase")
    visualizer.visualize3D()
    visualizer.visualizeHand(True) #True als ge rechts wilt visualizeren
    #plt.show()
    #visualizer.fft(visualizer.time_vector,visualizer.speed_vector)
    #visualizer.fft(visualizer.time_vector, visualizer.sv_unity)
    visualizer.sparcOnApples()
    plt.show()