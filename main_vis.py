import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from Visualizer import Visualizer

if __name__ == '__main__':
    visualizer = Visualizer("QuestController_5946284.csv")
    visualizer.visualize3D()
    visualizer.visualizeHand(True) # True als ge rechts wilt visualizeren
    plt.show()