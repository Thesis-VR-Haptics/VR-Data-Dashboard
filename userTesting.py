import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import smoothness
import random as rnd

if __name__ == '__main__':
    x = [0,1,2,3,4]
    tine = [66.1, 79.5, 81.4, 83.4, 79.7]
    kato = [68.5, 65.5, 69.0, 78.1, 83.2]
    tobit =[80.8, 77.0, 76.5, 60.8, 72.8]
    toon = [73.9, 74.6, 65.7,71.5,73.8]
    mama = [65.7, 57.4, 83.3, 67.2, 84.7]
    papa = [24.9, 67.3, 38.8, 37.9, 57.8]
    mami = [79.8, 74.1, 76.8, 82.8, 84.3]
    papi = [59.5, 76.7, 38.9, 69.1, 80.9]
    anna = [77.1,77.3,82.1,79.4, 83.6]
    ward = [84.3,90.35,88.3,84.5,89.0]

    plt.figure()
    plt.plot(x, tine, label='User 1')
    plt.plot(x, kato, label='User 2')
    plt.plot(x, tobit, label='User 3')
    plt.plot(x, toon, label='User 4')
    plt.plot(x, mama, label='User 5')
    plt.plot(x, papa, label='User 6')
    plt.plot(x, mami, label='User 7')
    plt.plot(x, papi, label='User 8')
    plt.plot(x, anna, label='User 9')
    plt.plot(x, ward, label='User 10')

    plt.xlabel('X')
    plt.ylabel('Movement Quality score')
    plt.title('Movement Quality progress over 5 trainings')
    plt.xticks(np.arange(5), x)

    plt.figure()
    average = np.mean([tine, kato, tobit, toon, mama, papa, mami, papi, anna, ward], axis=0)
    plt.plot(x, average)
    plt.xlabel('X')
    plt.ylabel('Average Score')
    plt.title('Average Movement Quality score of all test subjects over 5 trainings')
    plt.xticks(np.arange(5), x)

    plt.show()