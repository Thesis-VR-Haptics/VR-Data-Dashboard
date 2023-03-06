import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.interpolate import interp1d
import smoothness

class Visualizer():
    def __init__(self, fn):
        self.filename = fn
        self.original_db = pd.read_csv(fn)
        self.x_axis_r = np.array(self.original_db.iloc[:, 2])
        self.y_axis_r = np.array(self.original_db.iloc[:, 3])
        self.z_axis_r = np.array(self.original_db.iloc[:, 4])

        self.x_axis_l = np.array(self.original_db.iloc[:, 7])
        self.y_axis_l = np.array(self.original_db.iloc[:, 8])
        self.z_axis_l = np.array(self.original_db.iloc[:, 9])

    def visualize3D(self):
        fig_r = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.x_axis_r, self.y_axis_r, self.z_axis_r)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.scatter3D(self.x_axis_l, self.y_axis_l, self.z_axis_l)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def visualizeHand(self, righthand):
        dist = 0
        speed_vector = []
        acceleration_vector = []
        time = self.original_db.iloc[:, 0]/1000
        if(righthand):
            x_axis = self.x_axis_r
            y_axis = self.y_axis_r
            z_axis = self.z_axis_r
            svUnity = np.array(self.original_db.iloc[:, 5])
        else:
            x_axis = self.x_axis_l
            y_axis = self.y_axis_l
            z_axis = self.z_axis_l
            svUnity = np.array(self.original_db.iloc[:, 10])

        for i in range(len(self.original_db) - 1):
            distance = math.sqrt((x_axis[i + 1] - x_axis[i]) ** 2 + (y_axis[i + 1] - y_axis[i]) ** 2 + (z_axis[i + 1] - z_axis[i]) ** 2)
            dist += distance
            speed =  distance/(time[i + 1] - time[i]) # Value in m/s
            speed_vector.append(speed)

        speed_vector = np.array(speed_vector)
        speed_vector = np.append(speed_vector, 0)

        for i in range(len(speed_vector) - 1):
            acceleration = (speed_vector[i + 1] - speed_vector[i]) / (time.iloc[i + 1] - time.iloc[i])
            acceleration_vector.append(acceleration)

        acceleration_vector = np.array(acceleration_vector)
        acceleration_vector = np.append(acceleration_vector, 0)

        fig1 = plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        plt.title("Speed and Acceleration for Right Hand")
        plt.plot(time, x_axis)
        plt.plot(time, y_axis)
        plt.plot(time, z_axis)
        plt.ylabel("Displacement (?)")
        self.speed_vector = speed_vector

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time, speed_vector, "o", markersize=3, color="green")
        svUnity[svUnity == 0.0] = np.nan

        plt.plot(time, svUnity, "o", markersize=3, color="red")
        plt.ylabel("Speed (m/s)")

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(time, acceleration_vector, "o", markersize=1)
        plt.ylabel("Acceleration (?)")
        plt.xlabel("Time (ms)")

        self.time_vector = np.array(time*1000)

    def applyMovingAvg(self, window = 3):
        self.x_axis_r = np.convolve(self.x_axis_r, np.ones(window),'valid')/window
        for x in range(window-1):
            self.x_axis_r = np.append(self.x_axis_r, 0)

        self.y_axis_r = np.convolve(self.y_axis_r, np.ones(window), 'valid') / window
        for x in range(window-1):
            self.y_axis_r = np.append(self.y_axis_r, 0)

        self.z_axis_r = np.convolve(self.z_axis_r, np.ones(window), 'valid') / window
        for x in range(window-1):
            self.z_axis_r = np.append(self.z_axis_r, 0)

        self.x_axis_l = np.convolve(self.x_axis_l, np.ones(window), 'valid') / window
        for x in range(window-1):
            self.x_axis_l = np.append(self.x_axis_l, 0)

        self.y_axis_l = np.convolve(self.y_axis_l, np.ones(window), 'valid') / window
        for x in range(window-1):
            self.y_axis_l = np.append(self.y_axis_l, 0)

        self.z_axis_l = np.convolve(self.z_axis_l, np.ones(window), 'valid') / window
        for x in range(window-1):
            self.z_axis_l = np.append(self.z_axis_l, 0)

    def fft(self, timevector, speedvector):
        # To use sample code, i need uniform samples.
        # I found that the amount of time between samples is on average 14.14ms, it ranges from 12 to 15 ms
        f = interp1d(timevector, speedvector)
        x_uniform = np.arange(int(math.ceil(timevector[0])), int(timevector[-1]), 12)
        ynew = f(x_uniform)
        print("Sparc analysis: ")
        print(smoothness.sparc(ynew, 12)[0])

        N = x_uniform.size
        yf = scipy.fftpack.fft(ynew)
        xf = np.linspace(0.0, 1.0 / (2.0 * 12), N // 2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        plt.show()
        