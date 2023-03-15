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
        self.z_axis_r = np.array(self.original_db.iloc[:, 3])
        self.y_axis_r = np.array(self.original_db.iloc[:, 4])

        self.x_axis_l = np.array(self.original_db.iloc[:, 7])
        self.z_axis_l = np.array(self.original_db.iloc[:, 8])
        self.y_axis_l = np.array(self.original_db.iloc[:, 9])

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

    def initializeVectors(self, righthand):
        dist = 0
        self.speed_vector = []
        self.acceleration_vector = []
        time = self.original_db.iloc[:, 0] / 1000
        if (righthand):
            self.x_axis = self.x_axis_r
            self.y_axis = self.y_axis_r
            self.z_axis = self.z_axis_r
            self.svUnity = np.array(self.original_db.iloc[:, 5])
        else:
            self.x_axis = self.x_axis_l
            self.y_axis = self.y_axis_l
            self.z_axis = self.z_axis_l
            self.svUnity = np.array(self.original_db.iloc[:, 10])

        for i in range(len(self.original_db) - 1):
            distance = math.sqrt(
                (self.x_axis[i + 1] - self.x_axis[i]) ** 2 + (self.y_axis[i + 1] - self.y_axis[i]) ** 2 + (
                            self.z_axis[i + 1] - self.z_axis[i]) ** 2)
            dist += distance
            speed = distance / (time[i + 1] - time[i])  # Value in m/s
            self.speed_vector.append(speed)

        self.speed_vector = np.array(self.speed_vector)
        self.speed_vector = np.append(self.speed_vector, 0)

        for i in range(len(self.speed_vector) - 1):
            acceleration = (self.speed_vector[i + 1] - self.speed_vector[i]) / (time.iloc[i + 1] - time.iloc[i])
            self.acceleration_vector.append(acceleration)

        self.acceleration_vector = np.array(self.acceleration_vector)
        self.acceleration_vector = np.append(self.acceleration_vector, 0)
        self.time = time

    def visualizeHand(self, righthand):
        self.initializeVectors(righthand)
        time = self.time
        fig1 = plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        plt.title("Speed and Acceleration for Right Hand")
        plt.plot(time, self.x_axis)
        plt.plot(time, self.y_axis)
        plt.plot(time, self.z_axis)
        plt.ylabel("Displacement (?)")

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time, self.speed_vector, "o", markersize=3, color="green")
        self.svUnity[self.svUnity == 0.0] = np.nan

        plt.plot(time, self.svUnity, "o", markersize=3, color="red")
        plt.ylabel("Speed (m/s)")

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(time, self.acceleration_vector, "o", markersize=1)
        plt.ylabel("Acceleration (?)")
        plt.xlabel("Time (ms)")

        self.time_vector = np.array(time*1000)
        self.sv_unity = self.svUnity

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
        # Reason for the ms unit of the time signal is that otherwise i can't interpolate. This doesn't influence sparc.
        print("Sparc analysis: ")
        print(smoothness.sparc(ynew, 12)[0])

        # Rescale time signal to seconds
        x_uniform = [float(x) for x in x_uniform]
        x_uniform = np.array(x_uniform)
        x_uniform *= 0.001

        N = x_uniform.size
        yf = scipy.fftpack.fft(ynew)
        xf = np.linspace(0.0, 1.0 / (2.0 * 0.012), N // 2)
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
        