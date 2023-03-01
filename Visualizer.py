import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        speed_vector = []
        acceleration_vector = []
        time = self.original_db.iloc[:, 0]
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
            speed = math.sqrt(((x_axis[i + 1] - x_axis[i]) ** 2 + (
                        y_axis[i + 1] - y_axis[i]) ** 2 + (z_axis[i + 1] - z_axis[i]) ** 2) / (
                                          (time[i + 1] - time[i]) / 3600000))
            speed_vector.append(speed)

        speed_vector = np.array(speed_vector)
        speed_vector = np.append(speed_vector, 0)

        for i in range(len(speed_vector) - 1):
            acceleration = (speed_vector[i + 1] - speed_vector[i]) / ((time.iloc[i + 1] - time.iloc[i]) / 3600000)
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

        speed_vector[speed_vector == 0] = np.nan

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(time, speed_vector / 3.6, "o", markersize=3, color="green")

        svUnity[svUnity == 0.0] = np.nan

        plt.plot(time, svUnity, "o", markersize=3, color="red")
        plt.ylabel("Speed (?)")

        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(time, acceleration_vector, "o", markersize=1)
        plt.ylabel("Acceleration (?)")
        plt.xlabel("Time (ms)")

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
            
        