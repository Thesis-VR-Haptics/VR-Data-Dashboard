import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Ideeen om signaal beter te krijgen; moving average toepassen op beweging
    # int algeneen ligt het probleem momenteel bij het positie signaal dat niet smooth is


    o_db = pd.read_csv("QuestController_4215953.csv")

    # Visualise movement in 3D
    x_axis_r = np.array(o_db.iloc[:,2])
    x_axis_r_ma = np.convolve(x_axis_r, np.ones(3),'valid')/3
    x_axis_r_ma = np.append(x_axis_r_ma, 0)
    x_axis_r_ma = np.append(x_axis_r_ma, 0)

    y_axis_r = np.array(o_db.iloc[:,3])
    y_axis_r_ma = np.convolve(y_axis_r, np.ones(3), 'valid') / 3
    y_axis_r_ma = np.append(y_axis_r_ma, 0)
    y_axis_r_ma = np.append(y_axis_r_ma, 0)

    z_axis_r = np.array(o_db.iloc[:,4])
    z_axis_r_ma = np.convolve(z_axis_r, np.ones(3), 'valid') / 3
    z_axis_r_ma = np.append(z_axis_r_ma, 0)
    z_axis_r_ma = np.append(z_axis_r_ma, 0)

    # Visualise speed & acceleration vector for right hand
    speed_vector = []
    acceleration_vector = []

    for i in range(len(o_db)-1):
        speed = math.sqrt(((x_axis_r_ma[i+1]-x_axis_r_ma[i])**2 + (y_axis_r_ma[i+1]-y_axis_r_ma[i])**2 + (z_axis_r_ma[i+1]-z_axis_r_ma[i])**2)/((o_db.iloc[i+1,0]-o_db.iloc[i,0])/3600000))
        speed_vector.append(speed)

    speed_vector = np.array(speed_vector)
    speed_vector = np.append(speed_vector, 0)

    for i in range(len(speed_vector)-1):
        acceleration = (speed_vector[i+1] - speed_vector[i])/((o_db.iloc[i+1,0]-o_db.iloc[i,0])/3600000)
        acceleration_vector.append(acceleration)

    acceleration_vector = np.array(acceleration_vector)
    acceleration_vector = np.append(acceleration_vector,0)

    fig1 = plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.title("Speed and Acceleration for Right Hand")
    plt.plot(o_db.iloc[:, 0], x_axis_r_ma)
    plt.plot(o_db.iloc[:, 0], y_axis_r_ma)
    plt.plot(o_db.iloc[:, 0], z_axis_r_ma)
    plt.ylabel("Displacement (?)")

    speed_vector[speed_vector==0] = np.nan

    plt.subplot(3,1,2, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], speed_vector, "o" , markersize = 3,color="green")

    svUnity = np.array(o_db.iloc[:, 5])
    svUnity[svUnity == 0.0] = np.nan

    plt.plot(o_db.iloc[:, 0], svUnity, "o" , markersize = 3,color="red")
    plt.ylabel("Speed (?)")

    plt.subplot(3,1,3, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], acceleration_vector, "o" , markersize = 1 )
    plt.ylabel("Acceleration (?)")
    plt.xlabel("Time (ms)")

    plt.show()