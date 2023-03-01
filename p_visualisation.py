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
    y_axis_r = np.array(o_db.iloc[:,3])
    z_axis_r = np.array(o_db.iloc[:,4])

    x_axis_l = np.array(o_db.iloc[:, 7])
    y_axis_l = np.array(o_db.iloc[:, 8])
    z_axis_l = np.array(o_db.iloc[:, 9])

    fig_r = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis_r, y_axis_r, z_axis_r)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig_l = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis_l, y_axis_l, z_axis_l)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    # Visualise speed & acceleration vector for right hand
    speed_vector = []
    acceleration_vector = []

    for i in range(len(o_db)-1):
        speed = math.sqrt(((o_db.iloc[i+1, 2]-o_db.iloc[i,2])**2 + (o_db.iloc[i+1, 3]-o_db.iloc[i,3])**2 + (o_db.iloc[i+1, 4]-o_db.iloc[i,4])**2)/((o_db.iloc[i+1,0]-o_db.iloc[i,0])/3600000))
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
    plt.plot(o_db.iloc[:, 0], x_axis_r)
    plt.plot(o_db.iloc[:, 0], y_axis_r)
    plt.plot(o_db.iloc[:, 0], z_axis_r)
    plt.ylabel("Displacement (?)")

    speed_vector[speed_vector==0] = np.nan

    plt.subplot(3,1,2, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], speed_vector/3.6, "o" , markersize = 3,color="green")

    svUnity = np.array(o_db.iloc[:, 5])
    svUnity[svUnity == 0.0] = np.nan

    plt.plot(o_db.iloc[:, 0], svUnity, "o" , markersize = 3,color="red")
    plt.ylabel("Speed (?)")

    plt.subplot(3,1,3, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], acceleration_vector, "o" , markersize = 1 )
    plt.ylabel("Acceleration (?)")
    plt.xlabel("Time (ms)")

    # Visualise speed & acceleration vector for left hand
    speed_vector = []
    acceleration_vector = []

    for i in range(len(o_db) - 1):
        speed = math.sqrt(((o_db.iloc[i + 1, 7] - o_db.iloc[i, 7]) ** 2 + (
                    o_db.iloc[i + 1, 8] - o_db.iloc[i, 8]) ** 2 + (o_db.iloc[i + 1, 9] - o_db.iloc[i, 9]) ** 2) / (
                                      (o_db.iloc[i + 1, 0] - o_db.iloc[i, 0]) / 3600000))
        speed_vector.append(speed)

    speed_vector = np.array(speed_vector)
    speed_vector = np.append(speed_vector, 0)

    for i in range(len(speed_vector) - 1):
        acceleration = (speed_vector[i + 1] - speed_vector[i]) / ((o_db.iloc[i + 1, 0] - o_db.iloc[i, 0]) / 3600000)
        acceleration_vector.append(acceleration)

    acceleration_vector = np.array(acceleration_vector)
    acceleration_vector = np.append(acceleration_vector, 0)

    fig1 = plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    plt.title("Speed and Acceleration for Left Hand")
    plt.plot(o_db.iloc[:, 0], x_axis_l)
    plt.plot(o_db.iloc[:, 0], y_axis_l)
    plt.plot(o_db.iloc[:, 0], z_axis_l)
    plt.ylabel("Displacement (?)")

    plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], speed_vector, "o" , markersize = 1,color="green")
    plt.plot(o_db.iloc[:, 0], o_db.iloc[:, 10], "o", markersize=1, color="red")
    plt.ylabel("Speed (?)")

    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(o_db.iloc[:, 0], acceleration_vector, "o", markersize=1)
    plt.ylabel("Acceleration (?)")
    plt.xlabel("Time (ms)")
    plt.show()