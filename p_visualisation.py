import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    o_db = pd.read_csv("oculus_data_csv.csv")

    # Visualise movement in 3D
    x_axis = np.array(o_db.iloc[:,2])
    y_axis = np.array(o_db.iloc[:,3])
    z_axis = np.array(o_db.iloc[:,4])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_axis, y_axis, z_axis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    # Visualise speed vector
    speed_vector = []
    acceleration_vector = []
    for i in range(len(o_db)-1):
        speed = math.sqrt(((o_db.iloc[i+1, 2]-o_db.iloc[i,2])**2 + (o_db.iloc[i+1, 3]-o_db.iloc[i,3])**2 + (o_db.iloc[i+1, 4]-o_db.iloc[i,4])**2)/((o_db.iloc[i+1,8]-o_db.iloc[i,8])/3600000))
        speed_vector.append(speed)

    speed_vector = np.array(speed_vector)
    speed_vector = np.append(speed_vector, 0)

    for i in range(len(speed_vector)-1):
        acceleration = (speed_vector[i+1] - speed_vector[i])/((o_db.iloc[i+1,8]-o_db.iloc[i,8])/3600000)
        acceleration_vector.append(acceleration)

    acceleration_vector = np.array(acceleration_vector)
    acceleration_vector = np.append(acceleration_vector,0)

    fig1 = plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.title("Speed and Acceleration")
    plt.plot(o_db.iloc[:,8], x_axis)
    plt.plot(o_db.iloc[:, 8], y_axis)
    plt.plot(o_db.iloc[:, 8], z_axis)
    plt.ylabel("Displacement (?)")

    plt.subplot(3,1,2, sharex=ax1)
    plt.plot(o_db.iloc[:, 8], speed_vector)
    plt.ylabel("Speed (?)")

    plt.subplot(3,1,3, sharex=ax1)
    plt.plot(o_db.iloc[:, 8], acceleration_vector)
    plt.ylabel("Acceleration (?)")
    plt.xlabel("Time (ms)")
    plt.show()
    print("Test")