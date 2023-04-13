import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.interpolate import interp1d
import smoothness
import mysql.connector

class Visualizer():
    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="192.168.9.192",
            user="haptics",
            password="haptics1",
            database="thesisdata"
        )

        self.mycursor = self.mydb.cursor()
        #TODO: bij deze functie rekening houden met parameters username en controller

    def setArrays(self, fn):
        self.filename = fn
        self.original_db = pd.read_csv(fn)
        self.x_axis_r = np.array(self.original_db.iloc[:, 2])
        self.z_axis_r = np.array(self.original_db.iloc[:, 3])
        self.y_axis_r = np.array(self.original_db.iloc[:, 4])

        self.x_axis_l = np.array(self.original_db.iloc[:, 7])
        self.z_axis_l = np.array(self.original_db.iloc[:, 8])
        self.y_axis_l = np.array(self.original_db.iloc[:, 9])

    def getDataFromDB(self, runID = 28):
        self.mycursor.execute("SELECT * FROM oculuscontroller WHERE runID = "+str(runID))
        myresult = self.mycursor.fetchall()

        rundata = pd.DataFrame(myresult)
        rundata = rundata.iloc[:, 1:]
        rundata = rundata.sort_values(by=rundata.columns[0]).reset_index(drop=True)
        return rundata

    def setArraysFromDB(self, db):
        self.original_db = db
        self.x_axis_r = np.array(self.original_db.iloc[:, 2])
        self.z_axis_r = np.array(self.original_db.iloc[:, 3])
        self.y_axis_r = np.array(self.original_db.iloc[:, 4])

        self.x_axis_l = np.array(self.original_db.iloc[:, 7])
        self.z_axis_l = np.array(self.original_db.iloc[:, 8])
        self.y_axis_l = np.array(self.original_db.iloc[:, 9])
        self.time = np.array(self.original_db.iloc[:, 0])

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

    def getUserDB(self):
        self.mycursor.execute("SELECT * FROM users")
        myresult = self.mycursor.fetchall()

        users = pd.DataFrame(myresult)
        users = users.iloc[:, :]
        return users

    def getRunsDB(self):
        self.mycursor.execute("SELECT * FROM rundata")
        myresult = self.mycursor.fetchall()

        rundata = pd.DataFrame(myresult)
        rundata = rundata.iloc[:, :]
        return rundata

    def getRangesDB(self, runID):
        self.mycursor.execute("SELECT * FROM rundata WHERE r_id = "+str(runID))
        myresult = self.mycursor.fetchall()

        rundata = pd.DataFrame(myresult)
        lRl = rundata[4][0]
        lRR = rundata[5][0]
        lRF = rundata[6][0]
        lRU = rundata[7][0]

        rRl = rundata[8][0]
        rRR = rundata[9][0]
        rRF = rundata[10][0]
        rRU = rundata[11][0]

        return rRl, rRR,rRF,rRU

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

    def getValues(self):
        return 0,0,0, 1,1,1

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

    def calculateSmoothness(self):
        self.initializeVectors(True)
        time = self.time

        self.svUnity[self.svUnity == 0.0] = np.nan
        self.time_vector = np.array(time * 1000)
        self.sv_unity = self.svUnity

        f = interp1d(self.time_vector, self.speed_vector)
        x_uniform = np.arange(int(math.ceil(self.time_vector[0])), int(self.time_vector[-1]), 12)
        ynew = f(x_uniform)
        # Reason for the ms unit of the time signal is that otherwise i can't interpolate. This doesn't influence sparc.
        print("Sparc analysis: ")
        sm = smoothness.sparc(ynew, 12)[0]
        print(sm)
        return sm

    def sparcOnApples(self):
        ol = self.original_db[(self.original_db[14] == 6) & (self.original_db[13] == 3)].reset_index(drop=True)
        olVisualizer = Visualizer()
        olVisualizer.setArraysFromDB(db = ol)
        olsm = np.round(olVisualizer.calculateSmoothness(),3)
        olavg = np.round(np.mean(olVisualizer.speed_vector),3)

        il = self.original_db[(self.original_db[14] == 5) & (self.original_db[13] == 3)].reset_index(drop=True)
        ilVisualizer = Visualizer()
        ilVisualizer.setArraysFromDB(db=il)
        ilsm = np.round(ilVisualizer.calculateSmoothness(),3)
        ilavg = np.round(np.mean(ilVisualizer.speed_vector),3)

        ori = self.original_db[(self.original_db[14] == 7) & (self.original_db[13] == 3)].reset_index(drop=True)
        oriVisualizer = Visualizer()
        oriVisualizer.setArraysFromDB(db=ori)
        orism = np.round(oriVisualizer.calculateSmoothness(),3)
        oriavg = np.round(np.mean(oriVisualizer.speed_vector),3)

        ir = self.original_db[(self.original_db[14] == 8) & (self.original_db[13] == 3)].reset_index(drop=True)
        irVisualizer = Visualizer()
        irVisualizer.setArraysFromDB(db=ir)
        irsm = np.round(irVisualizer.calculateSmoothness(),3)
        iravg = np.round(np.mean(irVisualizer.speed_vector), 3)
        return olsm, ilsm, orism, irsm, olavg, ilavg, oriavg, iravg

    def sparcOnLvl2(self):
        croissant = self.original_db[(self.original_db[14] == 1) & (self.original_db[13] == 2)].reset_index(drop=True)
        milkshake = self.original_db[(self.original_db[14] == 0) & (self.original_db[13] == 2)].reset_index(drop=True)

        crVisualizer = Visualizer()
        crVisualizer.setArraysFromDB(db = croissant)
        crsm = np.round(crVisualizer.calculateSmoothness(),3)
        cravg = np.round(np.mean(crVisualizer.speed_vector),3)

        msVisualizer = Visualizer()
        msVisualizer.setArraysFromDB(db=milkshake)
        mssm = np.round(msVisualizer.calculateSmoothness(), 3)
        msavg = np.round(np.mean(msVisualizer.speed_vector), 3)

        return crsm, cravg, mssm, msavg

    def sparcOnLvl1(self):
        coffee = self.original_db[(self.original_db[14] == 1) & (self.original_db[13] == 1)].reset_index(drop=True)
        cupcake = self.original_db[(self.original_db[14] == 0) & (self.original_db[13] == 1)].reset_index(drop=True)

        coVisualizer = Visualizer()
        coVisualizer.setArraysFromDB(db = coffee)
        cosm = np.round(coVisualizer.calculateSmoothness(),3)
        coavg = np.round(np.mean(coVisualizer.speed_vector),3)

        ckVisualizer = Visualizer()
        ckVisualizer.setArraysFromDB(db=cupcake)
        cksm = np.round(ckVisualizer.calculateSmoothness(), 3)
        ckavg = np.round(np.mean(ckVisualizer.speed_vector), 3)

        return cosm, coavg, cksm, ckavg

    def getAxesForPart(self, part):
        temp_db = self.original_db[self.original_db[13] == part].reset_index(drop=True)
        x = np.array(temp_db.iloc[:,2])
        z = np.array(temp_db.iloc[:, 3])
        y = np.array(temp_db.iloc[:, 4])
        t = temp_db.iloc[:, 0] / 1000
        return x,y,z,t





        