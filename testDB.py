import math

import mysql.connector
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from statistics import mean
import plotly.graph_objects as go
from scipy.interpolate import interp1d
import smoothness

from Visualizer import Visualizer

if __name__ == '__main__':

    def calculateSmoothness(self):
        self.initializeVectors(True)
        time = self.time

        self.svUnity[self.svUnity == 0.0] = np.nan
        self.time_vector = np.array(time * 1000)
        self.sv_unity = self.svUnity

        f = interp1d(self.time_vector, self.speed_vector)
        x_uniform = np.arange(int(math.ceil(self.time_vector[0])), int(self.time_vector[-1]), 12)
        ynew = f(x_uniform)
        sm = smoothness.sparc(ynew, 12)[0]
        return sm

    def getSpeedFig(visualizer, part):
        ol = visualizer.original_db[(visualizer.original_db[13] == part) & (visualizer.original_db[14] == 1)].reset_index(drop=True)
        olVisualizer = Visualizer()
        olVisualizer.setArraysFromDB(db=ol)
        olVisualizer.initializeVectors(True)
        color = olVisualizer.original_db.iloc[:, -1]
        plt.figure()
        plt.plot(olVisualizer.time[:-4], olVisualizer.speed_vector[:-4])
        plt.title(f"Smoothness = {smoothness.sparc(olVisualizer.speed_vector[:-4],12)[0]}")

        perfectFunction = []
        tryfunction = []
        weirdSine = []

        x = -0.4
        for i in range(50):
            perfectFunction.append(((-(x - 5) ** 2 + 20)/37+0.4))#+(math.sin(x))**3 * 0.15)

            weirdSine.append((math.sin(x))**3 * 0.3)
            x += 0.22
            tryfunction.append(1)
        weirdSine = np.array(weirdSine)
        perfectFunction = np.array(perfectFunction)
        plt.figure()
        plt.plot(olVisualizer.time[:-6], perfectFunction)
        plt.title(f"Smoothness = {smoothness.sparc(perfectFunction, 12)[0]}")

        plt.figure()
        plt.plot(olVisualizer.time[:-6], tryfunction)
        plt.title(f"Smoothness = {smoothness.sparc(tryfunction, 12)[0]}")

        plt.figure()
        plt.plot(olVisualizer.time[:-6], weirdSine)
        plt.title(f"Smoothness = {smoothness.sparc(weirdSine, 12)[0]}")

        print(f"Smoothness raw: {calculateSmoothness(olVisualizer)}")
        print(f"Smoothness line: {smoothness.sparc(np.array(tryfunction), 12)[0]}")
        return fig, olVisualizer.speed_vector



    mydb = mysql.connector.connect(
        host="192.168.9.192",
        user="haptics",
        password="haptics1",
        database="thesisdata"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM oculuscontroller WHERE runID = 89")

    myresult = mycursor.fetchall()
    df = pd.DataFrame(myresult)
    visualizer = Visualizer()
    visualizer.setArraysFromDB(df)
    visualizer.initializeVectors()
    fig = make_subplots(rows=1, cols=1)

    visualizer.setArraysFromDB(visualizer.getDataFromDB(89))
    r1, r2, r3, r4 = visualizer.getRangesDB(89)
    visualizer.initializeVectors(True)

    figSpeedlvl2, sVector = getSpeedFig(visualizer, 2)
    print(f"Smoothness smooth signal : {smoothness.sparc(sVector, 12)[0]}")
    plt.show()


