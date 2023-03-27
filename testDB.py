import mysql.connector
import pandas as pd

from Visualizer import Visualizer

if __name__ == '__main__':

    mydb = mysql.connector.connect(
        host="192.168.0.125",
        user="haptics",
        password="haptics1",
        database="thesisdata"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM users")

    myresult = mycursor.fetchall()
    #for x in myresult:
     #   print(x)

    vis = Visualizer()
    vis.setArraysFromDB(vis.getDataFromDB(27))
    vis.sparcOnApples()
    print("test")
