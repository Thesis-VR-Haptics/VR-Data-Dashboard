import mysql.connector

if __name__ == '__main__':

    mydb = mysql.connector.connect(
        host="192.168.0.125",
        user="haptics",
        password="haptics1",
        database="thesisdata"
    )

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM oculuscontroller")

    myresult = mycursor.fetchall()

    for x in myresult:
        print(x)
