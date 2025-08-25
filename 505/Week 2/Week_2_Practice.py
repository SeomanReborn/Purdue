import pandas as pd
import numpy as np
airquality = pd.read_csv("http://www.brycejdietrich.com/files/airquality.csv")

print(airquality)

print(airquality["Ozone"].mean())

print(airquality["Wind"][airquality['Month']==7].min())

print(airquality["Ozone"][airquality["Temp"]==airquality["Temp"].max()])

import matplotlib.pyplot as plt
plt.plot(airquality.index+1, airquality['Temp'])
plt.show()

airquality.loc[airquality["Month"] == 5, 'MonthName'] = 'May'
airquality.loc[airquality['Month'] == 6, 'MonthName'] = 'June'
airquality.loc[airquality['Month'] == 7, 'MonthName'] = 'July'
airquality.loc[airquality['Month'] == 8, 'MonthName'] = 'August'
airquality.loc[airquality['Month'] == 9, 'MonthName'] = 'September'

def F_TO_C(degreeF):
  return (degreeF-32) * 5 /9

airquality["TempC"] = airquality["Temp"].apply(lambda x: F_TO_C(x))
airquality.head

airquality = airquality.drop(columns=['MonthName', 'TempC'])
airquality.head()
