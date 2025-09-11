import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Run the summary() function in R or the describe method in Python for airquality (from the previous week's worksheet).
# For each variable, describe its type and scale. Based on these statistics, describe the shape of each of 
# the quantitative variables.
airquality = pd.read_csv("http://www.brycejdietrich.com/files/airquality.csv")
airquality.describe()

#Ozone: quantitative, ratio, right-skewed (mean > median)
#Solar.R: quantitative, ratio, left-skewed (mean < median)
#Wind: quantitative, ratio, barely right-skewed (mean > median)
#Temp: quantitative, interval, barely left-skewed (mean > median)
#Month: qualitative, ordinal
#Day: qualitative, ordinal
plt.close()
plt.hist(airquality['Wind'])
plt.show()

plt.close()
plt.hist(airquality['Solar.R'])
plt.show()

means = airquality.groupby('Month')['Temp'].mean().reset_index()
print(means)


plt.close()
fig, ax = plt.subplots(figsize=(8, 6))

boxplot = ax.boxplot(
    [group['Temp'] for _, group in airquality.groupby('Month')],
    patch_artist=True,
    labels=airquality['Month'].unique()
)

ax.set_title("Boxplot of Temperature by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Temperature")

# add some fancy coloring
colors = plt.cm.viridis(np.linspace(0, 1, len(boxplot['boxes'])))
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()

plt.close()
OzoneQs = airquality['Ozone'].quantile([0.25, 0.75]).values
iqr = OzoneQs[1] - OzoneQs[0]
OzoneLLandUL = OzoneQs + np.array([-1.5, 1.5]) * iqr
OzoneOutliers = airquality['Ozone'][(airquality['Ozone'] > OzoneLLandUL[1]) & airquality['Ozone'].notna()]
print(OzoneOutliers)
# Have to remove missing values
plt.boxplot(airquality['Ozone'].dropna(), vert=False)
plt.show()


plt.close()
plt.scatter(airquality['Ozone'], airquality['Solar.R'])
plt.show()


plt.close()
frequency_table = airquality['Month'].value_counts().sort_index() # create frequency table (equivlanet to table(cyl_cat))

labels_int = frequency_table.index.tolist() # returns the names, which are integers (cylinders) in this case
labels = list(map(str, labels_int)) # convert those integers to strings
values = frequency_table.values

# Create the bar plot

plt.bar(labels, values)
plt.title("Months")
plt.xlabel("Months")
plt.ylabel("Values")
plt.show()

