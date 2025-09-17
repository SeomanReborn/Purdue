import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#######R CONTENT VIDEO -- EXPLORATORY DATA ANALYSIS#######

# Example data
data = pd.read_csv('http://brycejdietrich.com/mtcars.csv')

mean = np.mean(data['mpg'])
median = np.median(data['mpg'])
std_dev = np.std(data['mpg'])
variance = np.var(data['mpg'])
quantiles = np.percentile(data['mpg'], [25, 50, 75])
iqr = np.percentile(data['mpg'], 75) - np.percentile(data['mpg'], 25)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Quantiles: {quantiles}")
print(f"Interquartile Range (IQR): {iqr}")

quantiles*iqr+[-1.5,0,1.5]

x1=x2=range(1,6) #can't assign things to a range abject
x1
x2[5]=50

x1=x2=list(range(1,6))
x2[4]=50

np.mean(x1)
np.mean(x2)

np.median(x1)
np.median(x2)

np.percentile(x1, 75) - np.percentile(x1, 25)
np.percentile(x2, 75) - np.percentile(x2, 25)

np.std(x1)
np.std(x2)

#######R CONTENT VIDEO -- OUTLIERS AND BOXPLOTS#######
####BOXPLOTS####
# Create the boxplot (matplotlib)
# R FUNCTION: boxplot(mpg)
plt.boxplot(data['mpg'])

# Add labels and title (optional)
plt.title('Boxplot Example')
plt.xlabel('MPG')
plt.ylabel('Values')

# Show the plot
plt.show()

# Create the boxplot (seaborn)
sns.boxplot(data=data['mpg'])

# Add labels and title (optional)
plt.title('Boxplot Example')
plt.xlabel('MPG')
plt.ylabel('Values')

# Show the plot
plt.show()

# Create a boxplot grouped by 'cylinders' (cyl)
# R FUNCTION: boxplot(mpg~cyl,col=2:4)
sns.boxplot(x='cyl', y='mpg', data=data, palette='coolwarm')

# Show the plot
plt.show()

####HISTOGRAMS####
# Create the histogram (matplotlib)
# R FUNCTION: hist(mpg)
plt.hist(data['mpg'], color='skyblue', edgecolor='black')

# Add titles and labels
plt.title('Histogram Example (MPG)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# R FUNCTION: hist(mpg,breaks=100)
# Create the histogram (seaborn)
sns.histplot(data['mpg'], bins=100, color='skyblue')

# Add titles and labels
plt.title('Histogram Example w/ 100 bins (MPG)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()
plt.close()
# Create the histogram with density plot (seaborn)
sns.histplot(data['mpg'], bins=30, kde=True, color='skyblue')

# Add titles and labels
plt.title('Histogram Example w/ 30 bins and kernel density estimate (MPG)')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

####SCATTERPLOTS####
# Create the scatter plot (matplotlib)
plt.close()
plt.scatter(data['wt'], data['mpg'], color='blue', edgecolor='black')

# Add labels and title
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Scatter Plot Example')

# Show the plot
plt.show()

# Create the scatter plot (seaborn)
plt.close()
sns.scatterplot(data=data, x='wt', y='mpg', color='blue')

# Add labels and title
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Scatter Plot Example')

# Show the plot
plt.show()

####SCATTERPLOT MATRIX####
# Create the scatter plot matrix (seaborn)
# R FUNCTION: pairs(mtcars[,c(1,3:7)],pch=16)

# Select the relevant columns (equivalent to mtcars[, c(1, 3:7)])
columns_to_plot = ['mpg', 'disp', 'hp','drat','wt', 'qsec']

# Create a pairplot
plt.close()
sns.pairplot(data[columns_to_plot], markers='o')  # 'o' is the filled circle (equivalent to pch=16)

# Show the plot
plt.show()

####PIE CHART####
# Create the pie chart (matplotlib)
# R FUNCTION: pie(tcyl)

# Data for the pie chart
frequency_table = data['cyl'].value_counts() # create frequency table (equivlanet to table(cyl_cat))

labels_int = frequency_table.index.tolist() # returns the names, which are integers (cylinders) in this case
labels = list(map(str, labels_int)) # convert those integers to strings
sizes = frequency_table.values/sum(frequency_table.values) # proportions of each category
explode = (0.1, 0, 0)  # Optional explode one slice for emphasis (e.g., the largest slice)

# Colors for the slices
colors = ['red', 'green', 'blue'] # list of names colors here: https://matplotlib.org/stable/gallery/color/named_colors.html

# Create the pie chart
plt.close()
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=90)
# autopct formats the percentage display on each slice. What is used here is equal to 1 decimal
# startangle determines which pie slice is at the bottom

# Equal aspect ratio ensures that the pie chart is drawn as a circle.
plt.axis('equal')

# Add a title
plt.title('Pie Chart Example')

# Show the plot
plt.show()

####BARPLOT####
# Create a bar plot (matplotlib)
# R FUNCTION: barplot(tcyl)

# Data for the bar plot
frequency_table = data['cyl'].value_counts() # create frequency table (equivlanet to table(cyl_cat))

labels_int = frequency_table.index.tolist() # returns the names, which are integers (cylinders) in this case
labels = list(map(str, labels_int)) # convert those integers to strings
values = frequency_table.values

# Create the bar plot
plt.close()
plt.bar(labels, values)
plt.title("Bar Plot Example (matplotlib)")
plt.xlabel("Cylinders")
plt.ylabel("Values")
plt.show()

# Create the bar plot (seaborn)
# R FUNCTION: barplot(tcyl)

# Data for the bar plot
frequency_table = data['cyl'].value_counts() # create frequency table (equivlanet to table(cyl_cat))

labels_int = frequency_table.index.tolist() # returns the names, which are integers (cylinders) in this case
labels = list(map(str, labels_int)) # convert those integers to strings
values = frequency_table.values

# Create the bar plot
plt.close()
sns.barplot(x=labels, y=values, colors='blue')
plt.title("Bar Plot Example (seaborn)")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()
