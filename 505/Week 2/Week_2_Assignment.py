from sklearn import datasets
iris = datasets.load_iris()
import numpy as np
import pandas as pd

# Problem 1
#Found this syntax to convert via copilot. I couldn't find an example in our introduction. pd.DataFrame(iris) did not work.
irispd = pd.DataFrame(iris.data, columns=iris.feature_names)
irisNames = pd.DataFrame(iris.target, columns=['Names'])
irisTotal = pd.concat([irispd, irisNames], ignore_index=False, axis=1)
irisTotal['Names'] = irisTotal['Names'].apply(lambda x: iris.target_names[x])
x = len(irisTotal.columns)
print(f"Number of columns is {x}") #5 Columns found

# Problem 2
sub1 = irisTotal.iloc[:9]
temp = irisTotal.iloc[[len(irispd)-1]]
sub1 = pd.concat([sub1, temp], ignore_index=True)
print(sub1)


# Problem 3
sub2 = irisTotal.loc[irispd["sepal width (cm)"]<2.4]
sub2.drop(['petal length (cm)', 'petal width (cm)'], axis=1, inplace=True) #Asks Copilot about this syntax
print(sub2)

# Problem 4
tempVector = irisTotal['Names'].apply(lambda x: 100 if x == 'versicolor' else 0)
Versicolor_Is_The_Best = np.array(tempVector)
print(Versicolor_Is_The_Best)


# Problem 5
sw = np.array(irisTotal['sepal width (cm)'])
swMean = np.mean(sw)
swMedian = np.median(sw)
swMax = np.max(sw)
swMin = np.min(sw)
print(f"Mean: {swMean:.2f}")
print(f"Median: {swMedian:.2f}")
print(f"Max: {swMax:.2f}")
print(f"Min: {swMin:.2f}")

# Problem 6
count = 0
total = 0
for temp in sw:
  total = total + temp
  count = count + 1
  if total>100:
    print(f"It took {count} loops to exceed 100")
    break
    
# Problem 7
def cmtoin(cm):
  return cm/2.54

sw_in = cmtoin(sw)
print(sw_in[:7])

# Problem 8
import matplotlib.pyplot as plt
plt.scatter(irisTotal['sepal length (cm)'], irisTotal['petal length (cm)'], marker='x', color='blue',s=100)
plt.xlabel('Sepal Length in Centimeters')
plt.ylabel('Petal Length in Cetimeters')
plt.title('A Comparison of Sepal and Petal Lengths')


plt.show()

