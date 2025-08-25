###VIDEO 1: WHAT IS PYTHON?
# (1) Python is an interpreted language, meaning that it executes code line by line rather than compiling the entire code at once. 
# (2) In Python, you don't need to explicitly declare variable types.

x = 10       # x is an integer
y = "Hello"  # y is a string

# (3) Python has a clean, straightforward syntax that resembles plain English, making it easy to learn, even for beginners.

print("Hello, World!")

# (4) Python supports both object-oriented and procedural programming paradigms.

# (5) Python has a large community which provides many modules (libraries) and resources.

import numpy as np
import pandas as pd

# Ultimately, over the next several videos, I will show you how to do various tasks in Python 3. These will mirror what you 
# have already learned in R.

###VIDEO 2: HOW DO I DO BASIC CALCULATIONS IN PYTHON?

# Addition
a = 10 + 5  # result: 15

# Subtraction
b = 10 - 5  # result: 5

# Multiplication
c = 10 * 5  # result: 50

# Division
d = 10 / 5  # result: 2.0

# Exponent (raising a number to the power of another)
e = 2 ** 3  # result: 8

# Modulus (remainder after division)
f = 10 % 3  # result: 1

# Integer Division (returns the whole number part of the division)
g = 10 // 3  # result: 3

# PEMDAS
h = 2 + 3 * 5  # result: 17 (multiplication first, then addition)
i = (2 + 3) * 5  # result: 25 (parentheses first, then multiplication)

# NOTICE how I am writing my comments in this video?
# This is an example of a single-line comment

x = 10  # This is an inline comment explaining that x is assigned the value 10
print(x)  # Print the value of x

# When I write long comments, I often like to write multiple single-line comments
# like this.

"""
However, you can also use triple quotes to create a multi-line comment.
It can span across multiple lines, and Python will ignore it
if it's not assigned to a variable or used as a docstring.
"""

###VIDEO 3: HOW DO I USE VARIABLES AND FUNCTIONS IN PYTHON?

# Declaring variables
x = 10
y = 5
result = (x + y) * 2  # result: 30

#Declaing a simple function (notice the spaces)
def add_two_numbers(num1, num2):
    return num1 + num2

sum = add_two_numbers(10, 5)  # result: 15

# Assigning None
a = None

# Checking for None
if a is None:
    print("This value is missing.")

# Example: TypeError
result = None + 5  # This will raise a TypeError

##NUMPY
#NumPy (Numerical Python) is a VERY popular module that is (essentially) the swiss army knife of numerical computing.

# Creating a numpy array with a missing value
arr = np.array([1, 2, np.nan, 4])

# Check for NaN
print(np.isnan(arr))  # [False, False, True, False]

result = arr + 5
print(result)  # result: [ 6.  7. nan  9.]

##PANDAS
# Pandas is a powerful and popular open-source data analysis and manipulation library in Python.

# Creating a pandas Series with missing values
data = pd.Series([1, 2, None, 4])

# Checking for missing values
print(data.isnull())  # result: [False, False, True, False]

# Fill missing values
data_filled = data.fillna(0)
print(data_filled)  # result: [1.0, 2.0, 0.0, 4.0]

# Drop rows with missing values
data_dropped = data.dropna()
print(data_dropped)  # result: [1.0, 2.0, 4.0]

###VIDEO 4: HOW DO I USE BOOLEAN OPERATORS IN PYTHON?

#In Python, a boolean value is a data type that can only have one of two values: True or False.

# Basic boolean values
a = True
b = False

if a:
    print("This is true.")
else:
    print("This is false.")

# However, consider this if/then statement...

number = 7
if number:
    print(number) 

# If number equals what?

# Now, consider this if/then statement...

number = 0
if number:
    print(number) 
    
# Why doesn't it print zero?

# In Python, individual values can evaluate as either True or False. 
# Values that evaluate to True are "Truthy", and values that evaluate to False are "Falsy".

# Examples of truthy values include:
# (1) Non-zero numbers (e.g., 1, -5, 3.14)
# (2) Non-empty strings (e.g., "hello", "world")
# (3) Non-empty lists, tuples, sets, and dictionaries (e.g.,, ("a", "b"), {1, 2, 3}, {"a": 1})
# (4) The special value True 

# Examples of falsy values include:
# (1) The number 0
# (2) Empty strings (e.g., "")
# (3) Empty lists, tuples, sets, and dictionaries (e.g., [], (), set(), {})
# (4) The special values False and None

# In Python, the bool() function is used to convert a value to a Boolean value, which can be either True or False.

# Falsy values
print(bool(False))      # False
print(bool(None))       # False
print(bool(0))          # False
print(bool(""))         # False
print(bool([]))         # False
print(bool({}))         # False

# Truthy values
print(bool(1))          # True
print(bool("Hello"))    # True
print(bool([1, 2, 3]))  # True
print(bool({"key": "value"}))  # True

# AND operator
print(True and False)  # False

# OR operator
print(True or False)   # True

# NOT operator
print(not True)        # False

# Comparison examples
x = 5
y = 10

print(x == y)  # False
print(x != y)  # True
print(x > y)   # False
print(x < y)   # True

# Chained comparison
x = 5
print(1 < x < 10)  # True (checks if x is greater than 1 and less than 10)

# Let's return to our original if/else statement.
# However, instead of declaring a as True, we will declare it as 0
a = 0
b = 1927835235

if a:
    print("x is truthy")
else:
    print("x is falsy")  # This will print, because 0 is falsy.

if b:
    print("x is truthy")  # This will print, because any integer other than 0 is truthy.
else:
    print("x is falsy") 

###VIDEO 5: HOW DO I USE VECTORS IN PYTHON?

# Creating a simple vector using a Python list
vector = [1, 2, 3]

# Manually adding two vectors
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

# Adding vectors element-wise
result = [vector1[i] + vector2[i] for i in range(len(vector1))]
print(result)  # result: [5, 7, 9]

# Multiplying vector by a scalar
scalar = 2
result = [scalar * element for element in vector]
print(result)  # [2, 4, 6]

import numpy as np

# Creating a vector with numpy
vector = np.array([1, 2, 3])
print(vector)  # result: [1 2 3]

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Element-wise vector addition
result = vector1 + vector2
print(result)  # result: [5 7 9]

# Scalar multiplication
scalar = 2
result = scalar * vector1
print(result)  # result: [2 4 6]

# Dot product
dot_product = np.dot(vector1, vector2)
print(dot_product)  # result: 32 (1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32)

# Magnitude of a vector
magnitude = np.linalg.norm(vector1)
print(magnitude)  # result: 3.7416573867739413 (square root of (1^2 + 2^2 + 3^2))

# Cross product (a2b3 - a3b2, a3b1 - a1b3, a1b2 - a2b1)
#vector1 = np.array([1, 2, 3]) = [a1, a2, a3]
#vector2 = np.array([4, 5, 6]) = [b1, b2, b3]
cross_product = np.cross(vector1, vector2)
print(cross_product)  # result: [-3  6 -3] (2×6−3×5, 3×4−1×6, 1×5−2×4) = (12−15, 12−6, 5−8) = (−3, 6, −3)

###VIDEO 6: HOW DO I USE SOME OF MY FAVORITE R FUNCTIONS IN PYTHON?
##R code
##seq(1, 10)

# Start from 1, end at 10 (excluding 10)
sequence = range(1, 11)  # result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(sequence))

##R code
#seq(1, 10, by = 2)

# Step size of 2
sequence = range(1, 11, 2)  # result: [1, 3, 5, 7, 9]
print(list(sequence))

##R code
#seq(1, 10, by = 0.5)

import numpy as np

# Generating a sequence from 1 to 10 with a step of 0.5
sequence = np.arange(1, 10.5, 0.5)  # result: [1.  1.5  2.  2.5 ... 10.]
print(sequence)

##R code
#seq(10, 1, by = -1)

# Generating a sequence from 10 to 1
sequence = range(10, 0, -1)  # result: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
print(list(sequence))

##R code
#rep(5, times = 3)
# result: [5, 5, 5]

# Repeating a single value 3 times
value = 5
repeated_values = [value] * 3  # result: [5, 5, 5]
print(repeated_values)

##R code
#rep(c(1, 2, 3), times = 2)
# result: [1, 2, 3, 1, 2, 3]

# Repeating a list [1, 2, 3] two times
values = [1, 2, 3]
repeated_list = values * 2  # result: [1, 2, 3, 1, 2, 3]
print(repeated_list)

###R code
# paste('/path/to/bryce/is/','awesome.csv',sep='')
# result: "/path/to/bryce/is/awesome.csv"

path = '/path/to/bryce/is/' + 'awesome.csv'
print(path)

###R code
# strsplit("Bryce is awesome, uh, awesome?", ",")
# result: "Bryce is awesome" " uh"              " awesome?" 

sentence = 'Bryce is awesome, uh, awesome?'
print(sentence.split(','))

###R code
# gsub('Bryce','Tim','Bryce is awesome!')
# result: "Tim is awesome!"

import re

# Python equivalent
text = 'Bryce is awesome!'
result = re.sub("Bryce", "Tim", text)
print(result)

###VIDEO 7: HOW DO I USE MATRICES IN PYTHON?

# Creating a 3x3 matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Accessing elements
print(matrix[0])  # result: [1, 2, 3] (first row)
print(matrix[1][2])  # result: 6 (element in the second row, third column)

##R code
# R code
#rbind(c(1, 2), c(3, 4))
# result: 
#      [,1] [,2]
# [1,]    1    2
# [2,]    3    4

import numpy as np

# Creating two arrays
array1 = np.array([1, 2])
array2 = np.array([3, 4])

result = np.row_stack([array1, array2])
print(result)
# result:
# [[1 2]
#  [3 4]]

# R code
#rbind(data.frame(a = 1, b = 2), data.frame(a = 3, b = 4))
# result:
#   a b
# 1 1 2
# 2 3 4

import pandas as pd

# Creating two dataframes
df1 = pd.DataFrame({'a': [1], 'b': [2]})
df2 = pd.DataFrame({'a': [3], 'b': [4]})

# Using pandas.concat() to combine by rows
result = pd.concat([df1, df2], ignore_index=True)
print(result)
# result:
#    a  b
# 0  1  2
# 1  3  4

# R code
#cbind(c(1, 2), c(3, 4))
# result:
#      [,1] [,2]
# [1,]    1    3
# [2,]    2    4

import numpy as np

# Creating two arrays (vectors)
array1 = np.array([1, 2])
array2 = np.array([3, 4])

# Using column_stack() to combine by columns
result = np.column_stack([array1, array2])
print(result)
# result:
# [[1 3]
#  [2 4]]

# R code
#cbind(data.frame(a = 1:2), data.frame(b = 3:4))
# result:
#   a b
# 1 1 3
# 2 2 4

import pandas as pd

# Creating two dataframes
df1 = pd.DataFrame({'a': [1, 2]})
df2 = pd.DataFrame({'b': [3, 4]})

# Using pandas.concat() to combine by columns
result = pd.concat([df1, df2], axis=1)
print(result)
# result:
#    a  b
# 0  1  3
# 1  2  4

# Using join to combine by columns
result = df1.join(df2)
print(result)
# result:
#    a  b
# 0  1  3
# 1  2  4

###VIDEO 8: HOW DO I USE DATA FRAMES IN PYTHON?

import pandas as pd

# Creating a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print(df)
# df.to_csv('/path/to/example_data.csv')

# Creating a DataFrame from a CSV file
# df = pd.read_csv('/path/to/example_data.csv')
df = pd.read_csv('http://brycejdietrich.com/example_data.csv')
print(df)

# Access the 'Age' column
age_column = df['Age']
print(age_column)

# Access the 'Age' column using loc
age_column = df.loc[:, 'Age']
print(age_column)

# Access the 'Age' column using iloc (by column index)
age_column = df.iloc[:, 1]  # 'Age' is the second column (index 1)
print(age_column)

# Access both 'Name' and 'City' columns
subset = df[['Name', 'City']]
print(subset)

# Access the row with index label 1 (which is the second row)
row = df.loc[1]
print(row)

# Access the second row by its position (index 1)
row = df.iloc[1]
print(row)

# Access rows with index 0 and 2
rows = df.loc[[0, 2]]
print(rows)

# Access the first two rows (positions 0 and 1)
rows = df.iloc[:2]
print(rows)

# Access the rows where the 'Age' column is greater than 30
rows = df[df['Age'] > 30]
print(rows)

# Access the value in the second row and 'City' column
city = df.at[1, 'City']
print(city)

# Access the value in the second row and 'City' column, then change
df.at[1, 'City'] = 'Kansas City'
print(df)

###VIDEO 9: HOW DO I CREATE LOOPS IN PYTHON?

# A list of numbers
numbers = [1, 2, 3, 4, 5]

# For loop to iterate over the list and print each number
for num in numbers:
    print(num)

# Using range() to loop through numbers 0 to 4
for i in range(5):
    print(i)

# Initialize a counter for while loop
counter = 1

# While loop to run as long as counter is less than or equal to 5
while counter <= 5:
    print(counter)
    counter += 1  # Increment the counter by 1

# Initialize a counter for while loop
counter = 1

# While loop that runs indefinitely until break is encountered
while True:
    print(counter)
    counter += 1
    if counter > 5:
        break  # Exit the loop when counter is greater than 5

# Initialize a counter for while loop
counter = 0

# While loop to demonstrate continue statement
while counter < 5:
    counter += 1
    if counter == 3:
        continue  # Skip the rest of the loop when counter is 3
    print(counter)



