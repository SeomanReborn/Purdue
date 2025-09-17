import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stemgraphic
from scipy.special import comb  # For combinatorial calculations (see line 209)
from scipy.stats import hypergeom # For dhyper() equivalent (see line 225)
from scipy.stats import binom # For dbinom() equivalent (see line 264)
from scipy.stats import poisson # For dpois() equivalent (see line 304)
from scipy.stats import nbinom # For dnbinom() equivalent (see line 324)
from scipy.stats import geom # For dgeom() equivalent (see line 338)
from scipy.stats import norm # For pnorm() equivalent (see line 349)
from scipy.stats import probplot # For qqnorm() equivalent (see line 367)
from scipy.stats import lognorm # For plnorm() equivalent (see line 429)
from scipy.stats import expon # For pexp() equivalent (see line 438)
from scipy.stats import gamma # For pgamma() equivalent (see line 453)
from scipy.stats import weibull_min # For pweibull() equivalent (see line 462)

###CONTENT VIDEOS -- Summarizing Data (WEEK 4)###
# Data from Table 3.1
Claim = list(range(1,41))

Days = [48,41,35,36,37,26,36,46,35,47,
        35,34,36,42,43,36,56,32,46,30,
        37,43,17,26,28,27,45,33,22,27,
        16,22,33,30,24,23,22,30,31,17]

# Numerical summaries
n=len(Days)
n
n/2
n/2 + 1

Days.sort() # this will change the Days object
sorted(Days) # this will leave the Days object unchanged

sorted(Days)[19:21] # this is equivalent to this in R: sort(Days)[20:21]

# Some more explanation on this point...
# Python lists are zero-indexed, meaning it starts at 0
# So, 19 and 20 accesses the 20th and 21st elements of the list
# 19:21 will begin at 19 indice and continue tunil it reaches the 21st indice

np.mean(sorted(Days)[19:21]) # this is equivalent to n/2 to n/2+1 (20:21)
np.median(Days)

np.quantile(Days, [.10]) # give me the value that exists at the 10th percentile (quantile(Days,.10))
np.quantile(Days, [.90]) # give me the value that exists at the 90th percentile (quantile(Days,.90))

quarts = np.quantile(Days, [.25, .50, .75])
quarts # notice 33.5 is the middle element

quarts[2]-quarts[0] # same as quarts[3]-quarts[1] in R which is IQR(Days)
iqr = quarts[2]-quarts[0]
quarts[0]-1.5*iqr # in a boxplot this corresponds with the lower whisker, with values falling below being "outliers"
quarts[2]+1.5*iqr # in a boxplot this corresponds with the upper whisker, with values falling above being "outliers"

np.mean(Days)
np.sum(Days)/len(Days)

np.var(Days) # population variance
np.var(Days, ddof=1) # sample variance, which is equivalent to var(Days) in R

def population_variance(data):
    # Calculate the mean
    mean = sum(data) / len(data)
    
    # Calculate the variance
    # numerator =  sum the squared deviations of each data point from the mean.
    # denominator = N
    variance = sum((x - mean) ** 2 for x in data) / len(data) 
    
    return variance

population_variance(Days) # same as np.var(Days)

def sample_variance(data, ddof):
    # Calculate the mean
    mean = sum(data) / len(data)
    
    # Calculate the sample variance
    # numerator =  sum the squared deviations of each data point from the mean.
    # denominator = N - Degrees of Freedom (typically N-1 or Bessel's Correction)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - ddof)
    
    return variance

sample_variance(Days, 1) # same as np.var(Days, ddof=1) and var(Days) in R

np.std(Days) # population standard deviation
np.std(Days, ddof=1) # sample standard deviation, which is equivalent to sd(Days) in R

np.sqrt(np.var(Days)) # population standard deviation
np.sqrt(np.var(Days, ddof=1)) # sample standard deviation, which is equivalent to sd(Days) in R

stemgraphic.stem_graphic(Days, scale=1, leaf_order=True) # close to stem(Days) in R
# scale option controls the level of detail by adjusting the number of stems
# higher values of scale increases the level of detail by dividing each stem into multiple parts, increasing the plot resolution
# lower values of scale decreases the level of detail by grouping the data into fewer stems
# for example, if you set scale = 2, each stem is split in half, create two sub-stems per stem
plt.close()
plt.show()

# when leaf_order is set to True, the leaves within each stem are ordered from smallest to largest, this is default behavior in R

# stem(days)
# The decimal point is 1 digit(s) to the right of the |
# 
# 1 | 677 # smallest = 16.67
# 2 | 22234
# 2 | 66778
# 3 | 00012334
# 3 | 555666677
# 4 | 1233
# 4 | 56678
# 5 | 
# 5 | 6 # largest = 56
plt.close()
stemgraphic.stem_graphic(Days, scale=10, leaf_order=True)
plt.show()

# 40 | 5 | 6 # largest
# 39 | 4 | 123356678 
# ...
# 3 | 1 | 677 # smallest

# Key: aggr|stem|leaf
# aggr = number of data points (leaves) within each stem grouping
# for example, if aggr = 3, then there are three values within that specific steam-leaf row
# Note this row: 40 | 5 | 6
# and this row: 39 | 4 | 123356678 
# There is only 1 differnce between aggr in these rows, so only 1 number equal to 56

# stem is the leading part of each data value, typically representing higher place values (e.g., tens, hundreds).
# For example, in this row: 40 | 5 | 6 the stem would be 5 (representing the tens place)

# leaf is the trailing part of each data value that shows the lower place values (usually the units).
# For example, in this row: 39 | 4 | 123356678 the leaf would be 123356678 (representing the units place)
# 41, 42, 43, 43, 45, 46, 46, 47, 48

plt.close()
plt.scatter(Claim, Days) # equivalent to plot(y=Days, x=Claim) in R

# Add labels
plt.xlabel('Claim')
plt.ylabel('Days')

# Show the plot

plt.show()

plt.close()
plt.plot(Claim, Days) # equivalent to plot(y=Days, x=Claim, type='l') in R

# Add labels
plt.xlabel('Claim')
plt.ylabel('Days')

# Show the plot

plt.show()

plt.close()
plt.plot(Claim, Days, marker='o', linestyle='--') # equivalent to plot(Days~Claim, type='b') in R

# Add labels
plt.xlabel('Claim')
plt.ylabel('Days')

# Show the plot
plt.show()

Thickness = [438,413,444,468,445,472,474,454,455,449,
            450,450,450,459,466,470,457,441,450,445,
            487,430,446,450,456,433,455,459,423,455,
            451,437,444,453,434,454,448,435,432,441,
            452,465,466,473,471,464,478,446,459,464,
            441,444,458,454,437,443,465,435,444,457,
            444,471,471,458,459,449,462,460,445,437,
            461,453,452,438,445,435,454,428,454,434,
            432,431,455,447,454,435,425,449,449,452,
            471,458,445,463,423,451,440,442,441,439]
plt.close()
plt.hist(Thickness, bins='auto') # equivalent to hist(Thickness) in R
plt.show()

plt.close()
plt.hist(Thickness, bins=15) # equivalent to hist(Thickness, breaks=15) in R
plt.show()

# Add labels
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()

Diameter = [120.5,120.9,120.3,121.3,
           120.4,120.2,120.1,120.5,
           120.7,121.1,120.9,120.8]
plt.close()
plt.boxplot(Diameter, vert=False) # equivalent to boxplot(Diameter, horizontal=TRUE) in R
plt.show()

###CONTENT VIDEOS -- Probability Distributions (WEEK 4)###

# Hypergeometric distribution
# Let's calculate the probabilities of drawing a certain number of "successes" (in this case, hearts) in a hypergeometric distribution, given a deck of cards. 

N=52  # cards in deck
D=13  # Hearts in deck
n=5   # cards drawn

# possible number of hearts drawn
x = range(0, min(n, D) + 1)

# calculate probabilities
probs = [comb(D, k) * comb(N - D, n - k) / comb(N, n) for k in x]

# Display results as a pandas DataFrame (similar to cbind in R)
results = pd.DataFrame({'x': x, 'probs': probs})
print(results)

# In R, dhyper(x, m=D, n=N-D, k=n) to calculate the probability of drawing exactly x hearts in the sample of n cards, where:
# m=D: Number of "successes" (hearts) in the population.
# n=N-D: Number of "failures" (non-hearts) in the population.
# k=n: Total number of draws (sample size)

# Calculate hypergeometric probabilities

# Probability Mass Function (PMF) is a function that gives the probability of each possible outcome for a discrete random variable. 
# In other words, it maps each value that the random variable can take to its corresponding probability.

probs = [hypergeom.pmf(k, N, D, n) for k in x]

# Display results as a DataFrame for readability
results = pd.DataFrame({'x': x, 'probs': probs})
print(results)

#    x     probs
# 0  0  0.221534 #probability that you draw 0 hearts in a 5-card hand
# 1  1  0.411420
# 2  2  0.274280
# 3  3  0.081543
# 4  4  0.010729
# 5  5  0.000495 #probability that you draw 5 hearts in a 5-card hand

# hypergeom.pmf(k, N, D, n): This is the equivalent of dhyper() in R.
# k: The number of "successes" (hearts) you want to draw (i.e., each value in x).
# N: The population size (total number of cards).
# D: The number of "successes" in the population (hearts).
# n: The sample size (number of cards drawn).

# P(at least 3 Hearts) =
np.sum(probs[3:6])

# P(fewer than 2 Kings) =
D=4 # Kings in deck
x=range(0, min(n, D) + 1)

probs = [hypergeom.pmf(k, N, D, n) for k in x]
np.sum(probs[0:2])

# Binomial distribution
# Parameters
n = 150  # Number of trials
p = 0.90 # Probability of success

# Range of possible successes
x = range(0, n + 1)

# Calculate binomial probabilities 
probs = [binom.pmf(k, n, p) for k in x] # equivalent to dbinom() in R

# Display results as a DataFrame
results = pd.DataFrame({'x': x, 'probs': probs})
print(results)

# Calculate binomial probabilities (hard coded)
probs = [comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in x]

# Display results as a DataFrame
results = pd.DataFrame({'x': x, 'probs': probs})
print(results)

# In R, this function: plot(probs~x, type='h')
# produces vertical lines for each (x, probs) pair

# Create a vertical line plot
plt.close()
plt.stem(x, probs)

# Add labels and title
plt.xlabel("x")
plt.ylabel("probs")

# Display the plot
plt.show()

#P(between 3 and 6, inclusive, successes) =
np.sum(probs[3:7])

# Poisson distribution

# Models the count of events within a fixed interval, assuming events happen at a constant 
# rate and independently of each other.

# Parameters
n = 1000  # The number of occurrences 
x=range(0, n + 1)
lambda_val = 4 # The rate parameter (average occurrences)

# Calculate Poisson probability
probs = [poisson.pmf(k, lambda_val) for k in x] # equivalent to dbinom() in R
plt.close()
plt.stem(x, probs)
plt.show()

# P(2 or fewer defects) =
np.sum(probs[0:3])

# P(more than 10 defects) =
1 - np.sum(probs[0:11])

# Negative Binomial

# Models the number of failures before achieving a specified number of successes. 
# It's a generalization of the geometric distribution.

n = 1000
x = range(0, n + 1)
r = 1
prob = 1/6

probs = [nbinom.pmf(k, r, prob) for k in x] # equivalent to dbinom() in R

# P(2 or fewer defects) =
np.sum(probs[0:3])

# Geometric distributions

# A special case of the negative binomial where only one success is required, modeling the number of 
# failures before the first success.

n = 1000
x = range(0, n + 1)
prob = 1/6

probs = [geom.pmf(k + 1, prob) for k in x] # equivalent to dgeom() in R

# P(2 or fewer defects) =
np.sum(probs[0:3])

# Normal distribution

mu = 40       # Mean of the distribution
sigma = 2     # Standard deviation of the distribution

# P(x <= 35) =
norm.cdf(35, loc=mu, scale=sigma) # cumulative probabilities F(x)

# P(x > 41) =
1 - norm.cdf(41, loc=mu, scale=sigma)

mu = 10       # Mean of the distribution
sigma = 3     # Standard deviation of the distribution

norm.ppf(.95, loc=mu, scale=sigma) # inverse cumulative probabilities F(x)

# norm.ppf(p, mean, sd) returns the quantile (or z-score) for which the cumulative probability up to that quantile is p for a normal distribution with specified mean and sd.
# In this example, you would have to have a z-score of 14.93 to have a cumulative probablity of .95 (mean = 10 and sd = 3)

# QQ-Line Plot
octane=[88.9, 87.0, 90.0, 88.2, 87.2, 87.4, 87.8, 89.7, 86.0, 89.6]

# Generate the Q-Q plot data
plt.close()
fig, ax = plt.subplots()
res = probplot(octane, dist="norm")

# Plot the data on the y-axis and theoretical quantiles on the x-axis
ax.scatter(res[0][0], res[0][1], label="Data Points")  # Scatter plot for octane data
ax.plot(res[0][0], res[1][1] + res[1][0] * res[0][0], color="red", label="Q-Q Line")  # Reference line

# Set labels and title
ax.set_title("Q-Q Plot with Data on X-axis")
ax.set_ylabel("Sample Quantiles (Octane Data)")
ax.set_xlabel("Theoretical Quantiles (Normal)")

# Show legend and plot
ax.legend()
plt.show()

# Central limit theorem with dice

# Parameters
n = 10000  # Number of samples (dice rolls)

# Declare random seed
np.random.seed(42)

# 1. Single die roll
single_roll = np.random.randint(1, 7, n)
single_probs, single_counts = np.unique(single_roll, return_counts=True)
single_probs = single_counts / n
plt.close()
plt.stem(single_probs)
plt.show()

# 2. Average of two rolls
two_rolls = (np.random.randint(1, 7, n) + np.random.randint(1, 7, n)) / 2
two_probs, two_counts = np.unique(two_rolls, return_counts=True)
two_probs = two_counts / n
plt.close()
plt.stem(two_probs)
plt.show()

# 3. Average of four rolls
four_rolls = np.mean(np.random.randint(1, 7, (4, n)), axis=0)
four_probs, four_counts = np.unique(four_rolls, return_counts=True)
four_probs = four_counts / n
plt.close()
plt.stem(four_probs)
plt.show()

# 4. Average of twelve rolls
twelve_rolls = np.mean(np.random.randint(1, 7, (12, n)), axis=0)
twelve_probs, twelve_counts = np.unique(twelve_rolls, return_counts=True)
twelve_probs = twelve_counts / n
plt.close()
plt.stem(twelve_probs)
plt.show()

# Lognormal distribution

# Parameters
meanlog = 6     # Mean of the log (log of the mean)...called theta=6 in R code
sdlog = 1.2     # Standard deviation of the log ...called omega=1.2 in R code

# Calculate shape and scale parameters
shape = sdlog           # Shape parameter in scipy, equivalent to sdlog in R
scale = np.exp(meanlog) # Scale parameter is exp(meanlog)

# P(x>500) =
1-lognorm.cdf(500, s=shape, scale=scale)

# Exponential distribution

# Parameters
rate = 10 ** -4    # This is equivalen to lambda=10^-4 in R
scale = 1 / rate   # mean

# P(x < 10000) =
expon.cdf(10000, scale=scale)

# P(x > 10000) =
1-expon.cdf(10000, scale=scale)

# Gamma distribution

# Parameters
shape = 2           # Shape parameter...r=2 in the R code
rate = 10 ** -4     # Rate parameter (inverse of scale)...defined as lambda in the R code...lambda=10^-4

# Convert rate to scale
scale = 1 / rate

# P(x > 10000) =
1 - gamma.cdf(10000, a=shape, scale=scale)

# Weibull distribution

# Parameters
shape = 1/2    # Shape parameter...beta=1/2 in the R code
scale = 5000     # Scale parameter...theta=5000 in the R code

# P(x > 10000)
1 - weibull_min.cdf(10000, c=shape, scale=scale)

