import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # For lm() equivalent (see line 322)
import pandas as pd
from statsmodels.stats.anova import anova_lm # For anova() equivalent (see line 330)
from statsmodels.formula.api import ols # For lm() equivalent (see line 339)
from scipy.stats import hypergeom
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import chi2 # For pchisq() equivalent (see line 111)
from scipy.stats import f # For pf() equivalent (see line 129)
from scipy.stats import t # For pt() equivalent (see line 147)
from scipy.stats import ttest_1samp # For one-sample t.test() equivalent (see line 157)
from scipy.stats import ttest_ind # For two-sample t.test() equivalent (see line 189)
from scipy.stats import ttest_rel # For paired two-sample t.test() equivalent (see line 199)
from scipy.stats import binomtest # For one-sample prop.test() equivalent (see line 265)
from statsmodels.stats.proportion import proportions_ztest # For two-sample prop.test() equivalent (see line 274)

###CONTENT VIDEOS -- Probability Plots and Approximations (WEEK 5)###

# Binomial Approximation to Hypergeometric

# Hypergeometric
N = 200
D = 5
n = 10
x = range(0, min(n, D) + 1)

probs = [hypergeom.pmf(k, N, D, n) for k in x]
rounded_probs = [round(num, 4) for num in probs]

results = pd.DataFrame({'x': x, 'probs': rounded_probs})
print(results)

# Binomial
p = D/N
probs = [binom.pmf(k, n, p) for k in x] 
rounded_probs = [round(num, 4) for num in probs]

results = pd.DataFrame({'x': x, 'probs': rounded_probs})
print(results)

# Poisson Approximation to Binomial
n = 100
p = 0.025
x = range(0, n + 1)

probs = [binom.pmf(k, n, p) for k in x]
rounded_probs = [round(num, 4) for num in probs]

results = pd.DataFrame({'x': x, 'probs': rounded_probs})
results.head()

lambda_val = n*p
probs = [poisson.pmf(k, lambda_val) for k in x]
rounded_probs = [round(num, 4) for num in probs]

results = pd.DataFrame({'x': x, 'probs': rounded_probs})
results.head()

# Normal Approximation to Binomial
n = 40
p = 0.5
mu = n*p
sigma=np.sqrt(n*p*(1-p))

norm.cdf(18.5, loc=mu, scale=sigma)-norm.cdf(17.5, loc=mu, scale=sigma) 
binom.pmf(18, n, p)

1-norm.cdf(24.5, loc=mu, scale=sigma)
np.sum([binom.pmf(k, n, p) for k in range(25, n+1)])

# Create a vertical line plot
x_values = list(range(0, n+1))
y_values = [binom.pmf(k, n, p) for k in x_values]
plt.stem(x_values, y_values)
line_values = [norm.pdf(k, loc=mu, scale=sigma) for k in x_values]

plt.plot(x_values, line_values , color='purple')
plt.show()

###CONTENT VIDEOS -- SAMPLING DISTRIBUTIONS (WEEK 5)

# Declare random seed
np.random.seed(1)

# Parameters
lambda_val = 4  # Mean (rate parameter) of the Poisson distribution

# Generate random numbers from Poisson distribution
random_numbers = poisson.rvs(mu=lambda_val, size=10000*5)
samples = np.array(random_numbers).reshape(10000, 5)

samples[:5,] # first five rows

means = np.mean(samples, axis=1) #rowMeans()
means[0:5] # first five
np.mean(means)

vars = np.var(samples, axis=1) #rowMeans()
vars[0:5] # first five
np.mean(vars)

# Chi-Square distribution

# Parameters
q = 2     # Quantile value
df = 5    # Degrees of freedom

# Calculate the cumulative probability
chi2.cdf(q, df)

# t distribution

# Parameters
q = 2     # Quantile value
df = 8    # Degrees of freedom

# Calculate the cumulative probability
t.cdf(q, df)

# F distribution

# Parameters
q = 2            # Quantile value
df1 = 2          # Degrees of freedom for the numerator
df2 = 10         # Degrees of freedom for the denominator

f.cdf(q, df1, df2)

###CONTENT VIDEOS -- CONFIDENCE INTERVALS (WEEK 5)

# Confidence Intervals

# One sample t-based CI for mu
Viscosity = [3193,3124,3153,3145,3093,
             3466,3355,2979,3182,3227,
             3256,3332,3204,3282,3170]

# Calculate necessary statistics
mean_viscosity = np.mean(Viscosity)
std_viscosity = np.std(Viscosity, ddof=1)  # Sample standard deviation (ddof=1 for unbiased estimate)
n = len(Viscosity)
df = n - 1  # Degrees of freedom

# Calculate t critical value for 95% confidence interval
t_critical = t.ppf(0.975, df)

# Calculate margin of error
margin_of_error = t_critical * std_viscosity / np.sqrt(n)

# Calculate confidence interval
confidence_interval = (mean_viscosity - margin_of_error, mean_viscosity + margin_of_error)
print(confidence_interval)

# Perform the one-sample t-test
t_stat, p_value = ttest_1samp(Viscosity, popmean=0)
print("t-statistic:", t_stat)
print("p-value:", p_value / 2)  # One-sided p-value for 'greater'

# Calculate necessary statistics
mean_viscosity = np.mean(Viscosity)
std_viscosity = np.std(Viscosity, ddof=1)  # Sample standard deviation
n = len(Viscosity)
df = n - 1  # Degrees of freedom

# Set the confidence level and alpha for a one-sided test
confidence_level = 0.90
alpha = 1 - confidence_level

# Calculate the critical t-value for a one-sided test
t_critical = t.ppf(1 - alpha, df)

# Calculate the margin of error for the one-sided confidence interval
margin_of_error = t_critical * (std_viscosity / np.sqrt(n))

# Calculate the one-sided confidence interval
# Since alternative='greater', we only need the lower bound
confidence_interval = (mean_viscosity - margin_of_error, np.inf)
print("90% One-Sided Confidence Interval (greater):", confidence_interval)

# Two-Sample Independent t-test

# Sample data for two groups
data1 = np.array([5, 7, 8, 9, 10])
data2 = np.array([6, 8, 9, 10, 12])

# Perform independent two-sample t-test
result = ttest_ind(data1, data2)
print("Two-sample independent t-test results:", result)

# Two-Sample Paired t-test

# Sample data for two groups
data1 = np.array([5, 7, 8, 9, 10])
data2 = np.array([6, 8, 9, 10, 12])

# Perform paired two-sample t-test
result = ttest_rel(data1, data2)
print("Paired t-test results:", result)

# One sample chi-squared-based CI for variance

# Calculate sample variance
sample_variance = np.var(Viscosity, ddof=1)  # ddof=1 for sample variance
n = len(Viscosity)
df = n - 1  # Degrees of freedom

# Confidence level
confidence_level = 0.95
alpha = 1 - confidence_level

# Calculate chi-squared critical values
chi2_lower = chi2.ppf(alpha / 2, df)
chi2_upper = chi2.ppf(1 - alpha / 2, df)

# Calculate the confidence interval for the variance
variance_ci_lower = (df * sample_variance) / chi2_upper
variance_ci_upper = (df * sample_variance) / chi2_lower
variance_confidence_interval = (variance_ci_lower, variance_ci_upper)

print("Sample Variance:", sample_variance)
print("95% Confidence Interval for Variance:", variance_confidence_interval)

# Two sample F-based CI for ratio of variances

import numpy as np
from scipy.stats import f

# Sample data
data1 = np.array([5, 7, 8, 9, 10])
data2 = np.array([6, 8, 9, 10, 12])

# Calculate sample variances and sizes
s1_squared = np.var(data1, ddof=1)  # Sample variance for data1
s2_squared = np.var(data2, ddof=1)  # Sample variance for data2
n1 = len(data1)
n2 = len(data2)
df1 = n1 - 1  # Degrees of freedom for data1
df2 = n2 - 1  # Degrees of freedom for data2

# Confidence level
confidence_level = 0.95
alpha = 1 - confidence_level

# Calculate F critical values
f_lower = f.ppf(alpha / 2, df1, df2)
f_upper = f.ppf(1 - alpha / 2, df1, df2)

# Calculate the confidence interval for the ratio of variances
ratio_variances = s1_squared / s2_squared
ci_lower = ratio_variances / f_upper
ci_upper = ratio_variances / f_lower
confidence_interval = (ci_lower, ci_upper)

print("Sample Variance Ratio:", ratio_variances)
print(f"{int(confidence_level*100)}% Confidence Interval for Ratio of Variances:", confidence_interval)

# One sample z-based CI for proportion

x = 15
n = 80
p_null = 0.5  

binomtest(x, n, p_null, alternative='two-sided') # returns observed proportion as the statistic

# Two sample z-based CI for difference in props

# Parameters
successes = np.array([30, 45])  # Number of successes in each sample
n = np.array([50, 60])          # Total number of trials in each sample

# Perform two-sample z-test for proportions
stat, p_value = proportions_ztest(successes, n, alternative='two-sided')
print("Two-sample proportion test p-value:", p_value)

###CONTENT VIDEOS -- HYPOTHESIS TESTS (WEEK 5)

ttest_1samp(Viscosity, popmean=3200)
ttest_1samp(Viscosity, popmean=3200, alternative='greater')

#varTest()...no equivalent

# prop_test()

x = 15
n = 80
p_null = 0.5  

binomtest(x, n, p_null, alternative='two-sided') # returns observed proportion as the statistic

###CONTENT VIDEOS -- ANOVA (WEEK 5)
# Linear Regression

# Table 4.4
Concentration=np.sort(np.repeat([5, 10, 15, 20], 6))
Strength=[7,8,15,11,9,10,
          12,17,13,18,19,15,
          14,18,19,17,16,18,
          19,25,22,23,18,20]

# Create DataFrame
df = pd.DataFrame({'Concentration': Concentration, 'Strength': Strength})

# Convert Concentration to Category and Create Dummies
df = pd.get_dummies(df, columns=['Concentration'], drop_first=True) # Drops one category to avoid multicollinearity (similar to R’s default behavior with factors in linear regression).
print(df.head())

# Convert Dummy Variables to 1 and 0
df['Concentration_10'] = df['Concentration_10'].astype(int)
df['Concentration_15'] = df['Concentration_15'].astype(int)
df['Concentration_20'] = df['Concentration_20'].astype(int)

# Define the response variable and predictors
X = df.drop(columns='Strength')  # Predictor variables
y = df['Strength']               # Response variable

# Add a constant to the predictors (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary (similar to R's summary(lm(...)))
print(results.summary())

# Perform ANOVA
model = ols('Strength ~ C(Concentration)', data=df).fit() 
anova_results = anova_lm(model)
print(anova_results)

# Table 4.10 / Example 4.13
Applications=[80,93,100,82,90,99,81,96,
              94,93,97,95,100,85,86,87]
Loans=[8,9,10,12,11,8,8,10,
       12,11,13,11,8,12,9,12]
Cost=[2256,2340,2426,2293,2330,2368,2250,2409,
      2364,2379,2440,2364,2404,2317,2309,2328]

# Create DataFrame
df = pd.DataFrame({'Applications': Applications, 'Loans': Loans, 'Cost': Cost})

# Define the response variable and predictors
X = df.drop(columns='Cost')  # Predictor variables
y = df['Cost']               # Response variable

# Add a constant to the predictors (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary (similar to R's summary(lm(...)))
print(results.summary())