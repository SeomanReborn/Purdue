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

#Problem 1.A
temp = [127, 125, 131, 124, 129, 121, 142, 151, 160, 125, 124, 123,
        120, 119, 128, 133, 137, 124, 142, 123, 121, 136, 140, 137,
        125, 124, 128, 129, 130, 122, 118, 131, 125, 133, 141, 125, 140, 131, 129, 126]
temp.sort()
np.median(temp)
#128

#Problem 1.B
#Any amount. It will not change the median

#Problem 1.C
np.mean(temp)
#Mean: 129.975
np.std(temp)
#Std Dev:8.8019

#Problem 1.D
plt.close()
plt.hist(temp, bins=15)
plt.xlabel('Temp')
plt.ylabel('Amount')
plt.show()

#Problem 1.E
plt.close()
stemgraphic.stem_graphic(temp, scale=10, leaf_order=True)
plt.show()

#Problem 1.F
tempQuarts = np.quantile(temp, [.25, .75])
print(tempQuarts)
#124. 133.75

#Problem 1.G
plt.close()
plt.boxplot(temp)
plt.ylabel('Temp')
plt.show()

#Problem 1.H
#Right Leaning. Has 2 outliers.

#Problem 1.I
plt.close()
fig, ax = plt.subplots()
res = probplot(temp, dist="norm")
ax.scatter(res[0][0], res[0][1], label="Data Points")  # Scatter plot for octane data
ax.plot(res[0][0], res[1][1] + res[1][0] * res[0][0], color="red", label="Q-Q Line")  # Reference line
ax.set_title("Q-Q Plot with Data on X-axis")
ax.set_ylabel("Sample Quantiles (Temp Data)")
ax.set_xlabel("Theoretical Quantiles (Normal)")
ax.legend()
plt.show()
#Yes. The data that isn't an outlier lay on the line for the most part.

#Problem 2.A
n = 3
p = .15
prob = geom.pmf(k=n, p=p)
print(prob)
#.108375

#Problem 2.B
#1/p or 6.6667 patients

#Problem 2.C
n = 50
k = 10
p = .15
prob = binom.pmf(k,n,p)
print(prob)
#.088989

