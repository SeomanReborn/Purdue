import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm # For lm() equivalent (see line 322)
import pandas as pd
import seaborn as sns
from statsmodels.stats.anova import anova_lm # For anova() equivalent (see line 330)
from statsmodels.formula.api import ols # For lm() equivalent (see line 339)
import scipy.stats as stats
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

exhaust = pd.read_excel(r'C:\Users\manam\Documents\GitHub\Purdue\505\Project\10_1_Class 8 Fleet Data.xlsx')
fuelecon = pd.read_excel(r'C:\Users\manam\Documents\GitHub\Purdue\505\Project\10_2_Class 8 Fleet Fuel Economy.xlsx')
#exhaust starts on row 1
exhaust = exhaust.iloc[1:].reset_index(drop=True)


#rename Truck Col
exhaust.rename(columns={'Unnamed: 0': 'Truck'}, inplace=True)
exhaust.rename(columns={'Fuel': 'Fuel %'}, inplace=True)
exhaust.rename(columns={'Fuel.1': 'Fuel Type'}, inplace=True)


month = 'Month'
mpg = 'Average Fuel Economy (mpg)'
fuelTypeEcon = 'Fleet'
season = 'Season'

truck = 'Truck'
sample = 'Samp. #'
fuelPct = 'Fuel %'
sootPct = 'Soot'
visc = 'Visc'
acidNum = 'Acid #'
baseNum = 'Base #'
oxid = 'Oxidation'
nitration = 'Nitration'
ironPpm = 'Iron'
leadPpm = 'Lead'
copperPpm = 'Copper'
chromPpm = 'Chro'
alumPpm = 'Aluminum'
siliconPpm = 'Silicon'
sodium_ppm = 'Sodium'
potasPpm = 'Potassium'
fuelTypeExhaust = 'Fuel Type'

#fuel types
ulsd = '#2ULSD'
b20 = 'B20'

ulsdMpg = fuelecon.loc[fuelecon[fuelTypeEcon]==ulsd].reset_index(drop=True)
b20Mpg = fuelecon.loc[fuelecon[fuelTypeEcon]==b20].reset_index(drop=True)


result = ttest_ind(ulsdMpg[mpg], b20Mpg[mpg])
print("Paired t-test results:", result)

diff = ulsdMpg[mpg].reset_index(drop=True) - b20Mpg[mpg].reset_index(drop=True)
plt.close()
stats.probplot(diff, dist="norm", plot=plt)
plt.title("Q-Q Plot of AMPG")
plt.xlabel("Quants")
plt.ylabel("Sample Quants")
plt.show()

data = [ulsdMpg[mpg], b20Mpg[mpg]]
plt.close()
plt.boxplot(data, labels=['#2ULSD', 'B20'])
plt.title("Boxplot #2ULSD and B20")
plt.ylabel("Values")
plt.xlabel("Biofuels")
plt.show()

ulsdExhaust = exhaust.loc[exhaust[fuelTypeExhaust]==ulsd]
b20Exhaust = exhaust.loc[exhaust[fuelTypeExhaust]==b20]

ulsdColPpmSum = ulsdExhaust.iloc[:, 9:17].sum()
b20ColPpmSum = b20Exhaust.iloc[:, 9:17].sum()

colTest = exhaust.columns[9:17]

for col in colTest:
  ulsd = pd.to_numeric(ulsdExhaust[col], errors='coerce').dropna()
  b20 = pd.to_numeric(b20Exhaust[col], errors='coerce').dropna()

  t_stat, p_val = ttest_ind(ulsd, b20, equal_var=False)
  print(f"Column: {col}")
  print(f"  t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")


columns = ['Iron', 'Lead', 'Copper', 'Chro', 'Aluminum', 'Silicon', 'Sodium', 'Potassium']

# Create 3x3 subplot grid
fig, axes = plt.subplots(3, 3, figsize=(9, 6))
axes = axes.flatten()

# Loop through each column and plot Q-Q of differences
for i, col in enumerate(columns):
    ulsd = pd.to_numeric(ulsdExhaust[col], errors='coerce').reset_index(drop=True)
    b20 = pd.to_numeric(b20Exhaust[col], errors='coerce').reset_index(drop=True)
    
    # Ensure same length
    min_len = min(len(ulsd), len(b20))
    diff = ulsd[:min_len] - b20[:min_len]
    diff = diff.dropna()

    stats.probplot(diff, dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q of Diff: {col}")
    axes[i].set_xlabel("Theoretical Quants")
    axes[i].set_ylabel("Quants")

# Hide unused subplot if fewer than 9
if len(columns) < 9:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()



