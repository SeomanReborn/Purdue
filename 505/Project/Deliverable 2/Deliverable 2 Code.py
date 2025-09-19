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

exhaust = pd.read_excel(r'C:\Users\manam\Documents\GitHub\Purdue\505\Project\10_1_Class 8 Fleet Data.xlsx')
fuelecon = pd.read_excel(r'C:\Users\manam\Documents\GitHub\Purdue\505\Project\10_2_Class 8 Fleet Fuel Economy.xlsx')
#exhaust starts on row 1
exhaust = exhaust.iloc[1:].reset_index(drop=True)


#rename Truck Col
exhaust.rename(columns={'Unnamed: 0': 'Truck'}, inplace=True)


month = 'Month'
mpg = 'Average Fuel Economy (mpg)'
fuelTypeEcon = 'Fleet'
season = 'Season'

truck = 'Truck'
sample = 'Samp. #'
fuelPct = 'Fuel'
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
fuelTypeExhaust = 'Fuel'

#fuel types
ulsd = '#2ULSD'
b20 = 'B20'

ulsdMpg = fuelecon.loc[fuelecon[fuelTypeEcon]==ulsd]
b20Mpg = fuelecon.loc[fuelecon[fuelTypeEcon]==b20]
plt.close()
plt.plot(ulsdMpg[month], ulsdMpg[mpg], marker='o', linestyle='-', color='blue', label=ulsd)
plt.plot(b20Mpg[month], b20Mpg[mpg], marker='s', linestyle='--', color='red', label=b20)

plt.xlabel('Month')
plt.ylabel('Average MPG')
plt.title('#2ULSD vs B20 Average Miles per Gallon')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

ulsdMpg.describe()
b20Mpg.describe()

month_to_season = {
    'Dec': 'Winter', 'Jan': 'Winter', 'Feb': 'Winter',
    'Mar': 'Spring', 'Apr': 'Spring', 'May': 'Spring',
    'June': 'Summer', 'July': 'Summer', 'Aug': 'Summer',
    'Sept': 'Fall', 'Oct': 'Fall', 'Nov': 'Fall'
}

ulsdMpg.loc[:,'Season'] = ulsdMpg[month].map(month_to_season)
b20Mpg.loc[:,'Season'] = b20Mpg[month].map(month_to_season)

# Spring
ulsdMpgSpring = ulsdMpg.loc[ulsdMpg[season] == 'Spring']
b20MpgSpring = b20Mpg.loc[b20Mpg[season] == 'Spring']

# Summer
ulsdMpgSummer = ulsdMpg.loc[ulsdMpg[season] == 'Summer']
b20MpgSummer = b20Mpg.loc[b20Mpg[season] == 'Summer']

# Fall
ulsdMpgFall = ulsdMpg.loc[ulsdMpg[season] == 'Fall']
b20MpgFall = b20Mpg.loc[b20Mpg[season] == 'Fall']

# Winter
ulsdMpgWinter = ulsdMpg.loc[ulsdMpg[season] == 'Winter']
b20MpgWinter = b20Mpg.loc[b20Mpg[season] == 'Winter']

ulsdMpg.loc[ulsdMpg[season] == 'Spring'].describe()
b20Mpg.loc[b20Mpg[season] == 'Spring'].describe()

ulsdMpg.loc[ulsdMpg[season] == 'Summer'].describe()
b20Mpg.loc[b20Mpg[season] == 'Summer'].describe()

ulsdMpg.loc[ulsdMpg[season] == 'Fall'].describe()
b20Mpg.loc[b20Mpg[season] == 'Fall'].describe()

ulsdMpg.loc[ulsdMpg[season] == 'Winter'].describe()
b20Mpg.loc[b20Mpg[season] == 'Winter'].describe()

plt.close()
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
fig.suptitle('Seasonal Fuel Economy Comparison: ULSD vs B20', fontsize=16)


# --- SPRING ---
axs[0, 0].plot(ulsdMpgSpring[month], ulsdMpgSpring[mpg], label='ULSD', linestyle='-', marker='o', color='blue')
axs[0, 0].plot(b20MpgSpring[month], b20MpgSpring[mpg], label='B20', linestyle='--', marker='s', color='red')
axs[0, 0].set_title('Spring')
axs[0, 0].set_ylabel('MPG')
axs[0, 0].legend()
axs[0, 0].grid(True)

# --- SUMMER ---
axs[0, 1].plot(ulsdMpgSummer[month], ulsdMpgSummer[mpg], label='ULSD', linestyle='-', marker='o', color='blue')
axs[0, 1].plot(b20MpgSummer[month], b20MpgSummer[mpg], label='B20', linestyle='--', marker='s', color='red')
axs[0, 1].set_title('Summer')
axs[0, 1].legend()
axs[0, 1].grid(True)

# --- FALL ---
axs[1, 0].plot(ulsdMpgFall[month], ulsdMpgFall[mpg], label='ULSD', linestyle='-', marker='o', color='blue')
axs[1, 0].plot(b20MpgFall[month], b20MpgFall[mpg], label='B20', linestyle='--', marker='s', color='red')
axs[1, 0].set_title('Fall')
axs[1, 0].set_xlabel('Month')
axs[1, 0].set_ylabel('MPG')
axs[1, 0].legend()
axs[1, 0].grid(True)

# --- WINTER ---
axs[1, 1].plot(ulsdMpgWinter[month], ulsdMpgWinter[mpg], label='ULSD', linestyle='-', marker='o', color='blue')
axs[1, 1].plot(b20MpgWinter[month], b20MpgWinter[mpg], label='B20', linestyle='--', marker='s', color='red')
axs[1, 1].set_title('Winter')
axs[1, 1].set_xlabel('Month')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



