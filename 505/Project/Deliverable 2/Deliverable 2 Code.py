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

#rename Truck Col
exhaust.rename(columns={'Unnamed: 0': 'Truck'}, inplace=True)


month = 'Month'
mpg = 'Average Fuel Economy (mpg)'
fuelTypeEcon = 'Fleet'

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
ulsd = '#2ULSD'
b20 = 'B20'

plt.plot(fuelecon[month], fuelecon[mpg], marker='o', linestyle='-', color='blue', label='Sample Line')
