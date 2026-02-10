import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
import pandas as pd
import geopandas as gpd
import geobr
from dbfread import DBF
from epiweeks import Week, Year
from scipy.optimize import least_squares
from scipy.stats import t, lognorm, norm, uniform
from scipy.optimize import fsolve, root

# Municipalities of interest and their codes ##################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Screening estimates of susceptibility and sd ################################
list_mean = pd.read_csv('res/mean_si_muns.csv', header=None, sep = ',') 
list_sd = pd.read_csv('res/sd_si_muns.csv', header=None, sep = ',')

# Cases and population totals #################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None, sep = ',') 
list_pop = pd.read_csv('res/pop_muns.csv', header=None, sep = ',') 

i = 0

fig, axes = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(15, 20),
    sharex = True
)

axes = axes.flatten()

for cod_mun in cod_muns:
    
    
    pop_age = np.array(list_pop.loc[i])
    cases_age = np.array(list_cases.loc[i])
    sus_age = np.array(list_mean.loc[i])
    sus_age = np.append(sus_age, 0.99)

    
    age_groups = [
        "<1", "1–4", "5–9", "10–14", "15–20",
        "20–29", "30–39", "40–49", "50–59", "60+"
    ]
    means = 1 - sus_age
    stds  = np.array(list_sd.loc[i])
    stds = np.append(stds, 0)
    
    inc_age_list = np.loadtxt('res/sus_list_'+str(names_muns[i])+'.csv',  delimiter=",")
    
    ax = axes[i]
    
    sns.boxplot(
       inc_age_list / pop_age,
       ax=ax,
       color="lightgray",
       linewidth=1.2,
       fliersize=2
     )

    ax.errorbar(
       np.arange(0, 9),
       means[:-1],
       yerr=stds[:-1],
       fmt='o',
       color='k',
       linewidth=2,
       capsize=4
       )

    ax.set_title(names_muns[i], fontsize=12)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(age_groups, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    if i % 2 == 0:
        ax.set_ylabel("Susceptibility")
    else:
        ax.set_ylabel("")

    if i < 8:
       ax.set_xlabel("")
    else:
       ax.set_xlabel("Age group")
    
    
    i = i + 1


plt.tight_layout()
plt.savefig('figs/sus_comparison.pdf')
plt.savefig('figs/sus_comparison.jpg')
plt.show()
