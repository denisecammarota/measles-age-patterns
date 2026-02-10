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
np.random.seed(123)

# Municipalities under study and their codes ##################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Loading data on cases #######################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None, sep = ',') 
list_cases = list_cases.to_numpy()

# Loading data on population ##################################################
list_pop = pd.read_csv('res/pop_muns.csv', header=None, sep = ',') 
list_pop = list_pop.to_numpy()

# Initializing plot ###########################################################
fig, axes = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(15, 20),
    sharex = True
)

axes = axes.flatten()

age_groups = [
     "<1", "1–4", "5–9", "10–14", "15–20",
     "20–29", "30–39", "40–49", "50–59", "60+"
]

# Looping and plotting again ##################################################
i = 0
for cod_mun in cod_muns:
    
    print(names_muns[i])
    
    inc_100_mc_list = np.loadtxt('res/inc_100_mc_list_'+str(names_muns[i])+'.csv')
    inc_100_sm_list = np.loadtxt('res/inc_100_sm_list_'+str(names_muns[i])+'.csv')
    
    pop_age = list_pop[i]
    
    #plt.figure(figsize=(12, 8))

    #sns.boxplot(
    #    inc_100_mc_list*pop_age,
    #    color="C0",
    #    width=0.5,
    #    boxprops=dict(
    #        facecolor="none",      # optional: keep hollow boxes
    #        edgecolor="C0",
    #        linewidth=2
    #    ),
    #    whiskerprops=dict(color="C0", linewidth=1.8),
    #    capprops=dict(color="C0", linewidth=1.8),
    #    medianprops=dict(color="C0", linewidth=2),
    #    label="Montecarlo"
    #)

    #plt.plot( list_cases[i], 'o', color='k', markersize=10, zorder=5, label = 'Real' )

    #plt.yscale("log")
    #age_groups = [ "<1 ano", "1–4 anos", "5–9 anos", "10–14 anos", "15–20 anos", "20–29 anos", "30–39 anos", "40–49 anos", "50–59 anos", "60+ anos" ] 
    #plt.xticks(np.arange(0, 10, 1), age_groups, rotation = 45, fontsize = 14)
    #plt.yticks(fontsize = 14)
    #plt.ylabel("Cases (log)", fontsize = 16)
    #plt.xlabel('Age Group', fontsize = 16)
    #plt.title(names_muns[i], fontsize = 20)
    #plt.grid(axis="y", linestyle="--", alpha=0.3)
    #plt.legend(fontsize = 14)
    #plt.title(names_muns[i], fontsize = 16)
    #plt.tight_layout()
    #plt.savefig('figs/compares_inc/'+str(names_muns[i])+'_log.pdf')
    #plt.show()
    
    ax = axes[i]

    sns.boxplot(
        inc_100_mc_list*pop_age,
        ax = ax,
        color="lightgray",
        linewidth=1.2,
        fliersize=2,
        #boxprops=dict(
        #    facecolor="none",      # optional: keep hollow boxes
        #    edgecolor="C0",
        #    linewidth=2
        #),
        #whiskerprops=dict(color="lightgray", linewidth=1.8),
       #capprops=dict(color="lightgray", linewidth=1.8),
        #medianprops=dict(color="lightgray", linewidth=2),
    label="Montecarlo"
    )
    
    ax.set_title(names_muns[i], fontsize=12)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(age_groups, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    if i % 2 == 0:
        ax.set_ylabel("Number of cases")
    else:
        ax.set_ylabel("")

    if i < 8:
       ax.set_xlabel("")
    else:
       ax.set_xlabel("Age group")

    ax.plot( list_cases[i], 'o', color='k', markersize=10, zorder=5, label = 'Real' )

    
    i = i + 1
    
plt.tight_layout()
plt.savefig('figs/compares_inc/compare_cases_inc.pdf')
plt.savefig('figs/compares_inc/compare_cases_inc.jpg')   
plt.show()
    