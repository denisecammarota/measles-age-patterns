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

# T1: from the screening method #######################################
list_mean_sus = 1 - list_mean
list_sd_sus = list_sd

age_cols = ["<1", "1-4", "5-9", "10-14", "15-19",
            "20-29", "30-39", "40-49", "50-59"]

mean_df = list_mean_sus.copy()
sd_df   = list_sd_sus.copy()

for df in (mean_df, sd_df):
    df.index = names_muns
    df.columns = age_cols

# Combine into "mean (sd)" strings, e.g. 0.533 (0.045)
table = mean_df.round(3).astype(str) + " (" + sd_df.round(3).astype(str) + ")"

table.index.name = "Municipality"
table.columns.name = "Age group"
table


# T2: from the montecarlo methodology #################################


## Extracting data for table ###################################################

i = 0

mean_rows = []
sd_rows = []

for cod_mun in cod_muns:
    
    pop_age = np.array(list_pop.loc[i])
    
    inc_age_list = np.loadtxt('res/sus_list_'+str(names_muns[i])+'.csv',  delimiter=",")
    inc_age_list = inc_age_list / pop_age

    mean_age_list = np.mean(inc_age_list, axis = 0)[0:9]
    sd_age_list = np.std(inc_age_list, axis = 0)[0:9]
    
    mean_rows.append(mean_age_list)   # array of 10 values (incl. 60+)
    sd_rows.append(sd_age_list)

    i = i + 1
    
mean_df_2 = pd.DataFrame(mean_rows, index=names_muns, columns=age_cols)
sd_df_2   = pd.DataFrame(sd_rows,  index=names_muns, columns=age_cols)

mean_df_2.index.name = sd_df_2.index.name = "Municipality"
mean_df_2.columns.name = sd_df_2.columns.name = "Age group"

table_2 = mean_df_2.map("{:.3f}".format) + " (" + sd_df_2.map("{:.3f}".format) + ")"
table_2