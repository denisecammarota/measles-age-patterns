import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dbfread import DBF
from epiweeks import Week, Year
import os


# Municipalities of interest ##################################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Order of the custom age groups ##############################################
order = [
    '< 01 ano',
    '01 a 04 anos',
    '05 a 09 anos',
    '10 a 14 anos',
    '15 a 19 anos',
    '20 a 29 anos',
    '30 a 39 anos',
    '40 a 49 anos',
    '50 a 59 anos',
    '60 e +'
]

# Cases #######################################################################
directory = 'data/sinan/'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
dfs = []

for file in csv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dfs.append(df)
    
combined_df = pd.concat(dfs, ignore_index=True)
df = combined_df.copy()

df = df[df['CLASSI_FIN'] == 1]
df = df[df['ID_MN_RESI'].isin(cod_muns)]
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'])
df["epiyear"] = df["DT_SIN_PRI"].dt.year
df = df[df['epiyear'].isin([2018, 2019, 2020, 2021])]
df['NU_IDADE_N'] = df['NU_IDADE_N'].astype(int)
#df.loc[df['NU_IDADE_N'] < 4000, 'NU_IDADE_N'] = 0
#df.loc[df['NU_IDADE_N'] >= 4000, 'NU_IDADE_N'] = df['NU_IDADE_N'] - 4000
#df = df[df['NU_IDADE_N'] <= 60]


# Population ##################################################################
df_pop = pd.read_csv('pop-sarampo.csv')
df_pop = df_pop[df_pop['ID_MUN'].isin(cod_muns)]
df_pop = df_pop.groupby(['ID_MUN', 'FX_ETARIA'])['POP'].sum()
df_pop = df_pop.reset_index()

# Getting cases and population for all municipalities #########################

list_cases = []
list_pop = []

i = 0
cod_mun = cod_muns[0]

for cod_mun in cod_muns:
    
    # cases
    df_tmp = df
    df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
    bins_grouped = np.array([0,1,5,10,15,20, 30, 40, 50, 60, 120])
    x_cases = df_tmp['NU_IDADE_N']
    bins = plt.hist(x_cases, bins=bins_grouped,  edgecolor = 'black')
    list_cases.append(bins[0])
    
    # population
    df_pop_tmp = df_pop
    df_pop_tmp = df_pop_tmp[df_pop_tmp['ID_MUN'] == cod_mun]
    
    df_pop_tmp['FX_ETARIA'] = pd.Categorical(
        df_pop_tmp['FX_ETARIA'],
        categories=order,
        ordered=True
        )
    df_pop_tmp = df_pop_tmp.sort_values('FX_ETARIA')
    list_pop.append(df_pop_tmp['POP'].tolist())
    
    
    i = i + 1

list_cases = np.array(list_cases).reshape(len(cod_muns), len(bins[0]))
list_pop = np.array(list_pop).reshape(len(cod_muns), len(df_pop_tmp['POP'].tolist()))

np.savetxt('res/cases_muns.csv', list_cases, delimiter=",")
np.savetxt('res/pop_muns.csv', list_pop, delimiter=",")



