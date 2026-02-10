import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dbfread import DBF
from epiweeks import Week, Year
import os

# Doing estimation for two scenarios ##########################################

def coverage_age(p,e):
    return (1 +((1-e)*((1/p) -1)))**-1


def error_coverage_age(p,e):
    var = (1-e)/((e*(p-1) + 1)**2)
    return np.sqrt(var)


# Municipalities of interest ##################################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Loading all .csv files and join #############################################
directory = 'data/sinan/'
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
#csv_files = csv_files[:-1]
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
df = df[df['NU_IDADE_N'] <= 60]
i = 0

list_mean = []
list_sd = []


for cod_mun in cod_muns:
    print(cod_mun)
    df_tmp = df
    df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
    x_vac = list(df_tmp[df_tmp['CS_VACINA'] == 1]['NU_IDADE_N'])
    x_nvac = list(df_tmp[df_tmp['CS_VACINA'] == 2]['NU_IDADE_N'])
    x_ign = list(df_tmp[df_tmp['CS_VACINA'] == 9]['NU_IDADE_N'])
    bins_grouped = np.array([0,1,5,10,15,20, 30, 40, 50, 60])
    colors=['blue', 'green', 'orange']
    names=['VAC', 'NVAC', 'IGN']
    bins = plt.hist([x_vac, x_nvac, x_ign], color=colors, label=names, bins=bins_grouped,  edgecolor = 'black')
    #plt.legend()
    #plt.show()
    
    # Considering ignored as vaccinated
    vacs = bins[0][0]
    nvacs = bins[0][1]
    ign = bins[0][2]
    vacs_2 = vacs.copy()
    nvacs_2 = nvacs.copy()
    ign_2 = ign.copy()
    vacs = vacs + ign
    # first we calculate the proportion of vaccinated
    N = vacs+nvacs # sample size
    p_vacs = vacs/N # proportion of sample
    # now we calculate errors for these proportions 
    e_p_vacs = np.sqrt(p_vacs*(1 - p_vacs)/N)
    # second we propagate to calculate the vaccination coverage
    eps = 0.9 # vaccine efficacy
    e_eps_lit = 0.05 # new
    e_eps_prop = ((1 + (1-eps)*((1/p_vacs)-1))**(-2))*((1/p_vacs) - 1) # new
    c_vacs = coverage_age(p_vacs,eps)
    e_c_vacs = np.sqrt((error_coverage_age(p_vacs,eps)*e_p_vacs)**2 + (e_eps_prop*e_eps_lit)**2) # new
    
    
    # Considering half are vaccinated
    vacs_2 = vacs_2 + ign_2/2
    nvacs_2 = nvacs_2 + ign_2/2
    N_2 = vacs_2 + nvacs_2 # sample size
    p_vacs_2 = vacs_2/N_2 # proportion of sample
    # now we calculate errors for these proportions 
    e_p_vacs_2 = np.sqrt(p_vacs_2*(1 - p_vacs_2)/N_2)
    # second we propagate to calculate the vaccination coverage
    eps = 0.9 # vaccine efficacy
    e_eps_lit = 0.05 # new
    e_eps_prop_2 = ((1 + (1-eps)*((1/p_vacs_2)-1))**(-2))*((1/p_vacs_2) - 1) # new
    c_vacs_2 = coverage_age(p_vacs_2,eps)
    print(c_vacs_2)
    e_c_vacs_2 = np.sqrt((error_coverage_age(p_vacs_2,eps)*e_p_vacs_2)**2 + (e_eps_prop_2*e_eps_lit)**2) # new


    # Mean of both scenarios
    cov_mean = (0.5)*(c_vacs + c_vacs_2)
    cov_sd = np.sqrt(e_c_vacs**2 + e_c_vacs_2**2)
    
    # Plotting for this municipality
    fig, ax = plt.subplots()
    ax.errorbar(np.arange(len(c_vacs)),c_vacs, yerr = e_c_vacs, label = 'IGN')
    ax.errorbar(np.arange(len(c_vacs_2)),c_vacs_2, yerr = e_c_vacs_2, label = 'IGN/2')
    ax.errorbar(np.arange(len(c_vacs_2)), 0.5*(c_vacs + c_vacs_2), yerr = np.sqrt(e_c_vacs_2**2 + e_c_vacs**2), label = 'MEAN')
    ax.set_xticks(np.arange(len(p_vacs)))
    ax.set_xticklabels(['< 1','1-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-59'])
    ax.axhline(0.93, color = 'r')
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Susceptibility '+r'$S_i$')
    plt.legend()
    plt.title(names_muns[i], fontsize = 14)
    plt.tight_layout()
    plt.savefig('figs/'+names_muns[i]+'.pdf')
    plt.show()
    
    # Adding counter + 1
    i = i + 1
    
    # Saving into our results
    list_mean.append(cov_mean)
    list_sd.append(cov_sd)
    

list_mean = np.array(list_mean)
list_mean = list_mean.reshape(len(names_muns), len(['< 1','1-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-59']))

list_sd = np.array(list_sd)
list_sd = list_sd.reshape(len(names_muns), len(['< 1','1-4', '5-9', '10-14', '15-19', '20-29', '30-39', '40-49', '50-59']))

np.savetxt('res/mean_si_muns.csv', list_mean, delimiter=",")
np.savetxt('res/sd_si_muns.csv', list_sd, delimiter=",")
