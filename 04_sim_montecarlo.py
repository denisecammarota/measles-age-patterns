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

# Final size numerical expression to be solved ################################
def eq_S0(var, cases, Rij, pop):
    eqs = np.log(var) - np.log(var - cases) - (np.dot(Rij, cases) / pop)
    return eqs

# Loading contact matrix and gamma parameters - common to all municipalities ##

## Contact matrix #############################################################
contact_matrix = pd.read_csv('G:/CCD/CVE/RESPIRATORIAS/13_PROJ_SARAMPO/8_pastas_auxiliares/sir_model/cm_aggregated.csv')
contact_matrix = contact_matrix.drop(columns='Unnamed: 0')
names_age = contact_matrix.columns
contact_matrix = contact_matrix.to_numpy()

first_row = contact_matrix[0, :]
first_col = contact_matrix[:, 0]
contact_matrix_new = np.vstack([first_row, contact_matrix])
contact_matrix_new = np.hstack([np.vstack([contact_matrix[0,0].reshape(1,1), first_col.reshape(9,1)]),
                                contact_matrix_new])
contact_matrix = contact_matrix_new

## Gamma - the inverse of the recovery period #################################
gamma = 1./8

## Number of samples - common to all municipalities ###########################
n_samples = 10000

# Municipalities of interest and their codes ##################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

#names_muns = ['São Paulo', 'Manaus', 'Belém',  'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
#cod_muns = [355030, 130260, 150140, 150080, 351880, 130250, 330455, 351630, 160030]

# Screening estimates of susceptibility and sd ################################
list_mean = pd.read_csv('res/mean_si_muns.csv', header=None, sep = ',') 
list_sd = pd.read_csv('res/sd_si_muns.csv', header=None, sep = ',')

# Cases and population totals #################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None, sep = ',') 
list_pop = pd.read_csv('res/pop_muns.csv', header=None, sep = ',') 

# Performing Montecarlo simulation ############################################

i = 0

#cod_muns = [355030]

for cod_mun in cod_muns:
    
    np.random.seed(123)
    
    print(names_muns[i])
    
    pop_age = np.array(list_pop.loc[i])
    cases_age = np.array(list_cases.loc[i])
    sus_age = np.array(list_mean.loc[i])
    sus_age = np.append(sus_age, 0.99)
    
    if(names_muns[i] == 'Curitiba'):
        sus_age[3] = sus_age[3] - 0.19
    
    S0_guess = (1 - sus_age) * pop_age 
    
    #i = i + 1
    
    R0_list = []
    inc_age_list = []
    
    for j in range(n_samples):
        R0 = uniform(10, 10).rvs()
        #R0 = uniform(10, 10).rvs()
        beta_0 = (R0 * gamma) / np.max(np.real(np.linalg.eigvals(contact_matrix)))
        beta = beta_0 * contact_matrix
        Rij = beta / gamma
        sol = least_squares(
            fun=eq_S0,
            x0=S0_guess,
            args=(cases_age, Rij, pop_age),
            method="trf",        # trust region reflective (default)
            bounds=(cases_age + 1e-6, np.inf)  # enforce S0 > cases
        )
        R0_list.append(R0)
        inc_age_list.append(sol.x.tolist())
        
    #sns.histplot(R0_list, stat="density")
    #plt.xlabel(r'$R_0$')
    #plt.ylabel(r'Count')
    #plt.show()
    
    #age_groups = [
    #    "<1", "1–4", "5–9", "10–14", "15–20",
    #    "20–29", "30–39", "40–49", "50–59", "60+"
    #]
    #means = 1 - sus_age
    #stds  = np.array(list_sd.loc[i])
    #stds = np.append(stds, 0)
    
    inc_age_list = np.array(inc_age_list).reshape(n_samples, 10)
    
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(inc_age_list/pop_age,  palette="pastel",  linewidth=1.2, boxprops=dict(alpha=1.0))
    #plt.errorbar(
    #    np.arange(0, 9, 1),
    #    means[0:-1],
    #    yerr=stds[0:-1],
    #    linewidth=2.5,
    #    marker='o',
    #    markersize=8,
    #    capsize=5,
    #    label='Screening',
    #    color = 'k'
    #)
    #plt.title(names_muns[i], fontsize = 16)
    #plt.xticks(np.arange(0, 10, 1), age_groups, rotation = 45)
    #plt.xlabel('Age group')
    #plt.ylabel('Susceptibility')
    #plt.tight_layout()
    #plt.grid(axis='y', linestyle='--', alpha=0.3)
    #plt.legend()
    #plt.show()
    
    np.savetxt('res/R0_list_'+str(names_muns[i])+'.csv', np.array(R0_list), delimiter = ',')
    np.savetxt('res/sus_list_'+str(names_muns[i])+'.csv', inc_age_list, delimiter = ',')

    i = i + 1
    
    
    

    
       







