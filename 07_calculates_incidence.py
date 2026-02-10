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

# Auxiliary functions #########################################################
def SIR_age(x, t, beta, gamma, age_structure):
    # derivatives of age-structured SIR system
    n_age = len(age_structure) # number of age groups
    S, I, R = np.split(x, 3)
    foi = np.dot(beta, I) / age_structure # force of infection for each age group
    dSdt = - foi * S
    dIdt = foi * S - gamma*I 
    dRdt = gamma*I
    return np.r_[dSdt, dIdt, dRdt]

def final_size_age(x0, t, R0, gamma, contact_matrix, pop_structure):
    
    # calculating beta
    beta_0 = (R0*gamma)/(np.max(np.real(np.linalg.eigvals(contact_matrix))))
    beta = beta_0 * contact_matrix
    
    # running simulation
    x = odeint(SIR_age, x0, t, (beta, gamma, pop_structure))
    
    # calculating the number of cases per age group
    S, I, R = np.split(x, 3, axis=1)
    #print(R[-1] - R[0])
    #print((R[-1] - R[0])/pop_SP_2021)
    #plt.plot(t, I)
    #plt.show()
    #if(sum(I[-1]) > 1e-3):
      # print('please increase integration time')
      #plt.plot(t, I)
      #plt.show()
        
    
    #plt.plot(t, I)
    #plt.show()
    return R[-1] - R[0]


# Municipalities under study and their codes ##################################
names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Loading data on cases #######################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None, sep = ',') 
list_cases = list_cases.to_numpy()

# Loading data on population ##################################################
list_pop = pd.read_csv('res/pop_muns.csv', header=None, sep = ',') 
list_pop = list_pop.to_numpy()

# Loading screening method data ###############################################
list_mean = pd.read_csv('res/mean_si_muns.csv', header=None, sep = ',') 
list_sd = pd.read_csv('res/sd_si_muns.csv', header=None, sep = ',')
list_mean = list_mean.to_numpy()

# Loading the contact matrix (same for all) ###################################
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

# Loading data on initial conditions ##########################################
df_ic = pd.read_csv('res/initial_conditions.csv')
df_ic = df_ic.rename(columns={'Unnamed: 0': 'MN_RESI'})

# Gamma - the inverse of the recovery period ##################################
gamma = 7./8

# Looping through each municipality ###########################################
i = 0 # signals the municipality of interest
j = 0 # signals the simulation of interest
n_samples = 10000 # total simulations that were conducted
#cod_muns = [355030]

for cod_mun in cod_muns:
    print(names_muns[i])
    R0_list = pd.read_csv('res/R0_list_'+str(names_muns[i])+'.csv', header=None, sep = ',')
    R0_list = R0_list.to_numpy().reshape(1, -1)[0]
    sus_age_list = pd.read_csv('res/sus_list_'+str(names_muns[i])+'.csv', header=None, sep = ',')
    sus_age_list = sus_age_list.to_numpy()
    pop_age = list_pop[i]
    inc_100_mc_list = []
    inc_100_sm_list = []
    I_inicial = df_ic[df_ic['MN_RESI'] == names_muns[i]]
    I_inicial = I_inicial.drop(columns = 'MN_RESI')
    I_inicial = I_inicial.loc[i].tolist()
    for j in range(n_samples):
        #print(j)
        # Monte Carlo
        R0 = R0_list[j]
        sus_age = sus_age_list[j]
        #I_inicial = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        p_sus = sus_age
        S_inicial = p_sus
        R_inicial = pop_age - S_inicial - I_inicial
        x0 = np.array([S_inicial, I_inicial, R_inicial]).flatten()
        t = np.arange(0, 52*10, 1)
        x = final_size_age(x0, t, R0, gamma, contact_matrix, pop_age)
        inc_100_mc_list.append(x/pop_age)
        
        # Screening method
        R0 = R0_list[j]
        #sus_age = sus_age_list[j]
        #I_inicial = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        p_sus = list_mean[i]
        p_sus = np.append(p_sus, 0.99)
        S_inicial = (1 - p_sus)*pop_age
        R_inicial = pop_age - S_inicial - I_inicial
        x0 = np.array([S_inicial, I_inicial, R_inicial]).flatten()
        t = np.arange(0, 52*10, 1)
        x = final_size_age(x0, t, R0, gamma, contact_matrix, pop_age)
        inc_100_sm_list.append(x/pop_age)
        
        
        
    inc_100_mc_list = np.array(inc_100_mc_list)
    inc_100_sm_list = np.array(inc_100_sm_list)
    
    
    np.savetxt('res/inc_100_mc_list_'+str(names_muns[i])+'.csv', inc_100_mc_list)
    np.savetxt('res/inc_100_sm_list_'+str(names_muns[i])+'.csv', inc_100_sm_list)
    
    plt.figure(figsize=(12, 8))

    sns.boxplot(
        inc_100_mc_list,
        color="C0",
        width=0.5,
        boxprops=dict(
            facecolor="none",      # optional: keep hollow boxes
            edgecolor="C0",
            linewidth=2
        ),
        whiskerprops=dict(color="C0", linewidth=1.8),
        capprops=dict(color="C0", linewidth=1.8),
        medianprops=dict(color="C0", linewidth=2),
        label="Montecarlo"
    )

    sns.boxplot(
        inc_100_sm_list,
        color="C1",
        width=0.5,
        boxprops=dict(
            facecolor="none",
            edgecolor="C1",
            linewidth=2
        ),
        whiskerprops=dict(color="C1", linewidth=1.8),
        capprops=dict(color="C1", linewidth=1.8),
        medianprops=dict(color="C1", linewidth=2),
        label="Screening method"
    )

    plt.plot( list_cases[i] / pop_age, 'o', color='k', markersize=10, zorder=5, label = 'Real' )

    plt.yscale("log")
    age_groups = [ "<1 ano", "1–4 anos", "5–9 anos", "10–14 anos", "15–20 anos", "20–29 anos", "30–39 anos", "40–49 anos", "50–59 anos", "60+ anos" ] 
    plt.xticks(np.arange(0, 10, 1), age_groups, rotation = 45, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel("Incidence per 100,000 (log scale)", fontsize = 16)
    plt.xlabel('Age Group', fontsize = 16)
    plt.title(names_muns[i], fontsize = 20)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(fontsize = 14)
    plt.title(names_muns[i], fontsize = 16)
    plt.tight_layout()
    plt.savefig('figs/compares_inc/'+str(names_muns[i])+'_log.pdf')
    plt.show()
    
    plt.figure(figsize=(12, 8))

    sns.boxplot(
        inc_100_mc_list,
        color="C0",
        width=0.5,
        boxprops=dict(
            facecolor="none",      # optional: keep hollow boxes
            edgecolor="C0",
            linewidth=2
        ),
        whiskerprops=dict(color="C0", linewidth=1.8),
        capprops=dict(color="C0", linewidth=1.8),
        medianprops=dict(color="C0", linewidth=2),
        label="Montecarlo"
    )

    sns.boxplot(
        inc_100_sm_list,
        color="C1",
        width=0.5,
        boxprops=dict(
            facecolor="none",
            edgecolor="C1",
            linewidth=2
        ),
        whiskerprops=dict(color="C1", linewidth=1.8),
        capprops=dict(color="C1", linewidth=1.8),
        medianprops=dict(color="C1", linewidth=2),
        label="Screening method"
    )

    plt.plot(list_cases[i] / pop_age, 'o', color='k', markersize=10, zorder=5, label = 'Real' )

    #plt.yscale("log")
    age_groups = [ "<1 ano", "1–4 anos", "5–9 anos", "10–14 anos", "15–20 anos", "20–29 anos", "30–39 anos", "40–49 anos", "50–59 anos", "60+ anos" ] 
    plt.xticks(np.arange(0, 10, 1), age_groups, rotation = 45, fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylabel("Incidence per 100,000", fontsize = 16)
    plt.xlabel('Age Group', fontsize = 16)
    plt.title(names_muns[i], fontsize = 20)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(fontsize = 14)
    plt.title(names_muns[i], fontsize = 16)
    plt.tight_layout()
    plt.savefig('figs/compares_inc/'+str(names_muns[i])+'.pdf')
    plt.show()
    
    
    i = i + 1
        
        
        
    










