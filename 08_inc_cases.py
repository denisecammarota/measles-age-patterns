import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dbfread import DBF
from epiweeks import Week, Year
import os

# Municipalities of interest ##################################################

names_muns = ['São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua', 'Guarulhos', 'Manacapuru', 'Rio de Janeiro', 'Francisco Morato', 'Macapá']
cod_muns = [355030, 130260, 150140, 410690, 150080, 351880, 130250, 330455, 351630, 160030]

# Loading data on cases #######################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None, sep = ',') 
list_cases = list_cases.to_numpy()

# Loading data on population ##################################################
list_pop = pd.read_csv('res/pop_muns.csv', header=None, sep = ',') 
list_pop = list_pop.to_numpy()

# Age groups ##################################################################
age_groups = ['<1', '1–4', '5–9', '10–14', '15–19',
                 '20–29', '30–39', '40–49', '50–59', '60+']

# Plotting together both plots ################################################
i = 0
for cod_mun in cod_muns:
    name_muni = names_muns[i]
    cases = list_cases[i]
    pop = list_pop[i]
    df_tmp = pd.DataFrame({'age_group': age_groups,
                          'cases': cases,
                          'pop':pop})
    df_tmp['inc_100k'] = (10**5)*(cases/pop)
    
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # Left: number of cases
    sns.barplot(
        data=df_tmp,
        x='age_group',
        y='cases',
        color='skyblue',
        ax=axes[0]
    )
    axes[0].set_xlabel('Age group')
    axes[0].set_ylabel('Number of cases')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Right: incidence per 100k
    sns.barplot(
        data=df_tmp,
        x='age_group',
        y='inc_100k',
        color='skyblue',
        ax=axes[1]
    )
    axes[1].set_xlabel('Age group')
    axes[1].set_ylabel('Incidence per 100,000 inhabitants')
    axes[1].tick_params(axis='x', rotation=45)
    
    fig.suptitle(name_muni, fontsize=14)
    plt.tight_layout()
    plt.savefig('figs/inc_cases/'+str(name_muni)+'.pdf')
    plt.savefig('figs/inc_cases/'+str(name_muni)+'.jpg')
    plt.show()
    
    i = i + 1


