import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Municipalities of interest ##################################################
names_muns = [
    'São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua',
    'Guarulhos', 'Manacapuru', 'Rio de Janeiro',
    'Francisco Morato', 'Macapá'
]
cod_muns = [355030, 130260, 150140, 410690, 150080,
            351880, 130250, 330455, 351630, 160030]


names_muns = [
    'São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua',
]

cod_muns = [355030, 130260, 150140, 410690, 150080]

# Loading data ###############################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None).to_numpy()
list_pop   = pd.read_csv('res/pop_muns.csv', header=None).to_numpy()

# Age groups #################################################################
age_groups = ['<1', '1–4', '5–9', '10–14', '15–19',
              '20–29', '30–39', '40–49', '50–59', '60+']

# Figure #####################################################################
n_muns = len(cod_muns)

fig, axes = plt.subplots(
    nrows=n_muns,
    ncols=2,
    figsize=(12, 3 * n_muns),
    sharex=True
)

for i, name_muni in enumerate(names_muns):

    cases = list_cases[i]
    pop   = list_pop[i]

    df_tmp = pd.DataFrame({
        'age_group': age_groups,
        'cases': cases,
        'pop': pop
    })
    df_tmp['inc_100k'] = 1e5 * df_tmp['cases'] / df_tmp['pop']

    # Left column: cases
    sns.barplot(
        data=df_tmp,
        x='age_group',
        y='cases',
        color='skyblue',
        ax=axes[i, 0]
    )
    axes[i, 0].set_ylabel(f'{name_muni}\nCases')
    axes[i, 0].tick_params(axis='x', rotation=45)
    axes[i, 0].set_xlabel('Age group')

    # Right column: incidence
    sns.barplot(
        data=df_tmp,
        x='age_group',
        y='inc_100k',
        color='skyblue',
        ax=axes[i, 1]
    )
    axes[i, 1].set_ylabel('Incidence')
    axes[i, 1].tick_params(axis='x', rotation=45)
    axes[i, 1].set_xlabel('Age group')

# Column titles
axes[0, 0].set_title('Number of cases')
axes[0, 1].set_title('Incidence per 100,000 inhabitants')

plt.tight_layout()
#plt.savefig('figs/inc_cases/all_municipalities.pdf')
plt.savefig('figs/inc_cases/all_municipalities.jpg', dpi=300)
plt.show()