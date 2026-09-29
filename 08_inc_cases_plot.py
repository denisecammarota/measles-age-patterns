import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Municipalities of interest ##################################################
names_muns = [
    'São Paulo', 'Manaus', 'Belém', 'Curitiba', 'Ananindeua',
    'Guarulhos', 'Manacapuru', 'Rio de Janeiro',
    'Francisco Morato', 'Macapá'
]
cod_muns = [355030, 130260, 150140, 410690, 150080,
            351880, 130250, 330455, 351630, 160030]

# Loading data ###############################################################
list_cases = pd.read_csv('res/cases_muns.csv', header=None).to_numpy()
list_pop   = pd.read_csv('res/pop_muns.csv', header=None).to_numpy()

# Age groups #################################################################
age_groups = ['<1', '1–4', '5–9', '10–14', '15–19',
              '20–29', '30–39', '40–49', '50–59', '60+']

# Figure #####################################################################
n_muns = len(cod_muns)
ncols  = 2
nrows  = int(np.ceil(n_muns / ncols))

color_cases = 'black'
color_inc   = 'gray'

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(7 * ncols, 3.5 * nrows),
    sharex=True
)
axes = axes.flatten()

x     = np.arange(len(age_groups))
width = 0.4

for i, name_muni in enumerate(names_muns):

    cases = list_cases[i]
    pop   = list_pop[i]
    inc   = 1e5 * cases / pop

    ax1 = axes[i]
    ax2 = ax1.twinx()

    # Cases (left axis)
    b1 = ax1.bar(x - width / 2, cases, width,
                 color=color_cases, alpha=0.8, label='Cases')
    # Incidence (right axis)
    b2 = ax2.bar(x + width / 2, inc, width,
                 color=color_inc, alpha=0.8, label='Incidence')

    ax1.set_title(name_muni, fontsize = 18)
    #ax1.set_ylabel('Number of cases', color=color_cases, fontsize = 14)
    #ax2.set_ylabel('Incidence per 100,000', color=color_inc, fontsize = 14)
    ax1.tick_params(axis='y', colors=color_cases, labelsize=14)
    ax2.tick_params(axis='y', colors=color_inc, labelsize=14)

    ax1.set_xticks(x)
    #ax1.set_xticklabels(age_groups, rotation=45)
    #ax1.set_xlabel('Age group', fontsize = 14)

    # Single legend per panel
    #ax1.legend(handles=[b1, b2], loc='upper right', fontsize = 12)

     # Age group labels only on the bottom panel of each column
    is_bottom = (i + ncols) >= n_muns
    if is_bottom:
        ax1.set_xticklabels(age_groups, rotation=45, fontsize=14)
        ax1.set_xlabel('Age group', fontsize = 16)
    else:
        ax1.tick_params(axis='x', labelbottom=False)

    #ax1.legend(handles=[b1, b2], loc='upper right', fontsize=14)

# Hide unused axes (if n_muns is odd)
for j in range(n_muns, len(axes)):
    axes[j].set_visible(False)
    
    
fig.legend(
    handles=[b1, b2],
    labels=['Cases', 'Incidence per 100,000'],
    loc='lower center',
    ncol=2,
    fontsize=16,
    frameon=False
)

fig.text(0.02, 0.5, 'Number of cases', rotation=90,
         va='center', ha='center', color=color_cases, fontsize=22)
fig.text(0.98, 0.5, 'Incidence per 100,000', rotation=270,
         va='center', ha='center', color=color_inc, fontsize=22)

plt.tight_layout(rect=[0.03, 0.03, 0.97, 1])
plt.subplots_adjust(wspace=0.3)
plt.savefig('figs/inc_cases/all_municipalities.pdf')
plt.savefig('figs/inc_cases/all_municipalities.jpg', dpi=300)
plt.show()