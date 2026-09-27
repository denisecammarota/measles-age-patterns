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

# Information for time series ################################################
df = df[df['CLASSI_FIN'] == 1]
df = df[df['ID_MN_RESI'].isin(cod_muns)]
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'])
df["epiyear"] = df["DT_SIN_PRI"].apply(lambda d: Week.fromdate(d).year)
df["epiweek"] = df["DT_SIN_PRI"].apply(lambda d: Week.fromdate(d).week)
df = df[df['epiyear'].isin([2018, 2019, 2020, 2021])]
df['NU_IDADE_N'] = df['NU_IDADE_N'].astype(int)
df['case'] = 1

bins = [0, 1, 5, 10, 15, 20, 30, 40, 50, 60, np.inf]
labels = ['< 1 ano', '1–4', '5–9', '10–14', '15–19',
          '20–29', '30–39', '40–49', '50–59', '60+']

df["age_group"] = pd.cut(
    df["NU_IDADE_N"],
    bins=bins,
    labels=labels,
    right=False
)

# List of initial conditions ##################################################
list_ic = []


# Plotting time series and selecting initial conditions for simulations #######

###############################################################################

## São Paulo - 355030 #########################################################

### Filtering and completing ##################################################
cod_mun = 355030
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")


### Plotting full timeseries ##################################################
df_complete = df_complete[df_complete['epiyear'] >= 2019]
df_complete = df_complete[df_complete['epiyear'] <= 2020]
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
plt.title('São Paulo')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.title('São Paulo')
plt.axvline((2019) + (21/52), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/São Paulo.pdf')
plt.show()

#df_complete = df_complete[df_complete['epiyear'] >= 2019]
#df_complete = df_complete[df_complete['epiyear'] <= 2020]
#plt.plot(df_complete['tplot'], df_complete['case'])
#plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
#plt.title('São Paulo')
#plt.xlabel('Epidemiological Week')
#plt.ylabel('Number of cases')
#plt.show()


### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 21)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Manaus - 130260 #########################################################

### Filtering and completing ##################################################
cod_mun = 130260
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")


### Plotting full timeseries ##################################################
df_complete = df_complete[df_complete['epiyear'] >= 2018]
df_complete = df_complete[df_complete['epiyear'] <= 2018]
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2018, 2019], ["2018", "2019"])
plt.title('Manaus')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.title('Manaus')
plt.axvline((2018) + (13/52), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Manaus.pdf')
plt.show()

#df_complete = df_complete[df_complete['epiyear'] >= 2018]
#df_complete = df_complete[df_complete['epiyear'] <= 2019]
#plt.plot(df_complete['tplot'], df_complete['case'])
#plt.title('Manaus')
#plt.xlabel('Epidemiological Week')
#plt.ylabel('Number of cases')
#plt.show()


### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2018) & (df_ic['epiweek'] == 13)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Belém - 150140 #########################################################

### Filtering and completing ##################################################
cod_mun = 150140
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")


### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2018, 2019, 2020, 2021], ["2018", "2019", "2020", "2021"])
plt.title('Belém')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.title('Belém')
plt.axvline((2019) + (46/52), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Belém.pdf')
plt.show()

#df_complete = df_complete[df_complete['epiyear'] >= 2019]
#df_complete = df_complete[df_complete['epiyear'] <= 2020]
#plt.plot(df_complete['tplot'], df_complete['case'])
#plt.title('Belém')
#plt.xlabel('Epidemiological Week')
#plt.ylabel('Number of cases')
#plt.show()


### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 46)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic


###############################################################################

## Curitiba - 410690 #########################################################

### Filtering and completing ##################################################
cod_mun = 410690
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")


### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
plt.title('Curitiba')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.title('Curitiba')
plt.axvline((2019 + (28/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Curitiba.pdf')
plt.show()


### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 28)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic


###############################################################################

## Ananindeua - 150080 #########################################################

### Filtering and completing ##################################################
cod_mun = 150080
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")

### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
plt.title('Ananindeua')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2019 + (45/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Ananindeua.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 45)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Guarulhos - 351880 #########################################################
cod_mun = 351880
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")

### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
plt.title('Guarulhos')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2019 + (23/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Guarulhos.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 23)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Manacapuru - 130250 ########################################################

cod_mun = 130250
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")

### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.gca().ticklabel_format(
    style='plain',
    axis='x',
    useOffset=False
)
#plt.xticks([2018, 2019], ["2018", "2019"])
plt.title('Manacapuru')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2018 + (17/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Manacapuru.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2018) & (df_ic['epiweek'] == 17)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Rio de Janeiro - 330455 ####################################################

cod_mun = 330455
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")


### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2018, 2019, 2020, 2021, 2022], ["2018", "2019", "2020", "2021", "2022"])
plt.title('Rio de Janeiro')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2019 + (41/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Rio de Janeiro.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 41)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

###############################################################################

## Francisco Morato - 351630 ##################################################

cod_mun = 351630
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")

### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021], ["2019", "2020", "2021"])
plt.title('Francisco Morato')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2019 + (28/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Francisco Morato.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2019) & (df_ic['epiweek'] == 28)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic


###############################################################################

## Macapá - 160030 ############################################################

cod_mun = 160030
df_tmp = df.copy()
df_tmp = df_tmp[df_tmp['ID_MN_RESI'] == cod_mun]
df_tmp = df_tmp.groupby(['epiweek', 'epiyear'])['case'].sum()
df_tmp = df_tmp.reset_index()
full_index = pd.MultiIndex.from_product(
    [df_tmp["epiyear"].unique(), df_tmp["epiweek"].unique()],
    names=["epiyear", "epiweek"]
)
df_complete = (
    df_tmp
    .set_index(["epiyear", "epiweek"])
    .reindex(full_index)
    .reset_index()
    .fillna({"case": 0})
)
df_complete = df_complete[
    ~((df_complete["epiweek"] == 53) & (df_complete["epiyear"] != 2020))
]
df_complete['tplot'] = df_complete["epiyear"] + (df_complete["epiweek"]/52)
df_complete = df_complete.reset_index()
df_complete = df_complete.sort_values("tplot")

### Plotting full timeseries ##################################################
plt.plot(df_complete['tplot'], df_complete['case'])
plt.xticks([2019, 2020, 2021, 2022], ["2019", "2020", "2021", "2022"])
plt.title('Macapá')
plt.xlabel('Epidemiological Week')
plt.ylabel('Number of cases')
plt.axvline((2020 + (43/52)), c = 'red', linestyle = '--')
plt.tight_layout()
plt.savefig('figs/epidemic_curves/Macapá.pdf')
plt.show()

### Selecting initial conditions ##############################################
df_ic = df.copy()
df_ic = df_ic[df_ic['ID_MN_RESI'] == cod_mun]
df_ic = df_ic.groupby(['epiweek', 'epiyear', 'age_group'])['case'].sum()
df_ic = df_ic.reset_index()
df_ic['tplot'] = df_ic["epiyear"] + (df_ic["epiweek"]/52)
df_ic = df_ic[(df_ic['epiyear'] == 2020) & (df_ic['epiweek'] == 43)]
df_ic
list_ic.append(df_ic['case'].tolist())
list_ic

# Plotting heatmap of initial conditions and saving ###########################
df_ic = np.array(list_ic).reshape(10, 10)
df_ic = pd.DataFrame(df_ic)
df_ic.index = names_muns
df_ic.columns = ['<1', '1–4', '5–9', '10–14', '15–19',
                 '20–29', '30–39', '40–49', '50–59', '60+']
sns.heatmap(df_ic, annot=True, linewidth=.5)
plt.xlabel('Age group')
plt.ylabel(r'$I_i(0)$')
plt.tight_layout()
plt.savefig('figs/initial_conditions.pdf')
plt.savefig('figs/initial_conditions.jpg')
plt.show()

# Saving initial conditions ###################################################
df_ic.to_csv('res/initial_conditions.csv')


