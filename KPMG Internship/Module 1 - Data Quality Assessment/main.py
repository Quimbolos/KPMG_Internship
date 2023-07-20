# %%

import pandas as pd

# Read in all Worksheets as different DataSets

df_transactions = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name='Transactions', skiprows=1)
df_newcostumerlist = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name='NewCustomerList', skiprows=1)
df_costumerlistjoin = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name='NewCostumerList2', skiprows=0)
df_customerdemographic = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name='CustomerDemographic', skiprows=1)
df_customeraddress = pd.read_excel('KPMG_VI_New_raw_data_update_final.xlsx', sheet_name='CustomerAddress', skiprows=1)

# %%
df_transactions.head()
df_transactions.info()
df_transactions['transaction_id'].duplicated()

# %%
df_newcostumerlist.head()
# df_newcostumerlist['deceased_indicator'].nunique()
df_newcostumerlist.info()
# %%
df_customerdemographic.head()
df_customerdemographic.info()
df_customerdemographic['customer_id'].duplicated().sum()

# %%
df_customeraddress.head()
df_customeraddress.info()

# %%
df_newcostumerlist.info()
# %%
df_duplicates = df_newcostumerlist.drop_duplicates()

df_duplicates.info()
# %%
df_costumerlistjoin.head()
df_costumerlistjoin.info()
# %%
df_duplicates = df_costumerlistjoin.drop_duplicates()

df_duplicates.info()

# %%
df_duplicates['gender'].unique()
# %%
df_costumerlistjoin.head()
df_costumerlistjoin.info()

# %%
