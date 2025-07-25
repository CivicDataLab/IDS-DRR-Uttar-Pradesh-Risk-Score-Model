import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

master_variables = pd.read_csv(os.getcwd()+r'/RiskScoreModel/data/MASTER_VARIABLES.csv')



exposure_vars = ['total_hhd','sum_population'#,"sum_aged_population","schools_count","rail_length", "net_sown_area_in_hac",
                      #"road_length"
                      #"health_centres_count",
                     ]

exposure_df = master_variables[exposure_vars + ['timeperiod', 'object_id']]


exposure_df_months = []
for month in tqdm(exposure_df.timeperiod.unique()):
    exposure_df_month = exposure_df[exposure_df.timeperiod == month]
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit scaler to the data and transform it
    exposure_df_month[exposure_vars] = scaler.fit_transform(exposure_df_month[exposure_vars])
    
    # Sum scaled exposure vars
    
    exposure_df_month['sum'] = exposure_df_month[exposure_vars].sum(axis=1)
    
    # Calculate mean and standard deviation
    mean = exposure_df_month['sum'].mean()
    std = exposure_df_month['sum'].std()
    
    # Define the conditions for each category
    conditions = [
        (exposure_df_month['sum'] <= mean),
        (exposure_df_month['sum'] > mean) & (exposure_df_month['sum'] <= mean + std),
        (exposure_df_month['sum'] > mean + std) & (exposure_df_month['sum'] <= mean + 2 * std),
        (exposure_df_month['sum'] > mean + 2 * std) & (exposure_df_month['sum'] <= mean + 3 * std),
        (exposure_df_month['sum'] > mean + 3 * std)
    ]
    
    # Define the corresponding categories
    #categories = ['very low', 'low', 'medium', 'high', 'very high']
    categories = [1, 2, 3, 4, 5]
    
    # Create the new column based on the conditions
    exposure_df_month['exposure'] = np.select(conditions, categories)#, default='outlier')

    exposure_df_months.append(exposure_df_month)

exposure = pd.concat(exposure_df_months)
exposure = exposure.drop_duplicates(subset=['timeperiod','object_id'])


master_variables = master_variables.merge(
    exposure[['timeperiod','object_id','exposure']].drop_duplicates(),
    on=['timeperiod','object_id'],
    how='left'
)

dups = (
    master_variables
    .groupby(['timeperiod','object_id'])
    .size()
    .reset_index(name='count')
)
print(dups['count'].value_counts())

#master_variables = master_variables.merge(exposure[['timeperiod', 'object_id', 'exposure']],
#                       on = ['timeperiod', 'object_id'])
print(master_variables.shape)
master_variables.to_csv(os.getcwd()+r'/RiskScoreModel/data/factor_scores_l1_exposure.csv', index=False)