from topsis import Topsis
import pandas as pd
import numpy as np 
import os
import glob

fldhzd_w = 4
exp_w = 1
vul_w = 2
resp_w = 2

## MASTER DATA WITH FACTOR SCORES
print(os.getcwd())
## INPUT: FACTOR SCORES CSV
factor_scores_dfs = glob.glob(os.getcwd()+ '/RiskScoreModel/data/factor_scores_l1*.csv')

# Select only the columns that exist in both the DataFrame and the list
factors = ['exposure', 'flood-hazard', 'vulnerability', 'government-response']

merged_df = pd.read_csv(factor_scores_dfs[0])
# Merge successive DataFrames in the list
for df in factor_scores_dfs[1:]:
    df = pd.read_csv(df)
    selected_columns = [col for col in factors if col in df.columns]
    df = df[selected_columns + ['object_id', 'timeperiod']]
    merged_df = pd.merge(merged_df, df, on=['object_id', 'timeperiod'], how='inner')

def get_financial_year(timeperiod):
    if int(timeperiod.split('_')[1]) >= 4:
        return str(int(timeperiod.split('_')[0]))+'-'+str(int(timeperiod.split('_')[0])+1)
    else:
        return str(int(timeperiod.split('_')[0]) - 1)+'-'+str(int(timeperiod.split('_')[0]))
    
merged_df['financial_year'] = merged_df['timeperiod'].apply(lambda x: get_financial_year(x))

merged_df.sort_values(by=['object_id', 'financial_year', 'timeperiod'], inplace=True)

cumulative_vars = [
    "total_tender_awarded_value",
    'Repair and Restoration_tenders_awarded_value',
    'RIDF_tenders_awarded_value',           
    'Preparedness Measures_tenders_awarded_value', 
    'Immediate Measures_tenders_awarded_value', 
    "Others_tenders_awarded_value",        
]

for var in cumulative_vars:
    cum_var_name = var + "_fy_cumsum"
    merged_df[cum_var_name] = merged_df.groupby(['object_id', 'financial_year'])[var].cumsum()

df_months = []

for month in merged_df.timeperiod.unique():
    print(month)

    df_month = merged_df[merged_df.timeperiod==month]

    evaluation_matrix = np.array(df_month[['flood-hazard', 'exposure', 'vulnerability', 'government-response']].values)
    weights = [fldhzd_w, exp_w, vul_w, resp_w]
    criterias = [True, True, True, True]

    t = Topsis(evaluation_matrix, weights, criterias)
    t.calc()
    df_month = df_month.copy()
    df_month['TOPSIS_Score'] = t.worst_similarity
    df_month = df_month.sort_values(by='TOPSIS_Score', ascending=False)
    
    compositescorelabels = [1,2,3,4,5]
    compscore = pd.cut(df_month['TOPSIS_Score'], bins=5, precision=0, labels=compositescorelabels)
    df_month['risk-score'] = compscore

    df_months.append(df_month)

topsis = pd.concat(df_months)

topsis.columns = [col.lower().replace('_', '-').replace(' ', '-') for col in topsis.columns]
print(topsis.columns)
topsis.to_csv(os.getcwd()+'/RiskScoreModel/data/risk_score.csv', index=False)

## DISTRICT LEVEL SCORES
dist_ids = pd.read_csv(os.getcwd()+'/RiskScoreModel/assets/district_objectid.csv')

compositescorelabels = ['1','2','3','4','5']

dist_vul = topsis.groupby(['district','timeperiod'])['vulnerability'].mean().reset_index()
compscore = pd.cut(dist_vul['vulnerability'], bins=5, precision=0, labels=compositescorelabels)
dist_vul['vulnerability'] = compscore
dist_vul = dist_vul.merge(dist_ids, on='district')

dist_exp = topsis.groupby(['district','timeperiod'])['exposure'].mean().reset_index()
compscore = pd.cut(dist_exp['exposure'], bins=5, precision=0, labels=compositescorelabels)
dist_exp['exposure'] = compscore
dist_exp = dist_exp.merge(dist_ids, on='district')

dist_govt = topsis.groupby(['district','timeperiod'])['government-response'].mean().reset_index()
compscore = pd.cut(dist_govt['government-response'], bins=5, precision=0, labels=compositescorelabels)
dist_govt['government-response'] = compscore
dist_govt = dist_govt.merge(dist_ids, on='district')

dist_haz = topsis.groupby(['district','timeperiod'])['flood-hazard'].mean().reset_index()
compscore = pd.cut(dist_haz['flood-hazard'], bins=5, precision=0, labels=compositescorelabels)
dist_haz['flood-hazard'] = compscore
dist_haz = dist_haz.merge(dist_ids, on='district')

topsis['risk-score'] = topsis['risk-score'].astype(int)
dist_risk = topsis.groupby(['district','timeperiod'])['risk-score'].mean().reset_index()
compscore = pd.cut(dist_risk['risk-score'], bins=5, precision=0, labels=compositescorelabels)
dist_risk['risk-score'] = compscore
dist_risk = dist_risk.merge(dist_ids, on='district')

indicators = [
    'total-tender-awarded-value', 
    'repair-and-restoration-tenders-awarded-value',
    'ridf-tenders-awarded-value',           
    'preparedness-measures-tenders-awarded-value',  
    'immediate-measures-tenders-awarded-value', 
    'others-tenders-awarded-value',
    'total-tender-awarded-value-fy-cumsum',
    'repair-and-restoration-tenders-awarded-value-fy-cumsum',
    'ridf-tenders-awarded-value-fy-cumsum',
    'preparedness-measures-tenders-awarded-value-fy-cumsum',
    'immediate-measures-tenders-awarded-value-fy-cumsum',
    'others-tenders-awarded-value-fy-cumsum',
    #'relief-and-mitigation-sanction-value',
    'sum-population',
    'inundation-intensity-sum',
    'total-hhd',
    'sum-aged-population',
    'schools-count',
    'health-centres-count',
    'total-road-length',
    'total-rail-length',
    'sd-nosanitation-hhds-pct',
    'inundation-pct',
    'inundation-intensity-mean',
    'inundation-intensity-mean-nonzero',
    'avg-electricity',
    'sd-piped-hhds-pct',
    'mean-sex-ratio',
    'elevation-mean',
    'sum-young-population',
    'avg-tele',
    'rail-count',
    'max-rain',
    'mean-rain',
    'sum-rain',
    'topsis-score',
    'risk-score',
    'exposure',
    'vulnerability',
    'government-response',
]

# Remove duplicates from indicators list
indicators = list(dict.fromkeys(indicators))

aggregation_rules = {
    'total-tender-awarded-value': 'sum', 
    'repair-and-restoration-tenders-awarded-value': 'sum',
    'ridf-tenders-awarded-value': 'sum',     
    'preparedness-measures-tenders-awarded-value': 'sum', 
    'immediate-measures-tenders-awarded-value': 'sum', 
    'others-tenders-awarded-value': 'sum',
    'total-tender-awarded-value-fy-cumsum': 'sum',
    'repair-and-restoration-tenders-awarded-value-fy-cumsum': 'sum',
    'ridf-tenders-awarded-value-fy-cumsum': 'sum',
    'preparedness-measures-tenders-awarded-value-fy-cumsum': 'sum',
    'immediate-measures-tenders-awarded-value-fy-cumsum': 'sum',
    'others-tenders-awarded-value-fy-cumsum': 'sum',
    'sum-population': 'sum',
    'inundation-intensity-sum': 'sum',
    'total-hhd': 'sum',
    'sum-aged-population': 'sum',
    'schools-count': 'sum',
    'health-centres-count': 'sum',
    'total-road-length': 'sum',
    'total-rail-length': 'sum',
    'sum-rain': 'sum',
    'sum-young-population': 'sum',
    'rail-count': 'sum',
    'sd-nosanitation-hhds-pct': 'mean',
    'inundation-pct': 'mean',
    'inundation-intensity-mean-nonzero': 'mean',
    'inundation-intensity-mean': 'mean',
    'avg-electricity': 'mean',
    'sd-piped-hhds-pct': 'mean',
    'mean-sex-ratio': 'mean',
    'mean-rain': 'mean',
    'elevation-mean': 'mean',
    'avg-tele': 'mean',
    'topsis-score': 'mean',
    'risk-score': 'mean',
    'exposure': 'mean',
    'vulnerability': 'mean',
    'government-response': 'mean',
    'flood-hazard': 'mean',
    'max-rain': 'max',
}

rounding_rules = {
    'total-tender-awarded-value': 0, 
    'repair-and-restoration-tenders-awarded-value': 0,
    'immediate-measures-tenders-awarded-value': 0, 
    'ridf-tenders-awarded-value': 0, 
    'preparedness-measures-tenders-awarded-value': 0,
    'others-tenders-awarded-value': 0,
    'total-tender-awarded-value-fy-cumsum': 0,
    'repair-and-restoration-tenders-awarded-value-fy-cumsum': 0,
    'ridf-tenders-awarded-value-fy-cumsum': 0,
    'preparedness-measures-tenders-awarded-value-fy-cumsum': 0,
    'immediate-measures-tenders-awarded-value-fy-cumsum': 0,
    'others-tenders-awarded-value-fy-cumsum': 0,
    'avg-tele': 1,
    'avg-electricity': 1,
    'mean-sex-ratio': 2,  
    'inundation-intensity-mean-nonzero': 2,  
    'sd-piped-hhds-pct': 2,
    'sd-nosanitation-hhds-pct': 2,
    'inundation-intensity-sum': 2,
    'max-rain': 2,
    'mean-rain': 2,
    'sum-rain': 2,
    'sum-aged-population': 0,
    'sum-young-population': 0,
    'sum-population': 0,
    'total-rail-length': 0,
    'total-road-length': 0,
    'elevation-mean': 0,
    'total-hhd': 0,
    'flood-hazard': 0,
    'risk-score': 0,
    'exposure': 0,
    'vulnerability': 0,
    'government-response': 0,
}

dist_indicators = topsis.groupby(['district', 'timeperiod']).agg(aggregation_rules).reset_index()
dist_indicators = dist_indicators.merge(dist_ids, on='district')

def apply_rounding_rules(df, rounding_rules):
    for column, decimals in rounding_rules.items():
        if column in df.columns:
            df[column] = df[column].round(decimals)
        else:
            print(f"Column {column} does not exist in DataFrame.")
    return df

dist = pd.concat([dist_vul.set_index(['district', 'timeperiod']),
                  dist_exp.set_index(['district', 'timeperiod'])['exposure'],
                  dist_govt.set_index(['district', 'timeperiod'])['government-response'],
                  dist_haz.set_index(['district', 'timeperiod'])['flood-hazard'],
                  dist_risk.set_index(['district', 'timeperiod'])['risk-score'],
                  dist_indicators.set_index(['district', 'timeperiod'])[indicators]],
                  axis=1).reset_index()

# for debugging
dist.to_csv(os.getcwd()+'/RiskScoreModel/data/dist_test.csv')
topsis.to_csv(os.getcwd()+'/RiskScoreModel/data/topsis_test.csv')
print(topsis.shape)

# Remove duplicate columns before concat
dist = dist.loc[:, ~dist.columns.duplicated()]
topsis = topsis.loc[:, ~topsis.columns.duplicated()]

final = pd.concat([topsis, dist], ignore_index=True)

# Apply rounding rules
final = apply_rounding_rules(final, rounding_rules)

final['financial-year'] = final['timeperiod'].apply(lambda x: get_financial_year(x))

final = final.drop(columns=['year', 'Unnamed: 0'], errors='ignore')

final.to_csv(os.getcwd()+'/RiskScoreModel/data/risk_score_final_district.csv', index=False)