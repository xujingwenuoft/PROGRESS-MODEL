import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
 
#Read data from Excel
datafile = 'SR1 Input_Output_Feb20_2025.xlsx'
data = pd.read_excel(datafile) # DataFrame object

# Drop the deviation columns if not needed
data.drop(columns=['Age, standard deviation', 'Author', 'Cohort', 
       'Sex/gender, male/man',
       'Race, White',      
       'Race, Black', 'Race, Hispanic',
       'Race, Other', 'Race, Unknown',
       'Education, mean, years',
       'Education, standard deviation',
       'Education, <high school',
       'Education, high school', 'Education, some college',
       'Education, college', 'Education, <12 years', 'Education, 12 years',
       'Education, >12 years', 'Education,  ≥ 12 years',
       'Occupation, employed at time of injury',
       'Occupation, unemployed at time of injury',
       'Occupation, employed/student at time of injury',
       'Occupation, studying at time of injury',
       'Occupation, in prison at time of injury', 'Social capital, married',
       'Social capital, unmarried', 'Social capital, single',
       'Social capital, live alone', 'Social capital, live with partner',], inplace=True)

change = data['Last follow-up score on measure']-data['Baseline score on measure']
# change_min = change-(data['Last follow-up score standard deviation'] + data['Baseline score standard deviation'])
# change_max = change+(data['Last follow-up score standard deviation'] + data['Baseline score standard deviation'])

data['change'] = change
print(data['change'])
# data['change_min'] = change_min
# data['change_max'] = change_max

# Scaling, avoid dominate 
min = data['change'].min()
max = data['change'].max()
data['change'] = (data['change'] - min) / (max - min)
print(data['change'])
data.replace('NaN', pd.NA, inplace=True)

# Loop through each column and create a missing indicator
for column in data.columns:
    if data[column].isnull().any():  # Check if there are any missing values in the column
        # Create the indicator column with a '_missing' suffix
        data[column + '_missing'] = data[column].isna().astype(int)

data['Time for injury to baseline, months'] = data['Time for injury to baseline, months']/12
data['Time form baseline to last follow-up, months'] = data['Time form baseline to last follow-up, months']/12

data.drop(columns=['Baseline score on measure', 'Baseline score standard deviation', 'Last follow-up score on measure', 'Last follow-up score standard deviation','Baseline score on measure_missing',
       'Baseline score standard deviation_missing', 'Last follow-up score on measure_missing', 'Last follow-up score standard deviation_missing', 'change_missing',
       'Domain: Learning and memory_missing', 'Domain: Language_missing',
       'Domain: Learning and memory_missing', 'Domain: Complex attention_missing',
       'Domain: Executive function_missing',
       'Domain: Information processing, reaction time_missing',
       'Domain: Social cognition_missing', 'Domain: Learning and memory', 'Domain: Language',
       'Domain: Complex attention', 'Domain: Executive function',
       'Domain: Information processing, reaction time',
       'Domain: Social cognition'], inplace=True)
print(data.head())
print(data.columns)
data = data.drop(data[data['Domain: Learning and memory'] == 0].index)
data = data.drop(data[np.isnan(data['change'])].index)

data = pd.get_dummies(data, columns=['Cognitive test', 'Severity of TBI'], prefix='', prefix_sep='', drop_first=True)
print(data)

# X = data.drop(columns=['change_min', 'change_max'], axis=1)
# Y = data(['change_min', 'change_max'])
X = data.drop(columns=['change', 'Domain: Learning and memory'], axis=1)
Y = data['change']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators = 10)
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)
# Evaluate the model
print('Mean Absolute Error (MAE):', round(mean_absolute_error(Y_test, Y_pred), 3))
print('Mean Squared Error (MSE):', round(mean_squared_error(Y_test, Y_pred), 3))
print('Root Mean Squared Error (RMSE):', round(np.sqrt(mean_squared_error(Y_test, Y_pred)), 3))
print('Mean Absolute Percentage Error (MAPE):', round(mean_absolute_percentage_error(Y_test, Y_pred), 3))


feature_importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance':feature_importances})
# Round the 'importance' column to 4 decimal places
importance_df['importance'] = importance_df['importance'].round(5)
importance_df = importance_df.set_index('feature')

for feature in importance_df.index:
    if feature in ['Sex/gender, males/men',
       'Race, White',      
       'Race, Black', 'Race, Hispanic',
       'Race, Other', 'Race, Unknown',
       'Education, mean (years)',
       'Occupation, employed at time of injury',
       'Occupation, unemployed at time of injury',
       'Occupation, employed/student at time of injury',
       'Occupation, studying at time of injury',
       'Occupation, in prison at time of injury', 'Social capital, married',
       'Social capital, unmarried', 'Social capital, single',
       'Social capital, live alone', 'Social capital, live with partner',
       'Sex/gender, male/men_missing',
       'Race, White_missing',      
       'Race, Black_missing', 'Race, Hispanic_missing',
       'Race, Other_missing', 'Race, Unknown_missing',
       'Education, mean (years)_missing',
       'Occupation, employed at time of injury_missing',
       'Occupation, unemployed at time of injury_missing',
       'Occupation, employed/student at time of injury_missing',
       'Occupation, studying at time of injury_missing',
       'Occupation, in prison at time of injury_missing', 'Social capital, married_missing',
       'Social capital, unmarried_missing', 'Social capital, single_missing',
       'Social capital, live alone_missing', 'Social capital, live with partner_missing']:
        importance_df.at[feature, 'flag'] = 'Primary Predictors' # Avoid overwrting
    elif feature in ['Baseline', 'Baseline to Last Follow-Up','Age, mean','moderate-severe', 'severe', 'moderate']:
        importance_df.at[feature, 'flag'] = 'Effect Modifiers'
    else:
        importance_df.at[feature, 'flag'] = 'Cognitive Measurements & Domains'

importance_df = importance_df.sort_values(by=['flag'], ascending=[True])

heatmap_df = importance_df.drop(columns='flag')
fig = ff.create_annotated_heatmap(z=heatmap_df.values,
                                   x=heatmap_df.columns.tolist(),
                                   y=heatmap_df.index.tolist(),
                                   colorscale='YlGnBu')
# PP for primary predictor, EM for effect modifiers, CDM for cognitive domain&measures
fig.update_layout(title='Feature Importance Heatmap', xaxis_title='Importance_Domain: Learning and memory', yaxis_title='Features: PP & EM & CDM')
fig.show()
