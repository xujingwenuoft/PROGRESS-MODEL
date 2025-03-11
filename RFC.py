import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# from pycaret.classification import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
 
#Read data from Excel
datafile = 'SR1 Input_Output_Feb20_2025.xlsx'
data = pd.read_excel(datafile, sheet_name="Postprocessed raw score") # DataFrame object

data.drop(columns=['Cognitive test', 'Author', 'Score', 'Cohort', 
                   'Baseline (SD)', 'Baseline (other reporting)', 'Last FU (SD)', 'Last FU (other reporting)',
                   'Race, White (proportion)','Race, Black (proportion)', 'Race, Hispanic (proportion)','Race, Other (proportion)',
                   'Education, <high school (proportion)','Education, high school (proportion)','Education, some college (proportion)',
                   'Education, college (proportion)', 'Education, <12 years (proportion)','Education, 12 years (proportion)', 
                   'Education, >12 years (proportion)','Education,  ≥ 12 years (proportion)',
                   'Occupation, employed at time of injury (working for pay; full or part-time) (proportion)',
                  'Occupation, not employed at time of injury (retired, in prison, student, housewife, etc) (proportion)',
                  'Social capital, partnered (proportion)',
                  'Social capital, unpartnered (proportion)', 'Mechanism of Injury, MVA',
                  'Mechanism of Injury, falls', 'Mechanism of Injury, assault',
                  'Mechanism of Injury, sport related',
                  'Mechanism of Injury, work related/industrial',
                  'Mechanism of Injury, domestic accident',
                  'Mechanism of Injury, blast/explosive',
                  'Mechanism of Injury, blunt/mechanical', 'Mechanism of Injury, other',
                   'BL SD', 'Last FU value', 'Last FU SD'], inplace=True)
print(data.columns)


# # # Data Cleaning
# # # Deal with missing data: imputation + deletion
# data.replace('NaN', pd.NA, inplace=True)

# # Loop through each column and create a missing indicator
# for column in data.columns:
#     if data[column].isnull().any():  # Check if there are any missing values in the column
#         # Create the indicator column with a '_missing' suffix
#         data[column + '_missing'] = data[column].isna().astype(int)

# # Method 1
# age_min = max(0, data['Age, mean'][0] - 3*data['Age, standard deviation'][0])
# age_max =  data['Age, mean'] + 3*data['Age, standard deviation']

# norm_age = (data['Age, mean'] - age_min) / (age_max - age_min)

# # Change T1 and T2 into years unit
# norm_T1 = data['Baseline (months)']/12
# norm_T2 = data['BL to last FU (months)']/12

# # # Replace original columns with normalized values
# data['age_norm'] = norm_age
# data['Baseline (months)'] = norm_T1
# data['BL to last FU (months)'] = norm_T2

# # Drop the deviation columns if not needed
# data.drop(columns=['Age, mean', 'Age, standard deviation', 'Author', 'Cohort'], inplace=True)

# data.fillna(-1, inplace=True)  # Example: filling with -1

# # One-Hot Encoding for severity, race
# data = pd.get_dummies(data, columns=['Cognitive test', 'Severity'], prefix='', prefix_sep='', drop_first=True)
# # print(data.columns.tolist())

# # Label Encoding for outcome
# le_outcome = LabelEncoder()
# data['Assessment Outcome (BL to last FU)'] = le_outcome.fit_transform(data['Assessment Outcome (BL to last FU)'])
# label_mapping = dict(zip(le_outcome.classes_, range(len(le_outcome.classes_))))
# # print("Outcome labels:")
# # print(label_mapping)

# # Features and target variable separation
# # X = data.drop('Assessment Outcome (BL to last FU)', axis=1)
# # Y = data['Assessment Outcome (BL to last FU)']

# # Train-Test Split
# # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# data_train, data_test = train_test_split(data,test_size=0.2, random_state=42)

# # Initialize pycaret
# s = setup(data=data_train, test_data=data_test, fit_imbalance=True, target='Assessment Outcome (BL to last FU)',session_id=123)
# model = compare_models()
# print(model)
# evaluate_model(model)

# # Random Forest Model training
# # Model without class balance
# # rf_model = RandomForestClassifier(random_state=42)
# # rf_model.fit(X_train, Y_train)

# # Make prediction
# # Y_pred = rf_model.predict(X_test)

# # Evaluate the model
# # accuracy = accuracy_score(Y_test, Y_pred)
# # class_report = classification_report(Y_test, Y_pred)
# # conf_matrix = confusion_matrix(Y_test, Y_pred)

# # print(f'Accuracy: {accuracy}')
# # print("Classication Report")
# # print(class_report)
# # print("Confusion Matrix")
# # print(conf_matrix)

# # AUC-ROC (for binary classification)
# Y_pred_proba = model.predict_proba(data_test)[:, 1]  # Probability for the positive class
# # print("AUC-ROC:", roc_auc_score(Y_test, Y_pred_proba))

# # Examine and visualize feature importance, test overfitting
# feature_importances = model.feature_importances_
# feature_names = data_train.drop(columns=['Assessment Outcome (BL to last FU)'], inplace=True).columns
# # print(feature_names.tolist())
# importance_df = pd.DataFrame({'feature': feature_names, 'importance':feature_importances})
# # Round the 'importance' column to 4 decimal places
# importance_df['importance'] = importance_df['importance'].round(4)
# importance_df = importance_df.set_index('feature')
# # Reshape DataFrame for heatmap

# # print(importance_df.index.tolist())
# # print(importance_df.index.tolist()) # Same as X_train.columns
# # Assign group flages to primary predictors(y1), effect modifiers(y2), and cognitive domain&measurements(x)
# for feature in importance_df.index:
#     if feature in ['Sex/gender, male/men (proportion)', 'Sex/gender, male/men (proportion)', \
#                 'Race, White (proportion)', 'Race, Black (proportion)', 'Race, Hispanic (proportion)', \
#                 'Race, Other (proportion)', 'Race, Unknown (proportion)', \
#                 'Sex/gender, male/men (proportion)_missing', 'Race, White (proportion)_missing', \
#                 'Race, Black (proportion)_missing', 'Race, Hispanic (proportion)_missing', \
#                 'Race, Other (proportion)_missing', 'Race, Unknown (proportion)_missing']:
#         importance_df.at[feature, 'flag'] = 'Primary Predictors' # Avoid overwrting
#     elif feature in ['Baseline (months)', 'BL to last FU (months)','age_norm','mod-sev', 'sev']:
#         importance_df.at[feature, 'flag'] = 'Effect Modifiers'
#     else:
#         importance_df.at[feature, 'flag'] = 'Coganative Domains & Measurements'
# # print(importance_df.columns) # ['importance', 'flag']

# importance_df = importance_df.sort_values(by=['flag', 'importance'], ascending=[True, True])
# # print(importance_df)

# # importance_df = importance_df.groupby('flag')

# # for key, values in importance_df:
# #     print(importance_df.get_group(key), "\n\n")

# heatmap_df = importance_df.drop(columns='flag')

# # Interactive heatmap
# fig = ff.create_annotated_heatmap(z=heatmap_df.values,
#                                    x=heatmap_df.columns.tolist(),
#                                    y=heatmap_df.index.tolist(),
#                                    colorscale='YlGnBu')
# # PP for primary predictor, EM for effect modifiers, CDM for cognitive domain&measures
# fig.update_layout(title='Feature Importance Heatmap', xaxis_title='Importance', yaxis_title='Features: PP & EM & CDM')
# fig.show()






