import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import plotly.express as px
import plotly.figure_factory as ff
 
#Read data from Excel
datafile = 'Input_Output_December2.xlsx' # 19 data items with 10 input attributes, 1 output classes
data = pd.read_excel(datafile, sheet_name='Preprocessing - Unidomain only') # DataFrame object

# Drop the deviation columns if not needed
data.drop(columns=['Author','Cognitive test', 'Cohort',
        'Gender inequality', 'Country of recruitment', 
        'Race, White',
       'Race, Black', 'Race, Hispanic',
       'Race, Other',
       'Education, <high school',
       'Education, high school',
       'Education, some college',
       'Education, college', 'Education, <12 years',
       'Education, 12 years', 'Education, >12 years',
       'Education,  ≥ 12 years',
       'Occupation, occupied at time of injury (employed, studying, in prison)',
       'Occupation, unccupied at time of injury',
       'Social capital, partnered',
       'Social capital, unpartnered'], inplace=True)

data['Time from injury to baseline, months'] = data['Time from injury to baseline, months']/12
data['Time from baseline to last follow-up, months'] = data['Time from baseline to last follow-up, months']/12

age_SD_MEAN = data['Age, standard deviation'].median() # median imputation is less sensitive to outliers than mean imputation
data['Age, standard deviation'] = data['Age, standard deviation'].fillna(age_SD_MEAN)

edu_SD_MEAN = data['Education, standard deviation'].median()
data['Education, standard deviation'] = data['Education, standard deviation'].fillna(edu_SD_MEAN)

edu_mean_MEAN = data['Education, mean, years'].median()
data['Education, mean, years'] = data['Education, mean, years'].fillna(edu_mean_MEAN)

BL_score_SD_MEAN = data['Baseline score standard deviation'].median()
data['Baseline score standard deviation'] = data['Baseline score standard deviation'].fillna(BL_score_SD_MEAN)

LU_score_SD_MEAN = data['Last follow-up score standard deviation'].median()
data['Last follow-up score standard deviation'] = data['Last follow-up score standard deviation'].fillna(LU_score_SD_MEAN)

# Target normalization
change = data['Last follow-up score on measure']-data['Baseline score on measure']
change_SD = np.sqrt(pow(2,data['Baseline score standard deviation']) + pow(2,data['Last follow-up score standard deviation']))
mean_change = change.mean()
data['change'] = (change - mean_change) / change_SD

data.drop(columns=[ 
        'Last follow-up score on measure', 
        'Last follow-up score standard deviation',
        'Baseline score standard deviation',
       'Cognitive domain: Perceptual-motor', 'Cognitive domain: Learning and memory',
       'Cognitive domain: Complex attention', 'Cognitive domain: Executive function',
       'Cognitive domain: Information processing speed, reaction time',
       'Cognitive domain: Social cognition'], inplace=True)

data = data.drop(data[data['Cognitive domain: Language'] == 0].index)
data = pd.get_dummies(data, columns=['Severity of TBI'], prefix='', prefix_sep='', drop_first=False)
combined_weight =  data['Number of samples'] / pow(2,data['Age, standard deviation'])

X_e = data.drop(columns=['Cognitive domain: Language', 'Moderate-Severe','change', 'Age, standard deviation', 'Number of samples'])
X = X_e.drop(columns=['Education, mean, years', 'Education, standard deviation'])
Y = data['change']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=52)
X_e_train, X_e_test, Y_e_train, Y_e_test = train_test_split(X_e, Y, test_size=0.2, random_state=52)
w_train, w_test = train_test_split(combined_weight, test_size=0.2, random_state=52)

scaler1 = StandardScaler()
# Reshape combined_weight to 2D
combined_weight_reshaped = w_train.values.reshape(-1, 1)
# Apply StandardScaler
weight_2D = scaler1.fit_transform(combined_weight_reshaped)
weight = weight_2D.reshape(-1)

scaler2 = StandardScaler()
scaler2.fit(X)
X_train = scaler2.transform(X_train)
X_test = scaler2.transform(X_test)

scaler3 = StandardScaler()
scaler3.fit(X_e)
X_e_train = scaler3.transform(X_e_train)
X_e_test = scaler3.transform(X_e_test)

# Apply PCA on train and test data
pca1 = PCA(n_components=6) # set the n_component less than the #X_test
pca1.fit(X_train) # fit on the training set only
X_train = pca1.transform(X_train) # shape, (15,8)
X_test = pca1.transform(X_test)

pca2 = PCA(n_components=6)
pca2.fit(X_e_train)
X_e_train = pca2.transform(X_e_train)
X_e_test = pca2.transform(X_e_test)

# # Hyperparameters choosing
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 2, stop = 100, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(2, 35, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 4, 5, 6, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 3, 4, 5]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# pprint(random_grid)

# # Random Hyperparameter Grid
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train, Y_train, sample_weight=w_train)
# # pprint(rf_random.best_params_)

# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy

# base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
# base_model.fit(X_test, Y_test)
# base_accuracy = evaluate(base_model, X_test, Y_test)

# best_model = RandomForestRegressor(max_features = 'sqrt', n_estimators = 10, random_state = 0)
# best_model.fit(X_test, Y_test)
# random_accuracy = evaluate(best_model, X_test, Y_test)
# print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# Apply SMOTE only on the training set to augment small dataset
# smote = SMOTE(random_state=42)
# X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
# print("Original class distribution:", Y_train.value_counts())
# print("SMOTE class distribution:", Y_train_smote.value_counts())

rf = RandomForestRegressor(
    max_depth=2,                
    min_samples_split=5,        # further control overfit
    min_samples_leaf=3,         
    n_estimators=10,            
    max_features='sqrt',        
    random_state=42
) #v3
rf.fit(X_train, Y_train, sample_weight=w_train)
Y_pred = rf.predict(X_test)

rf_e = RandomForestRegressor(
    max_depth=2,                
    min_samples_split=5,        # further control overfit
    min_samples_leaf=3,         
    n_estimators=10,            
    max_features='sqrt',        
    random_state=42
)
rf_e.fit(X_e_train, Y_e_train, sample_weight=w_train)
Y_e_pred = rf_e.predict(X_e_test)

def plot_learning_curves(estimator, X_train, y_train, sample_weight=None, scoring='neg_mean_squared_error', cv=5, modelName='Baseline'):
    """
    Plots the learning curve for a given estimator.
    
    Parameters:
        - estimator: The model to evaluate.
        - X_train: Features for training.
        - y_train: Target for training.
        - sample_weight: Sample weights to apply during training (if needed).
        - scoring: Metric for evaluation. Defaults to negative mean squared error.
        - cv: Number of cross-validation splits.
        - modelName: model type
    """
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        fit_params={'sample_weight': sample_weight}
    )

    # Compute mean and standard deviation
    train_scores_mean = -np.mean(train_scores, axis=1)  # Reverse sign for MSE
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    # Plot learning curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label="Training error", marker='o')
    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std, 
                     train_scores_mean + train_scores_std, 
                     alpha=0.2)

    plt.plot(train_sizes, val_scores_mean, label="Validation error", marker='o')
    plt.fill_between(train_sizes, 
                     val_scores_mean - val_scores_std, 
                     val_scores_mean + val_scores_std, 
                     alpha=0.2)

    plt.title(f"{modelName} Learning Curves")
    plt.xlabel("Training Set Size")
    plt.ylabel("Mean Squared Error")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# plot_learning_curves(rf, X_train, Y_train, sample_weight=w_train, modelName='Baseline')
# plot_learning_curves(rf_e, X_e_train, Y_e_train, sample_weight=w_train, modelName='Baseline+Education')

# scores = cross_val_score(rf, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
# print(f"Baseline Mean Cross-Validation Error: {-scores.mean():.3f}")
# scores_e = cross_val_score(rf_e, X_e_train, Y_e_train, cv=5, scoring='neg_mean_squared_error')
# print(f"Baseline+Education Mean Cross-Validation Error: {-scores_e.mean():.3f}")

# # Evaluate the models
# print('\nError of Baseline')
# MAE_bl = mean_absolute_error(Y_test, Y_pred)
# MSE_bl = mean_squared_error(Y_test, Y_pred)
# RMSE_bl = np.sqrt(mean_squared_error(Y_test, Y_pred))
# MAPE_bl = mean_absolute_percentage_error(Y_test, Y_pred)
# print('Mean Absolute Error (MAE):', round(MAE_bl, 5))
# print('Mean Squared Error (MSE):', round(MSE_bl, 5))
# print('Root Mean Squared Error (RMSE):', round(RMSE_bl, 5))
# print('Mean Absolute Percentage Error (MAPE):', round(MAPE_bl, 5))
# print('\nError of Baseline+Education')
# MAE_e = mean_absolute_error(Y_e_test, Y_e_pred)
# MSE_e = mean_squared_error(Y_e_test, Y_e_pred)
# RMSE_e = np.sqrt(mean_squared_error(Y_e_test, Y_e_pred))
# MAPE_e = mean_absolute_percentage_error(Y_e_test, Y_e_pred)
# print('Mean Absolute Error (MAE):', round(MAE_e, 5))
# print('Mean Squared Error (MSE):', round(MSE_e, 5))
# print('Root Mean Squared Error (RMSE):', round(RMSE_e, 5))
# print('Mean Absolute Percentage Error (MAPE):', round(MAPE_e, 5))
# print('\nError decrease percentage from baseline to bl+edu')
# print(f'MAE: {100 * (MAE_e - MAE_bl) / MAE_bl:.2f}%')
# print(f'MSE: {100 * (MSE_e - MSE_bl) / MSE_bl:.2f}%')
# print(f'RMSE: {100 * (RMSE_e - RMSE_bl) / RMSE_bl:.2f}%')
# print(f'MAPE: {100 * (MAPE_e - MAPE_bl) / MAPE_bl:.2f}%')

# # Investigate feature importance after PCA
# print(f"Baseline PCA commponent variance ratio: {pca1.explained_variance_ratio_}")
# # Map PCA Components to original features
feature_contributions = pd.DataFrame(
    pca1.components_,
    columns=X.columns,
    index=[f'PC{i+1}' for i in range(pca1.n_components_)]
)
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 500)  # Adjust width for better readability
# print(f"Baseline feature_contributions: {feature_contributions}")
feature_importances = rf.feature_importances_
# print(f"Baseline model feature importance: {feature_importances}")

# # Investigate feature importance after PCA
# print(f"Baseline+Education PCA commponent variance ratio: {pca2.explained_variance_ratio_}")
# # Map PCA Components to original features
feature_contributions_e = pd.DataFrame(
    pca2.components_,
    columns=X_e.columns,
    index=[f'PC{i+1}' for i in range(pca2.n_components_)]
)
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', 500)  # Adjust width for better readability
# print(f"Baseline+Education feature_contributions: {feature_contributions_e}")
feature_importances_e = rf_e.feature_importances_
# print(f"Baseline+Education model feature importance: {feature_importances_e}")


# # print(pca2.components_) #(6,6)
# importances = feature_importances @ np.array(pca1.components_) 
# # # Convert result to a DataFrame or Series for readability
# # result_df = pd.DataFrame(importances, index=feature_contributions.columns, columns=['Weighted_PC_Contribution'])
# # # Print the result
# # print(result_df)
# feature_names = feature_contributions.columns
# # feature_importances = rf.feature_importances_
# importance_df = pd.DataFrame({'feature': feature_names, 'importance':importances})
# # # Round the 'importance' column to 5 decimal place
# importance_df['importance'] = importance_df['importance'].round(5)
# importance_df = importance_df.set_index('feature')

# # # Baseline Heatmap
# for feature in importance_df.index:
#     if feature in ['Sex/gender, male/man', 'Race, White',
#        'Race, Black', 'Race, Hispanic',
#        'Race, Other',
#        'Education, <high school',
#        'Education, high school',
#        'Education, some college',
#        'Education, college', 'Education, <12 years',
#        'Education, 12 years', 'Education, >12 years',
#        'Education,  ≥ 12 years',
#        'Occupation, occupied at time of injury (employed, studying, in prison)',
#        'Occupation, unccupied at time of injury',
#        'Social capital, partnered',
#        'Social capital, unpartnered'
#        'Sex/gender, male/man_missing',
#        'Race, White_missing',      
#        'Race, Black_missing', 'Race, Hispanic_missing',
#        'Race, Other_missing',
#        'Education, mean, years_missing',
#        'Education, standard deviation_missing',
#        'Occupation, occupied at time of injury (employed, studying, in prison)_missing',
#        'Occupation, unccupied at time of injury_missing', 'Social capital, partnered_missing',
#        'Social capital, unpartnered']:
#         importance_df.at[feature, 'flag'] = 'Primary Predictors' # Avoid overwrting
#     elif feature in ['Time from injury to baseline, months', 'Time from baseline to last follow-up, months',\
#                      'Age, mean, years','Moderate-Severe', 'Severe', 'Moderate', 'New Zealand', 'Norway', 'UK (Scotland)', 'USA', 'Canada',\
#                      'Number of samples']:
#     # elif feature in ['Time from injury to baseline, months', 'Time from baseline to last follow-up, months',\
#     #                  'Age, mean, years','Age, standard derivative','Moderate-Severe', 'Severe', 'Moderate', \
#     #                 'New Zealand', 'Norway', 'UK (Scotland)', 'USA',
#     #                  'Number of samples']:
#         importance_df.at[feature, 'flag'] = 'Effect Modifiers'
#     else:
#         importance_df.at[feature, 'flag'] = 'Cognitive Measurements & Domains'

# importance_df = importance_df.sort_values(by=['flag'], ascending=[True])

# importance_df = importance_df.drop(importance_df[(importance_df['importance'] == 0) & (importance_df['flag'] == 'Cognitive Measurements & Domains')].index)

# heatmap_df = importance_df.drop(columns='flag')
# fig = ff.create_annotated_heatmap(z=heatmap_df.values,
#                                    x=heatmap_df.columns.tolist(),
#                                    y=heatmap_df.index.tolist(),
#                                    colorscale='YlGnBu',
#                                    showscale=True)
# # Primary Predictors for primary predictor, Effect Modifiers for effect modifiers, Cognitive Measurements & Domains for cognitive domain&measures
# fig.update_layout(title='Feature Importance Heatmap', xaxis_title='Importance_Domain: Language', yaxis_title='Features: Primary Predictors & Effect Modifiers & Cognitive Measurements & Domains')
# fig.update_annotations(align="left")
# fig.show()

# # Education+Basline Heatmap
importances_e = feature_importances_e @ np.array(pca2.components_) 
# # Convert result to a DataFrame or Series for readability
# result_df = pd.DataFrame(importances, index=feature_contributions.columns, columns=['Weighted_PC_Contribution'])
# # Print the result
# print(result_df)
feature_names_e = feature_contributions_e.columns
# feature_importances = rf.feature_importances_
importance_df_e = pd.DataFrame({'feature': feature_names_e, 'importance':importances_e})
# # Round the 'importance' column to 5 decimal place
importance_df_e['importance'] = importance_df_e['importance'].round(5)
importance_df_e = importance_df_e.set_index('feature')

for feature in importance_df_e.index:
    if feature in ['Sex/gender, male/man',
       'Race, White',      
       'Race, Black', 'Race, Hispanic',
       'Race, Other', 'Race, Unknown',
       'Education, mean, years',
       'Education, standard deviation',
       'Occupation, employed at time of injury',
       'Occupation, unemployed at time of injury',
       'Occupation, employed/student at time of injury',
       'Occupation, studying at time of injury',
       'Occupation, in prison at time of injury', 'Social capital, married',
       'Social capital, unmarried', 'Social capital, single',
       'Social capital, live alone', 'Social capital, live with partner',
       'Sex/gender, male/man_missing',
       'Race, White_missing',      
       'Race, Black_missing', 'Race, Hispanic_missing',
       'Race, Other_missing', 'Race, Unknown_missing',
       'Education, mean, years_missing',
       'Education, standard deviation_missing',
       'Occupation, employed at time of injury_missing',
       'Occupation, unemployed at time of injury_missing',
       'Occupation, employed/student at time of injury_missing',
       'Occupation, studying at time of injury_missing',
       'Occupation, in prison at time of injury_missing', 'Social capital, married_missing',
       'Social capital, unmarried_missing', 'Social capital, single_missing',
       'Social capital, live alone_missing', 'Social capital, live with partner_missing']:
        importance_df_e.at[feature, 'flag'] = 'Primary Predictors' # Avoid overwrting
        # elif feature in ['Time from injury to baseline, months', 'Time from baseline to last follow-up, months',
        #                 'Age, mean, years','Age, standard derivative','Moderate-Severe', 'Severe', 'Moderate',\
        #                 'New Zealand', 'Norway', 'UK (Scotland)', 'USA',
        #                 'Number of samples']:
    elif feature in ['Time from injury to baseline, months', 'Time from baseline to last follow-up, months',
                    'Age, mean, years','Moderate-Severe', 'Severe', 'Moderate',\
                'New Zealand', 'Norway', 'UK (Scotland)', 'USA', 'Canada',\
                    'Number of samples']:
        importance_df_e.at[feature, 'flag'] = 'Effect Modifiers'
    else:
        importance_df_e.at[feature, 'flag'] = 'Cognitive Measurements & Domains'

importance_df_e = importance_df_e.sort_values(by=['flag'], ascending=[True])
importance_df_e = importance_df_e.drop(importance_df_e[(importance_df_e['importance'] == 0.0) & (importance_df_e['flag'] == 'Cognitive Measurements & Domains')].index)

heatmap_df_e = importance_df_e.drop(columns='flag')
fig_e = ff.create_annotated_heatmap(z=heatmap_df_e.values,
                                   x=heatmap_df_e.columns.tolist(),
                                   y=heatmap_df_e.index.tolist(),
                                   colorscale='YlGnBu',
                                   showscale=True)
# Primary Predictors for primary predictor, Effect Modifiers for effect modifiers, Cognitive Measurements & Domains for cognitive domain&measures
fig_e.update_layout(title='Education added Importance Heatmap', xaxis_title='Importance_Domain: Language', yaxis_title='Features: Primary Predictors(+Education) & Effect Modifiers & Cognitive Measurements & Domains')
fig_e.update_annotations(align="left")
fig_e.show()
