import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

#Read data from Excel
datafile = 'Input_Output_December2.xlsx'
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

# data['Time from injury to baseline, months'] = data['Time from injury to baseline, months']/12
# data['Time from baseline to last follow-up, months'] = data['Time from baseline to last follow-up, months']/12

age_SD_MEAN = data['Age, standard deviation'].median().round(4) # median imputation is less sensitive to outliers than mean imputation
data['Age, standard deviation'] = data['Age, standard deviation'].fillna(age_SD_MEAN)

edu_SD_MEAN = data['Education, standard deviation'].median().round(4)
data['Education, standard deviation'] = data['Education, standard deviation'].fillna(edu_SD_MEAN)

edu_mean_MEAN = data['Education, mean, years'].median().round(4)
data['Education, mean, years'] = data['Education, mean, years'].fillna(edu_mean_MEAN)

BL_score_SD_MEAN = data['Baseline score standard deviation'].median().round(4)
data['Baseline score standard deviation'] = data['Baseline score standard deviation'].fillna(BL_score_SD_MEAN)

data.drop(columns=[ 
        'Last follow-up score on measure', 
        'Last follow-up score standard deviation',
       'Cognitive domain: Perceptual-motor', 'Cognitive domain: Learning and memory',
       'Cognitive domain: Complex attention', 'Cognitive domain: Executive function',
       'Cognitive domain: Information processing speed, reaction time',
       'Cognitive domain: Social cognition'], inplace=True)

data = data.drop(data[data['Cognitive domain: Language'] == 0].index)
data = pd.get_dummies(data, columns=['Severity of TBI'], prefix='', prefix_sep='', drop_first=False)
X=data.drop(columns=["Cognitive domain: Language"])

# Compute the correlation matrix
correlation_matrix = X.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# # Plot the heatmap with adjusted font size
plt.figure(figsize=(10, 8))  # Adjust the figure size for better visibility
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    annot_kws={"size": 5},  # Set font size for annotations
    cbar_kws={'shrink': 0.8}
)

# Adjust axis label font size
plt.xticks(rotation=45, ha='right', fontsize=8)  # Set smaller font size for x-axis
plt.yticks(rotation=0, fontsize=8)  # Set smaller font size for y-axis
plt.title("Correlation Matrix Heatmap", pad=20, fontsize=10)  # Minimized title font
# Automatically adjust subplot parameters to fit labels
plt.tight_layout()
# Show the plot
plt.show()

# VIF
# Drop highly correlated variables: moderate-severe and severe to avoid linearly dependent
variables = X.drop(columns=['Moderate-Severe'])

# Standardization
scaler = StandardScaler()
variables_scaled = pd.DataFrame(scaler.fit_transform(variables), columns=variables.columns)

vif = pd.DataFrame()
vif['Variable'] = variables.columns
vif['VIF'] = [variance_inflation_factor(variables_scaled.values, i) for i in range(variables_scaled.shape[1])]
print(vif)