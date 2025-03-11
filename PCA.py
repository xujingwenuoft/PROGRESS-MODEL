import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

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

# data['Time from injury to baseline, months'] = data['Time from injury to baseline, months']/12
# data['Time from baseline to last follow-up, months'] = data['Time from baseline to last follow-up, months']/12

age_SD_MEAN = data['Age, standard deviation'].median() # median imputation is less sensitive to outliers than mean imputation
data['Age, standard deviation'] = data['Age, standard deviation'].fillna(age_SD_MEAN)

edu_SD_MEAN = data['Education, standard deviation'].median()
data['Education, standard deviation'] = data['Education, standard deviation'].fillna(edu_SD_MEAN)

edu_mean_MEAN = data['Education, mean, years'].median()
data['Education, mean, years'] = data['Education, mean, years'].fillna(edu_mean_MEAN)

BL_score_SD_MEAN = data['Baseline score standard deviation'].median()
data['Baseline score standard deviation'] = data['Baseline score standard deviation'].fillna(BL_score_SD_MEAN)

BL_score_SD_MEAN = data['Baseline score standard deviation'].median()
data['Baseline score standard deviation'] = data['Baseline score standard deviation'].fillna(BL_score_SD_MEAN)

# Target normalization
change_mean = data['Last follow-up score on measure']-data['Baseline score on measure']
data['change_mean'] = change_mean
min = data['change_mean'].min()
max = data['change_mean'].max()
data['change_mean'] = (data['change_mean'] - min) / (max - min) # min-max outcome normalization

data.drop(columns=[ 
        'Last follow-up score on measure', 
        'Last follow-up score standard deviation',
       'Cognitive domain: Perceptual-motor', 'Cognitive domain: Learning and memory',
       'Cognitive domain: Complex attention', 'Cognitive domain: Executive function',
       'Cognitive domain: Information processing speed, reaction time',
       'Cognitive domain: Social cognition'], inplace=True)

data = data.drop(data[data['Cognitive domain: Language'] == 0].index)
data = pd.get_dummies(data, columns=['Severity of TBI'], prefix='', prefix_sep='', drop_first=False)
X=data.drop(columns=["Cognitive domain: Language", 'Moderate-Severe','change_mean'])
Y = data['change_mean']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X.columns)
# Check whether X_scaled is centered
# print(X_scaled.mean(axis=0)) # [ 2.92163954e-17  4.09029535e-17  6.42760698e-17  1.09853647e-15
#  -6.19387582e-16 -4.90835442e-16  7.47939722e-15  1.11022302e-15
#   2.33731163e-16  4.20716094e-16 -7.01193489e-17], centred

# # Perform PCA
# pca = PCA()
# pca.fit(X_scaled)

# # Cumulative variance
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# print(cumulative_variance)

# # Select components retaining 95% variance
# n_components = np.argmax(cumulative_variance >= 0.95) + 1
# print(f"Number of components to retain 95% variance: {n_components}") # 6

# # # Get eigenvalues (explained variance)
# eigenvalues = pca.explained_variance_

# # Scree plot
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
# plt.axhline(y=eigenvalues[n_components - 1], color='r', linestyle='-', label=f'Threshold at PC {n_components}')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue')
# plt.xticks(range(1, len(eigenvalues) + 1))
# plt.grid()
# plt.show()

# Apply the number of component
# pca_screed = PCA(n_components=6)
# pca_screed.fit(X_scaled)
# x = pca_screed.transform(X_scaled)
# print(pca_screed.components_)

# Reserve 3 components for better visualization
# pca_screed = PCA(n_components=3)
# pca_screed.fit(X_scaled) # calculate the means for centering data
# x = pca_screed.transform(X_scaled) # apply the mean centering
# the above two lines have the same effect as fit_transform

# plt.hist(Y, bins=20, color='blue', edgecolor='black')
# plt.title("Distribution of Normalized Y")
# plt.xlabel("Normalized Y")
# plt.ylabel("Frequency")
# plt.show()

# fig = plt.figure(figsize=(10,10))
# axis = fig.add_subplot(111, projection='3d')
# # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
# sc = axis.scatter(x[:,0],x[:,1],x[:,2], c=Y,cmap='plasma')
# plt.colorbar(sc, ax=axis, label='Normalized Y')
# axis.set_xlabel("PC1", fontsize=10)
# axis.set_ylabel("PC2", fontsize=10)
# axis.set_zlabel("PC3", fontsize=10)
# axis.set_xlim(-6, 6) # adjust plot axes to the same size
# axis.set_ylim(-6, 6)
# axis.set_zlim(-6, 6)
# plt.title('3-Principal Components')
# plt.grid()
# plt.show()

# print(pca_screed.explained_variance_ratio_) # [0.35021579 0.26445787 0.16715889]


# If there are not center, mean, try SVD(more basic)
# Two datasets, one for PCA only numeric data; one only for description as rows(cohort&assessents)
