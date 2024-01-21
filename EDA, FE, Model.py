#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


# In[2]:


df1 = pd.read_csv('D:/Technocolabs/Individual Project/Datasets/1year_cleaned.csv')
df2 = pd.read_csv('D:/Technocolabs/Individual Project/Datasets/2year_cleaned.csv')
df3 = pd.read_csv('D:/Technocolabs/Individual Project/Datasets/3year_cleaned.csv')
df4 = pd.read_csv('D:/Technocolabs/Individual Project/Datasets/4year_cleaned.csv')
df5 = pd.read_csv('D:/Technocolabs/Individual Project/Datasets/5year_cleaned.csv')


# # Merging 5 years dataframes into a single dataframe

# In[3]:


data = pd.concat([df1,df2,df3,df4,df5],ignore_index=True)
data


# ## Checking for NULL VALUES

# In[4]:


data.isnull().sum()


# In[5]:


(data.isnull().sum() > 0).sum()


# In[6]:


data.shape


# In[7]:


data.info()


# ## Check for Duplicates

# In[8]:


# Check for duplicates
duplicates = data.duplicated()

# Count the number of duplicates
num_duplicates = duplicates.sum()

# Display the result
print(f"Number of duplicates: {num_duplicates}")
print("Duplicate rows:")
print(data[duplicates])


# In[9]:


data


# # EDA - EXPLORATORY DATA ANALYSIS

# 
# # UNIVARIATE ANALYSIS 

# In[10]:


# Assuming you have a DataFrame called 'df' with the columns for analysis
# Replace 'df' with the actual DataFrame you want to analyze

# Calculate the number of rows and columns for the subplots grid
num_rows = len(data.columns)
num_cols = 1

# Create subplots with the specified grid layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

# Loop through each column and plot its scatter plot on the corresponding subplot
for i, col in enumerate(data.columns):
    ax = axes[i] if num_rows > 1 else axes  # Handle the case of a single row of subplots
    ax.scatter(data.index, data[col], alpha=0.5)
    ax.set_xlabel("Index")
    ax.set_ylabel(col)
    ax.set_title(f"Scatter plot of {col}")

# Adjust the layout and spacing of subplots
plt.tight_layout()

# Show the subplots
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots with one row per column
for column_name in data.columns:
   if data[column_name].dtype in ['int64', 'float64']:
       fig, axes = plt.subplots(1, 3, figsize=(18, 6))

       # Histogram
       axes[0].hist(data[column_name], bins=10)
       axes[0].set_xlabel(column_name)
       axes[0].set_ylabel('Frequency')
       axes[0].set_title(f'Histogram of\n{column_name}')

       # Box Plot
       sns.boxplot(data=data, x=column_name, ax=axes[1])
       axes[1].set_xlabel(column_name)
       axes[1].set_title(f'Box Plot of\n{column_name}')

       # KDE Plot
       sns.kdeplot(data[column_name], ax=axes[2])
       axes[2].set_xlabel(column_name)
       axes[2].set_ylabel('Density')
       axes[2].set_title(f'KDE Plot of\n{column_name}')

       # Adjust layout
       plt.tight_layout()
       plt.show()
       summary_stats = data[column_name].describe()
       print(f'Summary statistics for {column_name}:\n{summary_stats}\n')


# # BI-VARIATE ANALYSIS

# In[12]:


# Assuming you have a DataFrame called 'df' with the columns for analysis, including the target variable 'Status'
# Replace 'df' with the actual DataFrame you want to analyze

# Define the name of the target variable
target_variable_name = 'Status'

# Calculate the number of rows and columns for the subplots grid
num_rows = len(data.columns)   # Excluding the target variable
num_cols = 1

# Create subplots with the specified grid layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5*num_rows))

# Loop through each column (excluding the target variable) and plot its scatter plot against the target
for i, col in enumerate(data.columns):
    if col == target_variable_name:  # Skip the target variable itself
        continue

    ax = axes[i] if num_rows > 1 else axes  # Handle the case of a single row of subplots
    ax.scatter(data[col], data[target_variable_name], alpha=0.5)
    ax.set_xlabel(col)
    ax.set_ylabel(target_variable_name)
    ax.set_title(f"Scatter plot of {col} vs. {target_variable_name}")

# Adjust the layout and spacing of subplots
plt.tight_layout()

# Show the subplots
plt.show()


# # FEATURE ENGINEERING

# ## Altman Z-Score

# #### Altman Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
# 
# - A = working capital / total assets
# - B = retained earnings / total assets
# - C = earnings before interest and tax / total assets
# - D = book value of equity / total liabilities
# - E = sales / total assets
# 
# ###### A score below 1.8 means it's likely the company is headed for bankruptcy, while companies with scores above 3 are not likely to go bankrupt.

# In[4]:


# Select the columns of interest (features and target)
features = ['working capital / total assets', 'retained earnings / total assets', 'EBIT / total assets', 'book value of equity / total liabilities', 'sales / total assets']  # Add more feature columns as needed
target = 'Status'

# Create a new DataFrame with only the selected columns
data_subset = data[features + [target]]

# Calculate the correlation matrix
correlation_matrix = data_subset.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap: Features vs. Target")
plt.show()


# In[14]:


# By using this function we can select correlated features
# it will remove the first feature that is correlated with anything other feature
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[15]:


len(correlation(data,0.8))


# In[16]:


correlated_features = correlation(data, 0.8)
filtered_data = data[list(correlated_features)]
filtered_data


# ### Correlartion

# In[17]:


.import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'df' with columns 'feature1', 'feature2', ..., 'featureN', and 'target'

# Select the columns of interest (features and target)
features = ['gross profit / total assets',
       'gross profit (in 3 years) / total assets',
       'profit on sales / total assets',
       '(short-term liabilities 365) / cost of products sold)',
       '(net profit + depreciation) / total liabilities',
       'equity / fixed assets', 'gross profit / sales',
       'total sales / total assets', 'total assets / total liabilities',
       'constant capital / total assets', '(gross profit + interest) / sales',
       '(gross profit + interest) / total assets',
       'current assets / total liabilities', 'sales / short-term liabilities',
       'net profit / sales', '(sales - cost of products sold) / sales',
       'profit on operating activities / sales',
       '(current assets - inventory - receivables) / short-term liabilities',
       'EBITDA (profit on operating activities - depreciation) / sales',
       'equity / total assets', 'sales / fixed assets',
       '(receivables 365) / sales', 'short-term liabilities / total assets',
       'constant capital / fixed assets',
       '(current assets - inventory) / short-term liabilities',
       'profit on operating activities / total assets', 'EBIT / total assets'] # Add more feature columns as needed
target = 'Status'

# Create a new DataFrame with only the selected columns
data_subset = data[features + [target]]

# Calculate the correlation matrix
correlation_matrix = data_subset.corr()

# Create a heatmap for the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Heatmap: Features vs. Target")
plt.show()


# In[18]:


#Creating a list out of set (correlated_features)
correlated_features1 = list(correlated_features) 

# Prepare data with selected features
X_selected = data[correlated_features1]

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter = 10000)
score = cross_val_score(model, X_selected, data['Status'], cv=5, scoring='accuracy').mean()

# Evaluate the model
print(f'Accuracy of the Logistic Regression model: {score:.4f}')


# # PCA

# ### STANDARDIZATION

# In[8]:


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform((data.loc[:, data.columns != 'Status']))


# ### Explore the variance ratio for all features

# In[9]:


# Specify the number of components you want to retain
num_components = 64  # Adjust as needed

pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(scaled_data)

# Create a DataFrame to store the principal components
principal_df = pd.DataFrame(data=principal_components[:, :2], columns=['PC1', 'PC2'])  # Adjust column names

explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
pca.explained_variance_


# ### For example: Let's visualize the first 2 components

# In[10]:


plt.figure(figsize=(8, 6))
plt.scatter(principal_df['PC1'], principal_df['PC2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.show()


# In[11]:


features_num = len(data.columns)

# Assuming you have a target variable 'y'
model = LogisticRegression(max_iter = 1000)  # Replace with your model
mx_acc = 0
mx_comp = 0
mx_acc2 = 0
mx_comp2 = 0
scores = []
for num_components in range(1, features_num):
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(scaled_data)
    score = cross_val_score(model, X_pca, data['Status'], cv=5, scoring='accuracy').mean()
    if score >= mx_acc:
        mx_acc2 = mx_acc
        mx_comp2 = mx_comp
        mx_acc = score
        mx_comp = num_components
    elif score > mx_acc2:
        mx_acc2 = score
        mx_comp2 = num_components
    scores.append(score)

plt.figure(figsize=(8, 6))
plt.plot(range(1, features_num), scores, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy vs. Number of Principal Components')
plt.show()

print(f'Max accuracy achieved is {mx_acc} with {mx_comp} number of components')
print(f'Second max accuracy achieved is {mx_acc2} with {mx_comp2} number of components')


# ### Explore the cumulative variance ratio for every principal component

# In[12]:


# Calculate cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 65), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio Plot')
plt.show()


# In[13]:


print(cumulative_variance_ratio[1]*100, "%", sep='')


# In[14]:


pca = PCA(n_components=1)
X_pca = pca.fit_transform(scaled_data)
score = cross_val_score(model, X_pca, data['Status'], cv=5, scoring='accuracy').mean()
print(f"Accuracy after reducing the dimenssions to 34 features is {score*100}%")


# # MUTUAL INFORMATION

# #### Calculate Mutual Information

# In[76]:


# Assuming you have a target variable 'y'
y = data['Status']

# Calculate mutual information
mutual_info = mutual_info_classif(data, y, random_state=42)
mutual_info


# ### MI Scores by Mutual Information

# In[77]:


# Create a DataFrame to store the feature names and their corresponding MI scores
feature_names = pd.DataFrame({'Feature': data.columns, 'MI Score': mutual_info})

# Sort the DataFrame by MI Score in descending order
feature_names_sorted = feature_names.sort_values(by='MI Score', ascending=False)

# Print the result
print(feature_names_sorted)


# ### Select Features Based on MI Threshold

# Choose features that have a significant mutual information value with the target variable based on a threshold.

# In[78]:


mutual_info_threshold = 0.01  # Adjust this value based on your preference

# Select features above the mutual information threshold
selected_features = [feature for i, feature in enumerate(data)
                     if mutual_info[i] > mutual_info_threshold and feature != 'Status']
print("The resulted number of features is", len(selected_features), ", which are: ")
selected_features


# #### The Filtered dataset according to Mutual Information

# In[79]:


selected_data = data[selected_features]
selected_data


# In[80]:


# Prepare data with selected features
X_selected = selected_data

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter = 10000)

# Evaluate the model
score = cross_val_score(model, X_selected, data['Status'], cv=5, scoring='accuracy').mean()
print(f'Accuracy of the Logistic Regression model: {score:.4f}')


# # MODEL

# In[141]:


X = data.drop(columns=['Status'])
Y = data['Status']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled.shape)
print(X_test_scaled.shape)

# Selected features by MUTUAL INFORMATION
X_train_mi, X_test_mi, y_train_mi, y_test_mi = train_test_split(selected_data, Y, test_size=0.2, random_state=42)

print(X_train_mi.shape)
print(X_test_mi.shape)

MI_accuracy = []
NO_MI_accuracy = []


# # Regularized Logistic Regression

# ## lbfgs Solver

# In[142]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='lbfgs', penalty='l2', random_state=42, max_iter = 1000)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("Model Accuracy:", model.score(X_test_scaled, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With MI

# In[143]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='lbfgs', penalty='l2', random_state=42, max_iter = 1000)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
print("Model Accuracy:", model.score(X_test_mi, y_test))

MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# ## Liblinear Solver

# #### l1

# In[144]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='liblinear', penalty='l1', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("Model Accuracy:", model.score(X_test_scaled, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With MI

# In[145]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='liblinear', penalty='l1', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
print("Model Accuracy:", model.score(X_test_mi, y_test))

MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)



# #### l2

# In[146]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='liblinear', penalty='l2', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("Model Accuracy:", model.score(X_test_scaled, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With MI

# In[147]:


# Initialize the LogisticRegression model with LIBLINEAR solver, L1 regularization (Lasso), and other parameters
model = LogisticRegression(solver='liblinear', penalty='l2', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
report = classification_report(y_test, y_pred, output_dict=True)
print("Model Accuracy:", model.score(X_test_mi, y_test))

MI_accuracy.append(accuracy_score(y_pred, y_test))

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# ## newton-cg Solver

# In[148]:


# Initialize and train the logistic regression model
model = LogisticRegression(solver='newton-cg', penalty='l2')
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

print(f'Accuracy after using newton-cg with 64 features: {accuracy*100:.4f}%')

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With mi

# In[149]:


# Initialize and train the logistic regression model
model = LogisticRegression(solver='newton-cg', penalty='l2')
model.fit(X_train_mi, y_train)

# Predict on the test set
y_pred = model.predict(X_test_mi)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

MI_accuracy.append(accuracy_score(y_pred, y_test))

print(f'Accuracy after using newton-cg with 1 features after performing PCA: {accuracy*100:.4f}')

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# ## newton-cholesky Solver

# In[150]:


# Define grid of hyperparameters for GridSearchCV
grid_params = {'penalty': ['l2']}

# Initialize the LogisticRegression model
model_without_mi = LogisticRegression(solver='newton-cholesky')

# Initialize GridSearchCV with the model, grid of parameters, and cross-validation
grid_search = GridSearchCV(model_without_mi, grid_params, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best model from the grid search
best_model_without_mi = grid_search.best_estimator_

# Predict using the best model and X_test
Y_pred_best_without_mi = best_model_without_mi.predict(X_test_scaled)

# Calculate accuracy using the best model
accuracy_best_without_mi = accuracy_score(y_test, Y_pred_best_without_mi)
print("Accuracy (best penalty) without mi:", accuracy_best_without_mi)


# #### With MI

# In[151]:


# Define grid of hyperparameters for GridSearchCV
grid_params = {'penalty': ['l2']}

# Initialize the LogisticRegression model
model_with_mi = LogisticRegression(solver='newton-cholesky')

# Initialize GridSearchCV with the model, grid of parameters, and cross-validation
grid_search = GridSearchCV(model_with_mi, grid_params, cv=5)

# Fit the grid search to the data
grid_search.fit(X_train_mi, y_train)

# Get the best model from the grid search
best_model_with_mi = grid_search.best_estimator_

# Predict using the best model and X_test
Y_pred_best_with_mi = best_model_with_mi.predict(X_test_mi)

# Calculate accuracy using the best model
accuracy_best_with_mi = accuracy_score(y_test, Y_pred_best_with_mi)
print("Accuracy (best penalty) with mi:", accuracy_best_with_mi)


# ## SAG

# In[152]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='sag', penalty='l2', random_state=42, max_iter = 10000)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With MI

# In[153]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='sag', penalty='l2', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test)*100)

MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# ## SAGA Solver

# In[154]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='saga', penalty='l2', random_state=42, max_iter = 10000)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# In[155]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='saga', penalty='l1', random_state=42, max_iter = 10000)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# In[156]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='saga', penalty='elasticnet', random_state=42, max_iter = 10000, l1_ratio = 0.5)

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test))

NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# #### With MI

# In[157]:


# Initialize the LogisticRegression model with sag solver, L2 regularization (Lasso), and other parameters
model = LogisticRegression(solver='saga', penalty='l2', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test)*100)

MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# In[158]:


# Initialize the LogisticRegression model with sag solver, L1 regularization, and other parameters
model = LogisticRegression(solver='saga', penalty='l1', random_state=42)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test)*100)

MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# In[159]:


# Initialize the LogisticRegression model with sag solver, elasticnet regularization, and other parameters
model = LogisticRegression(solver='saga', penalty='elasticnet', random_state=42, l1_ratio = 0.5)

# Fit the model to the scaled training data
model.fit(X_train_mi, y_train)

y_pred = model.predict(X_test_mi)

# Generate a classification report
print("Model Accuracy:", accuracy_score(y_pred, y_test)*100)

MI_accuracy.append(accuracy_score(y_pred, y_test))

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0)
# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)


# ## Random Forest

# In[160]:


# Creating a Random Forest Classifier with different criterion options
criteria = ["gini", "entropy", "log_loss"]

for criterion_option in criteria:
    # Creating the Random Forest Classifier with the specified criterion
    clf = RandomForestClassifier(criterion=criterion_option, random_state=42)

    # Fitting the classifier to the training data
    clf.fit(X_train_scaled, y_train)

    # Making predictions on the test set
    y_pred = clf.predict(X_test_scaled)

    NO_MI_accuracy.append(accuracy_score(y_pred, y_test))

    # Calculating and printing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Criterion: {criterion_option}")
    print(f"Accuracy: {accuracy*100:.5f}\n")


# #### With MI

# In[161]:


# Creating a Random Forest Classifier with different criterion options
criteria = ["gini", "entropy", "log_loss"]

for criterion_option in criteria:
    # Creating the Random Forest Classifier with the specified criterion
    clf = RandomForestClassifier(criterion=criterion_option, random_state=42)

    # Fitting the classifier to the training data
    clf.fit(X_train_mi, y_train)

    # Making predictions on the test set
    y_pred = clf.predict(X_test_mi)

    MI_accuracy.append(accuracy_score(y_pred, y_test))

    # Calculating and printing the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Criterion: {criterion_option}")
    print(f"Accuracy: {accuracy*100:.5f}\n")


# # EVALUATING THE MODELS

# In[176]:


algorithms = ['lbfgs', 'liblinear-l1', 'liblinear-l2', 'newton-cg', 'sag', 'saga-l2', 'saga-l1', 'saga-elasticnet', 'gini', 'entropy', 'log_loss']
x = np.arange(len(algorithms))
width = 0.4 

fig, ax = plt.subplots(figsize=(25,10))
rects1 = ax.bar(x - width/2, MI_accuracy, width, label='Using MI')
rects2 = ax.bar(x + width/2, NO_MI_accuracy, width, label='Without using MI')

ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Algorithm')
ax.set_xticks(x, algorithms)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.title("ACCURACY OF EACH MODEL", size = 40)
plt.show()


# In[ ]:




