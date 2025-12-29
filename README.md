# cardiovascular-disease-data-analyser
The first step involves importing the dataset into a Pandas DataFrame. Exploratory Data Analysis (EDA) is performed to understand the structure of the data. We analyze the statistical properties (mean, standard deviation, min/max) to detect any potential outliers.  Target Variable: 



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data (Assuming the CSV name after download)
df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')

# Basic Exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for nulls/duplicates
df.drop_duplicates(inplace=True)
df.fillna(df.mean(), inplace=True)

# Check for nulls/duplicates
df.drop_duplicates(inplace=True)
df.fillna(df.mean(), inplace=True)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Split 70-30
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

param_grid_lr = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)

y_pred_lr = grid_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Feature Importance Plot
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=X.columns)
plt.title('Key Features for Heart Disease')
plt.show()
