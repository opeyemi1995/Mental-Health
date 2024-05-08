import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectFromModel
 
# Load the dataset
data = pd.read_csv("b_depressed.csv")
 
# Data Exploration
print("Dataset shape:", data.shape)
print("Missing values:\n", data.isnull().sum())
 
# Drop missing data
data.dropna(inplace=True)
 
# Identify outliers
outliers = IsolationForest().fit_predict(data.drop(columns=['depressed', 'Survey_id', 'Ville_id']))
outliers_mask = outliers != -1
print("Number of outliers:", outliers_mask.sum())
 
# Iterate over each feature and plot boxplot separately
for column in data.drop(columns=['depressed', 'Survey_id', 'Ville_id']).columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[column], orient='h')
    plt.title(f'Boxplot of {column}')
    plt.show()
 
# Bar charts
plt.figure(figsize=(12, 6))
data['depressed'].value_counts().plot(kind='bar')
plt.title('Distribution of Depressed')
plt.xlabel('Depressed')
plt.ylabel('Count')
plt.show()
 
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='depressed')
plt.title('Distribution of Depressed')
plt.xlabel('Depressed')
plt.ylabel('Count')
# Calculate percentages
total_count = data['depressed'].count()
for p in plt.gca().patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height / total_count:.1%}', ha='center')
plt.show()
 
# Pie Chart for Married vs. Not Married and Depressed
married_counts = data[data['Married'] == 1]['depressed'].value_counts()
not_married_counts = data[data['Married'] == 0]['depressed'].value_counts()
 
plt.figure(figsize=(10, 5))



plt.subplot(1, 2, 1)
plt.pie(married_counts, labels=married_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Married and Depressed')
 
plt.subplot(1, 2, 2)
plt.pie(not_married_counts, labels=not_married_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Not Married and Depressed')
 
plt.show()
 
# Ring Chart for Marital Status vs. Education Level and Depressed
married_below_10 = data[(data['Married'] == 1) & (data['education_level'] <= 10)]['depressed'].value_counts()
married_above_10 = data[(data['Married'] == 1) & (data['education_level'] > 10)]['depressed'].value_counts()
 
plt.figure(figsize=(10, 5))
 
plt.subplot(1, 2, 1)
plt.pie(married_below_10, labels=married_below_10.index, autopct='%1.1f%%', startangle=140)
plt.title('Married (Education <= 10) and Depressed')
 
plt.subplot(1, 2, 2)
plt.pie(married_above_10, labels=married_above_10.index, autopct='%1.1f%%', startangle=140)
plt.title('Married (Education > 10) and Depressed')
 
plt.show()
 
 
# Correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(data.drop(columns=['Survey_id', 'Ville_id']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
 
# Feature Engineering
data['asset'] = data['gained_asset'] + data['durable_asset'] + data['save_asset']
data['expenses'] = data['living_expenses'] + data['other_expenses'] + data['farm_expenses'] + data['labor_primary']
data['income'] = data['incoming_salary'] + data['incoming_own_farm'] + data['incoming_business'] + data['incoming_no_business'] + data['incoming_agricultural']
data['investment'] = data['lasting_investment'] + data['no_lasting_investmen']
 
# Remove original features that have been summed up
data.drop(columns=['gained_asset', 'durable_asset', 'save_asset', 'living_expenses', 'other_expenses', 'farm_expenses', 'labor_primary', 'incoming_salary', 'incoming_own_farm', 'incoming_business', 'incoming_no_business', 'incoming_agricultural', 'lasting_investment', 'no_lasting_investmen'], inplace=True)
 
# Remove Survey_id and Ville_id from the dataset
data.drop(columns=['Survey_id', 'Ville_id'], inplace=True)
 
# Identify outliers
outliers = IsolationForest().fit_predict(data)
outliers_mask = outliers != -1
print("Number of outliers:", outliers_mask.sum())
 
# Iterate over each feature and plot boxplot separately
for column in data.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[column], orient='h')
    plt.title(f'Boxplot of {column}')
    plt.show()
 
# Bar charts
plt.figure(figsize=(12, 6))
data['depressed'].value_counts().plot(kind='bar')

plt.title('Distribution of Depressed')
plt.xlabel('Depressed')
plt.ylabel('Count')
plt.show()
 
# Correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
 
# Split data into features and target
X = data.drop(columns=['depressed'])
y = data['depressed']
 
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# Classifier models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Machine": SVC(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Gaussian Process Classifier": GaussianProcessClassifier()
}
 
# Create ExcelWriter object
with pd.ExcelWriter('model_evaluation_results.xlsx') as writer:
    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_excel(writer, sheet_name=name)
        
        # Plot feature importance if applicable
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance for {name}')
            plt.show()



