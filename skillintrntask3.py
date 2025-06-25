# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import zipfile
import os

# Path to the updated ZIP file
zip_path = r'C:\Users\Priyanshu\Downloads\bank+marketing\bank.zip'

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(r'C:\Users\Priyanshu\Downloads\bank+marketing\extracted_bank')

# List extracted files
extracted_files = os.listdir(r'C:\Users\Priyanshu\Downloads\bank+marketing\extracted_bank')
print("Extracted Files:", extracted_files)

# Load the extracted CSV file (Check and update the filename if needed)
csv_file = r'C:\Users\Priyanshu\Downloads\bank+marketing\extracted_bank\bank-full.csv'
df = pd.read_csv(csv_file, sep=';')

# Basic Info
print("Dataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nFirst 5 Rows:\n", df.head())

# Encoding categorical variables using Label Encoding
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

print("\nData after encoding:\n", df.head())

# Splitting features and target
X = df.drop('y', axis=1)
y = df['y']  # 'y' is the target variable (0 = No, 1 = Yes)

# Splitting data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Plot Feature Importance
feature_importance.plot(kind='bar', color='teal')
plt.title('Feature Importance')
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()
