# SC_DS_TASK3
# 🤖 Bank Marketing Dataset - Decision Tree Classification

This repository contains my solution for **Task 3** assigned during my internship at **Skillcraft Technology**. The objective was to apply machine learning to the **Bank Marketing Dataset**, using a **Decision Tree Classifier** to predict whether a customer will subscribe to a term deposit.

## 📂 Dataset
- **Source:** UCI Machine Learning Repository - [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **File:** `bank-full.csv` (extracted from `bank.zip`)
- **Target Variable:** `y` (binary: yes/no)

## 🧰 Technologies & Libraries Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📊 Key Steps Performed

### 🧹 Data Preprocessing
- Extracted ZIP file programmatically.
- Loaded CSV file with `;` delimiter.
- Handled categorical variables using **Label Encoding**.

### 🔎 Exploratory Analysis
- Checked for null/missing values.
- Printed dataset info and head.

### 🤖 Model Building
- Split dataset into training and test sets (80/20).
- Trained a **Decision Tree Classifier** using `scikit-learn`.

### ✅ Evaluation Metrics
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix** (visualized using Seaborn heatmap)

### 📌 Feature Insights
- Calculated and plotted **feature importance**.
- Visualized the trained **Decision Tree**.

## 📉 Sample Output Visuals
- 🔷 Confusion Matrix
- 🔷 Feature Importance Bar Chart
- 🔷 Decision Tree Plot


