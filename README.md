# Pre-Processing-Techniques
To Demonstrate various Data Pre-Processing Techniques using a given data set

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Data dictionary
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda', 'James', np.nan],
    'Age': [28, 22, np.nan, 32, 45, 36],
    'Salary': [5000, 4000, np.nan, 6000, 8000, np.nan],  # Added np.nan for last salary
    'Department': ['HR', 'IT', 'Finance', 'IT', 'Finance', 'HR']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Data:\n", df)

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)
df['Name'].fillna("unknown", inplace=True)

print("\nAfter Handling Missing values:\n", df)

# Label Encoding for Department
label_encoder = LabelEncoder()
df['Dept_label'] = label_encoder.fit_transform(df['Department'])
print("\nAfter Label Encoding Department:\n", df)

# Feature scaling: MinMaxScaler then StandardScaler on 'Age' and 'Salary'
scaler_minmax = MinMaxScaler()
df[['Age', 'Salary']] = scaler_minmax.fit_transform(df[['Age', 'Salary']])

scaler_std = StandardScaler()
df[['Age', 'Salary']] = scaler_std.fit_transform(df[['Age', 'Salary']])

print("\nAfter Feature Scaling:\n", df)

# Prepare features and target
X = df.drop(['Name'], axis=1)
y = df['Name']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\nTraining Features:\n", X_train)
print("\nTesting Features:\n", X_test)
print("\nTraining Labels:\n", y_train)
print("\nTesting Labels:\n", y_test)
