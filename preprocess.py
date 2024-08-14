import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Convert 'TotalCharges' to numeric, forcing errors to NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Drop rows with missing 'TotalCharges'
data = data.dropna()

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, columns=[
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Encode the target variable 'Churn'
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Save the processed data to a CSV file
processed_file_path = 'processed_telco_churn_data.csv'
data.to_csv(processed_file_path, index=False)

print(f"Processed data saved to {processed_file_path}")
