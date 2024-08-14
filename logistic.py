import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the cleaned data
data = pd.read_csv('cleaned_telco_churn_data.csv')

# Plot 1: Distribution of Churn
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.savefig('churn_distribution.png')
plt.show()

# Plot 2: Expanded Correlation Heatmap
plt.figure(figsize=(18, 14))  # Increased size for better visibility
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, annot_kws={"size": 8})
plt.title('Expanded Correlation Heatmap', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('expanded_correlation_heatmap.png')
plt.show()

# Split the data for modeling
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Plot 3: Feature Importance based on Logistic Regression Coefficients
plt.figure(figsize=(10, 6))
importance = pd.Series(log_reg.coef_[0], index=X.columns)
importance_sorted = importance.abs().sort_values(ascending=False)
importance_sorted.plot(kind='bar')
plt.title('Feature Importance (Logistic Regression)')
plt.ylabel('Coefficient Value')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_logreg.png')
plt.show()

# Print out the file names where the plots are saved
print("Plots saved as 'churn_distribution.png', 'expanded_correlation_heatmap.png', and 'feature_importance_logreg.png'")
