import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Load the cleaned data
data = pd.read_csv('cleaned_telco_churn_data.csv')

# Plot 1: Distribution of Churn
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.savefig('churn_distribution_rf.png')
plt.show()

# Plot 2: Expanded Correlation Heatmap
plt.figure(figsize=(18, 14))  # Increased size for better visibility
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, annot_kws={"size": 8})
plt.title('Expanded Correlation Heatmap', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('expanded_correlation_heatmap_rf.png')
plt.show()

# Split the data for modeling
X = data.drop(columns=['Churn'])
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
rf_preds = rf.predict(X_test)

# Evaluate the model
rf_report = classification_report(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# Print the results
print("Random Forest Classification Report:")
print(rf_report)
print(f"Random Forest AUC-ROC: {rf_auc:.4f}")

# Plot 3: Feature Importance based on Random Forest
plt.figure(figsize=(10, 6))
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance_sorted = importance.sort_values(ascending=False)
importance_sorted.plot(kind='bar')
plt.title('Feature Importance (Random Forest)')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()

# Print out the file names where the plots are saved
print("Plots saved as 'churn_distribution_rf.png', 'expanded_correlation_heatmap_rf.png', and 'feature_importance_rf.png'")
