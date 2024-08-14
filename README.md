# CustomerChurn-ML

This repository contains the code and data used for predicting customer churn in the telecom industry using three machine learning models: Logistic Regression, Random Forest, and XGBoost. The project is designed to demonstrate the application of machine learning techniques to identify key factors that contribute to customer churn and to build predictive models that can be integrated into business processes for improving customer retention.

## Project Structure

- **data/**
  - `WA_Fn-UseC_Telco-Customer-Churn.csv`: The original dataset used for training and testing the models.
  - `cleaned_telco_churn_data.csv`: The preprocessed dataset ready for modeling.
  - `processed_telco_churn_data.csv`: The dataset after feature scaling and encoding, used for model training.
- **images/**
  - `churn_distribution.png`: Bar chart showing the distribution of churn in the dataset.
  - `churn_distribution_rf.png`: Churn distribution after applying the Random Forest model.
  - `churn_distribution_xgb.png`: Churn distribution after applying the XGBoost model.
  - `expanded_correlation_heatmap.png`: Correlation heatmap of features in the dataset.
  - `expanded_correlation_heatmap_rf.png`: Correlation heatmap after applying the Random Forest model.
  - `expanded_correlation_heatmap_xgb.png`: Correlation heatmap after applying the XGBoost model.
  - `feature_importance_logreg.png`: Feature importance plot from the Logistic Regression model.
  - `feature_importance_rf.png`: Feature importance plot from the Random Forest model.
  - `feature_importance_xgb.png`: Feature importance plot from the XGBoost model.
- **code/**
  - `logistic.py`: Script for training and evaluating the Logistic Regression model.
  - `preprocess.py`: Script for preprocessing the dataset, including handling missing values, encoding categorical features, and scaling.
  - `randomforest.py`: Script for training and evaluating the Random Forest model.
  - `run_xgboost.py`: Script for training and evaluating the XGBoost model.
- **README.md**: This file, providing an overview of the project and instructions for use.

## Installation and Setup

To run the scripts in this project, you'll need to have Python installed on your system along with the following packages:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
