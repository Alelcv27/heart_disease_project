# Personal Key Indicators of Heart Disease (heart_disease_project)

An implementation and comparative analysis of several machine learning models for predicting the onset of heart disease based on the robust CDC 2020 annual survey, developed as a project for the Data Mining course.

## Project Overview

This project implements a complete data mining and prediction pipeline from scratch using Python. By comprehensively analyzing the dataset, engineering features, and handling inherent class imbalances, it achieves high predictive performance through the careful selection and tuning of multiple classification algorithms.

### Key Features

- **Extensive EDA:** Detailed exploratory data analysis to pinpoint the most significant risk factors (BMI, AgeCategory, SleepTime, GeneralHealth).
- **Data Preprocessing:** Advanced categorical encoding (Ordinal, One-Hot, Label Encoding) and rigorous correlation analysis.
- **Imbalance Handling:** Implementation of sampling techniques like **SMOTE** (Oversampling) and **ClusterCentroids** (Undersampling) to balance the target class distribution.
- **Model Comparison:** Training and evaluation of an extensive array of algorithms: Random Forest, ExtraTrees, XGBoost, AdaBoost, Logistic Regression, SGD, KNN, Naive Bayes, MLP (Neural Networks), Bagging, and Voting Classifiers.
- **Hyperparameter Tuning:** Utilization of `GridSearchCV` to optimize the parameters of the most promising models to maximize the F1-Score.

## Technical Architecture

### Data Pipeline

The core logic of the analysis is broken down into structured stages:

- **Cleaning:** Strict removal of duplicates and seamless handling of empty records.
- **Feature Engineering:** Transformation of string-based categories into numerical/ordinal variables suited for model consumption.
- **Balancing:** Altering the training sets dynamically to ensure the minority (Heart Disease) class is represented adequately during training.

### Models Evaluated

- `Random Forest` & `ExtraTrees`: Ensemble methods combining multiple randomized decision trees to limit variance.
- `AdaBoost` & `XGBoost`: Advanced boosting algorithms focused sequentially on correcting errors from weak learners.
- `Logistic Regression` & `SGD`: Baseline probabilistic and linear classifications.
- `MLP Classifier`: Multi-layer neural network designed for complex underlying pattern recognition.
- `Voting Classifier`: Soft voting ensemble combining predictions from KNN, SGD, and Logistic Regression models.

## Performance Comparison (Random Forest)

The Random Forest architecture was benchmarked across different sampling strategies to demonstrate the impact of addressing class imbalance.

| Metric | Imbalanced Dataset | Undersampled Dataset | Oversampled Dataset (SMOTE) |
| --- | --- | --- | --- |
| **Accuracy** | High (~91%) | Lower (~70%) | High (~89%) |
| **Recall (Positive Class)** | Extremely Low | High (~76%) | **Very High (~81%)** |
| **F1 Score** | Low | Moderate | **Highest (~87%)** |

**Insights:**

- **Convergence:** Models trained on the SMOTE-oversampled dataset successfully generalise far better on identifying true positives (Recall) while retaining high Precision.
- **Efficiency:** The Random Forest Classifier tuned via Grid Search presented the absolute best trade-off between identifying sick patients properly and maintaining overall precision.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook / JupyterLab
- Essential ML libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imblearn`

### Required Data Files

The application expects the following files within the working directory:

- `heart_2020_cleaned.csv`: The cleaned CDC annual survey dataset.
- `heartdisease.ipynb`: The main notebook pipeline.


## Configuration

Hyperparameters tuned for the highest performing `Random Forest` (`best_rf_clf`):

- `n_estimators`: 30
- `max_depth`: None
- `min_samples_split`: 2

**Author:** Alessandro La Cava
