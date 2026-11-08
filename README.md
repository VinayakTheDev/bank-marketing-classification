# Bank Marketing Classification — Machine Learning Assignment 2

## Project Overview

This project evaluates five supervised machine learning classification algorithms on the UCI Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.

The models are evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Matthews Correlation Coefficient (MCC).

## Project Links

- **GitHub Repository:** https://github.com/VinayakTheDev/bank-marketing-classification
- **Live Streamlit Application:** https://bank-marketing-classificationgit-7mckqxpez8jyfmvov4o5dy.streamlit.app/

## Problem Statement

The objective is to build and compare multiple classification algorithms for predicting customer subscription to a bank term deposit based on demographic, financial, and campaign-related attributes.

The target variable is `y`:
- `yes`: Customer subscribed to a term deposit.
- `no`: Customer did not subscribe.

## Dataset Description

**Dataset:** Bank Marketing  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information from direct marketing campaigns conducted by a Portuguese banking institution.

| Attribute | Value |
|---|---|
| Total records | 45,211 |
| Original attributes | 17 |
| Predictors used | 15 |
| Target variable | y |
| Missing values | 0 |

The `duration` feature was excluded because it is only known after the customer call and could introduce data leakage.

## Algorithms Implemented

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Preprocessing

- Removed the `duration` feature to avoid target leakage.
- Encoded the categorical variables using One-Hot Encoding.
- Standardized numerical features for Logistic Regression, KNN, and SVM.
- Used stratified train-test splitting to preserve class distribution.
- Training set: 80%
- Testing set: 20%
- Random state: 42

## Model Evaluation

The models were evaluated on the same held-out test dataset containing 9,043 records.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.8361 | 0.3753 | 0.6030 | 0.4627 | 0.7914 | 0.3867 |
| Naive Bayes | 0.8410 | 0.3637 | 0.4792 | 0.4135 | 0.7558 | 0.3277 |
| Logistic Regression | 0.7548 | 0.2662 | 0.6238 | 0.3732 | 0.7722 | 0.2853 |
| Decision Tree | 0.8401 | 0.3087 | 0.2958 | 0.3021 | 0.6040 | 0.2119 |
| KNN | 0.8874 | 0.5529 | 0.1975 | 0.2911 | 0.7027 | 0.2833 |

## Observations

### Support Vector Machine
SVM achieved the highest F1-score, ROC-AUC, and MCC. It provides the best overall balance between precision and recall among the evaluated models.

### Naive Bayes
Naive Bayes achieved the second-highest F1-score and a comparatively balanced precision-recall performance.

### Logistic Regression
Logistic Regression achieved the highest recall. It identifies the largest proportion of customers who subscribed, although it also generates more false positives.

### Decision Tree
Decision Tree achieved relatively high accuracy but lower recall and ROC-AUC for the positive class.

### KNN
KNN achieved the highest accuracy and precision but the lowest recall. It misses a substantial number of customers who actually subscribed.

### Overall Conclusion

The dataset is imbalanced, with approximately 88.3% of customers belonging to the `no` class. Therefore, accuracy alone is not sufficient to evaluate model performance.

Based on F1-score, ROC-AUC, and MCC, SVM performed best among the five evaluated classifiers. Logistic Regression may be preferable when maximizing recall is the primary objective.

## Streamlit Application

The deployed application supports:

- Uploading a test CSV file.
- Selecting a trained classification model.
- Loading saved models without retraining.
- Displaying Accuracy, Precision, Recall, F1, ROC-AUC, and MCC.
- Visualizing the confusion matrix.
- Displaying the classification report.
- Downloading predictions.


## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/VinayakTheDev/bank-marketing-classification.git
cd bank-marketing-classification
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## Author

Vinayak Deshpande
