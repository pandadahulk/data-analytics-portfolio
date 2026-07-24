# Project Workflow

## 1. Application Data Preparation

Clean and prepare the Home Credit application train and test datasets.

Main tasks include:

- Data type checks
- Missing-value review
- Anomaly handling
- Application-level feature engineering
- Train and test column alignment

## 2. External Feature Engineering

Aggregate six external data sources from transaction or account level to customer level:

- Bureau
- Bureau balance
- Previous applications
- Installment payments
- POS cash balances
- Credit card balances

Each output table contains one row per customer.

## 3. Feature Integration

Use the application feature table as the main table and merge all external customer-level feature tables through left joins.

History availability indicators are retained to distinguish customers without external records from customers whose observed behaviour is genuinely zero.

## 4. Modelling Data Preparation

Prepare the final training and testing datasets by:

- Removing shared constant columns
- Filling external feature groups with zero when no corresponding history exists
- Filling categorical missing values with `Missing`
- Filling remaining numerical missing values with training medians
- Checking duplicate customer IDs
- Checking missing and infinite values
- Aligning train and test fields

## 5. Model Development

Train and compare three classification models:

- Logistic Regression
- Random Forest
- LightGBM

Models are evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1 Score

## 6. Final Prediction

Use the final LightGBM model to generate predicted default probabilities for test customers.

## 7. Risk Segmentation

Assign customers to four fixed risk levels:

| Risk Level | Predicted Default Probability |
|---|---|
| Low Risk | Below 0.10 |
| Medium Risk | 0.10 to below 0.30 |
| High Risk | 0.30 to below 0.50 |
| Very High Risk | 0.50 or above |

## 8. Portfolio Analysis

Generate aggregated analytical outputs covering:

- Risk-level customer distribution
- Financial risk profile
- Credit history profile
- Model feature importance

## 9. Dashboard Development

Build a Power BI dashboard with five planned pages:

1. Portfolio Overview
2. Risk Distribution
3. Financial Risk Profile
4. Credit History Analysis
5. Model Insights
