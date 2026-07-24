# Credit Risk Portfolio Analytics

## Project Overview

This project uses the Home Credit Default Risk dataset to build a customer-level credit risk analysis workflow.

The work covers data preparation, feature engineering, model comparison, default probability prediction, customer risk segmentation and portfolio-level analysis. The final stage is an interactive Power BI dashboard, which will be added after the dashboard is completed.

The aim of the project is not only to predict default risk, but also to turn model outputs into a form that can support portfolio monitoring and customer prioritisation.

## Business Questions

The project focuses on several practical questions:

- Which customers have the highest predicted probability of default?
- How is risk distributed across the portfolio?
- What financial characteristics are more common in higher-risk groups?
- How do credit and repayment histories differ across risk levels?
- Which features contribute most to the final model?

## Data

The dataset comes from the Home Credit Default Risk competition on Kaggle.

The main application data is combined with six external data sources:

- Bureau credit history
- Bureau balance history
- Previous applications
- Installment payments
- POS cash balances
- Credit card balances

The original Kaggle files and customer-level derived datasets are not included in this repository. Only selected aggregated outputs are published.

More information is available in [`data/README.md`](data/README.md).

## Project Workflow

The project was completed in the following stages:

1. Cleaned and prepared the application train and test datasets
2. Built customer-level features from six external data sources
3. Merged all feature tables into the final modelling datasets
4. Handled missing values and aligned train and test fields
5. Compared Logistic Regression, Random Forest and LightGBM
6. Selected LightGBM as the final model
7. Generated predicted default probabilities for the test customers
8. Divided customers into four fixed risk levels
9. Produced portfolio, financial and credit history summaries
10. Prepared the outputs for Power BI dashboard development

A more detailed workflow is available in [`docs/project_workflow.md`](docs/project_workflow.md).

## Modelling Dataset

After feature engineering and merging:

- Training set: 307,511 customers and 451 columns
- Test set: 48,744 customers and 450 columns

The train and test datasets contained the same fields apart from the target variable.

History availability indicators were retained so that customers with no external records could be separated from customers whose recorded behaviour was genuinely zero.

## Model Comparison

Three models were compared:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.7803 | 0.2701 |
| Random Forest | 0.7609 | 0.2337 |
| LightGBM | 0.7889 | 0.2929 |

LightGBM achieved the strongest result and was selected as the final model.

At the default threshold of 0.50, the LightGBM model achieved:

| Metric | Result |
|---|---:|
| Precision | 0.1916 |
| Recall | 0.6941 |
| F1 Score | 0.3003 |

The model evaluation is discussed in more detail in [`docs/model_evaluation.md`](docs/model_evaluation.md).

## Risk Segmentation

The final model produced a predicted default probability for each test customer.

Customers were grouped using the following rules:

| Risk Level | Predicted Default Probability |
|---|---|
| Low Risk | Below 0.10 |
| Medium Risk | 0.10 to below 0.30 |
| High Risk | 0.30 to below 0.50 |
| Very High Risk | 0.50 or above |

The resulting portfolio contained:

| Risk Level | Customer Count |
|---|---:|
| Low Risk | 4,097 |
| Medium Risk | 18,671 |
| High Risk | 12,754 |
| Very High Risk | 13,222 |

High Risk and Very High Risk customers together accounted for 25,976 customers, or about 53.3% of the scored portfolio.

The risk rules are documented in [`docs/risk_segmentation_rules.md`](docs/risk_segmentation_rules.md).

## Main Outputs

The project produces the following analytical outputs:

- Customer-level predicted default probabilities
- Customer risk scores and risk levels
- Risk-level portfolio summary
- Financial risk summary
- Credit history summary
- LightGBM feature importance
- Power BI dashboard inputs

Customer-level files are kept locally and are not published in this repository.

## Power BI Dashboard

The planned dashboard contains five pages:

1. Portfolio Overview
2. Risk Distribution
3. Financial Risk Profile
4. Credit History Analysis
5. Model Insights

The Power BI file and dashboard screenshots will be added after the dashboard is completed.

## Repository Structure

```text
Credit-Risk-Portfolio-Analytics/
├── data/
│   ├── README.md
│   └── aggregated/
├── docs/
│   ├── model_evaluation.md
│   ├── project_workflow.md
│   └── risk_segmentation_rules.md
├── notebooks/
├── reports/
│   ├── figures/
│   └── powerbi/
├── .gitignore
├── README.md
└── requirements.txt
