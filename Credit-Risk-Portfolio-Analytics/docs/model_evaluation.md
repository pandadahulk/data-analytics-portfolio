# Model Evaluation

Three classification models were tested in this project: Logistic Regression, Random Forest and LightGBM.

The comparison was not based only on accuracy because the default class is much smaller than the non-default class. In this situation, a model can appear accurate while still missing many customers who actually default. For that reason, ROC-AUC and PR-AUC were used as the main comparison metrics, together with precision, recall and F1 score.

## Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.7803 | 0.2701 |
| Random Forest | 0.7609 | 0.2337 |
| LightGBM | 0.7889 | 0.2929 |

LightGBM produced the strongest result on both ROC-AUC and PR-AUC, so it was selected as the final model.

The improvement over Logistic Regression was not extremely large, but LightGBM performed better overall and was more suitable for capturing non-linear relationships across the large number of application, credit history and repayment features.

## Default Threshold Results

Using the default classification threshold of 0.50, LightGBM achieved:

| Metric | Result |
|---|---:|
| Precision | 0.1916 |
| Recall | 0.6941 |
| F1 Score | 0.3003 |

The recall of 0.6941 means that the model identified around 69% of the customers who defaulted in the validation data.

Precision was much lower because many customers predicted as high risk did not actually default. This is expected in an imbalanced credit risk problem, especially when the model is set to capture more potential defaulters.

In a real lending environment, the final threshold would normally be chosen based on the cost of false approvals, the cost of unnecessary manual reviews and the organisation's risk tolerance. In this project, the model probability is kept as the main output, while fixed probability ranges are used for portfolio risk segmentation.

## Final Model

After the model comparison was completed, LightGBM was retrained using the full training dataset.

The final model was then used to generate predicted default probabilities for 48,744 customers in the test dataset.

These probabilities were used to create:

- A customer-level risk portfolio
- Four fixed risk levels
- Risk-level portfolio summaries
- Financial risk summaries
- Credit history summaries
- Power BI dashboard inputs

## Limitations

The model results should be interpreted as project-level analytical results rather than a production credit decision system.

Further work would be needed before production use, including probability calibration, threshold optimisation, fairness testing, stability monitoring and validation on newer or external data.
