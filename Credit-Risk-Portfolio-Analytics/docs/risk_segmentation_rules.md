# Risk Segmentation Rules

The final LightGBM model produces a predicted probability of default for each customer. To make the model output easier to use in portfolio analysis, the customers are grouped into four risk levels.

| Risk Level | Predicted Default Probability |
|---|---|
| Low Risk | Below 0.10 |
| Medium Risk | 0.10 to below 0.30 |
| High Risk | 0.30 to below 0.50 |
| Very High Risk | 0.50 or above |

These thresholds are used consistently throughout the project and are not changed during the Power BI stage.

Low Risk customers have a predicted default probability below 10%. They form the lowest-risk part of the portfolio and would normally require less attention than the other groups.

Medium Risk customers fall between 10% and 30%. This group is not necessarily problematic, but it may need closer monitoring when other warning signs are also present, such as a high credit-to-income ratio or weaker repayment history.

High Risk customers have a predicted probability between 30% and 50%. These customers are more likely to require additional review before any lending decision is made.

Very High Risk customers have a predicted default probability of 50% or above. They are the highest-priority group for manual review and further risk checks.

The scored test portfolio contains 48,744 customers.

| Risk Level | Customer Count |
|---|---:|
| Low Risk | 4,097 |
| Medium Risk | 18,671 |
| High Risk | 12,754 |
| Very High Risk | 13,222 |

High Risk and Very High Risk customers together account for 25,976 customers, or about 53.3% of the scored portfolio.

The purpose of this segmentation is to support portfolio monitoring and customer prioritisation. It should not be treated as a final lending policy because a real production decision would also need model calibration, cost analysis, regulatory review and further validation.
