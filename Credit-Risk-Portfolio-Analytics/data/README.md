# Data Availability

The data used in this project comes from the Home Credit Default Risk competition hosted on Kaggle.

The original competition files and customer-level derived datasets are not included in this repository. Users who wish to reproduce the analysis must obtain the data directly from Kaggle and accept the applicable competition rules and data-use conditions.

## Data Included in This Repository

The `data/aggregated` directory contains only selected aggregated analytical outputs, such as:

- Risk-level customer distribution
- Aggregated financial risk indicators
- Aggregated credit history indicators
- LightGBM feature importance results

These files do not contain customer identifiers or customer-level records.

## Data Not Included

This repository does not include:

- Original Kaggle competition files
- Cleaned customer-level application data
- Customer-level external credit features
- Training or testing feature matrices
- Customer-level default predictions
- Customer risk portfolio records
- Kaggle customer identifiers

## Reproduction

To reproduce the project:

1. Download the Home Credit Default Risk competition data directly from Kaggle.
2. Place the original files in a local `data/raw` directory.
3. Run the notebooks in the documented sequence.
4. Generated customer-level intermediate and processed files should remain local and should not be publicly redistributed.
