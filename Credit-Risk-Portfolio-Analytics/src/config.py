"""Phase 0 path and file configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "home_credit"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_FILES = (
    "application_train.csv", "application_test.csv", "bureau.csv",
    "bureau_balance.csv", "previous_application.csv", "POS_CASH_balance.csv",
    "credit_card_balance.csv", "installments_payments.csv", "sample_submission.csv",
)
DESCRIPTION_FILE = "HomeCredit_columns_description.csv"
EXPECTED_KEYS = {
    "application_train.csv": ("SK_ID_CURR", "TARGET"),
    "application_test.csv": ("SK_ID_CURR",),
    "bureau.csv": ("SK_ID_CURR", "SK_ID_BUREAU"),
    "bureau_balance.csv": ("SK_ID_BUREAU",),
    "previous_application.csv": ("SK_ID_CURR", "SK_ID_PREV"),
    "POS_CASH_balance.csv": ("SK_ID_CURR", "SK_ID_PREV"),
    "credit_card_balance.csv": ("SK_ID_CURR", "SK_ID_PREV"),
    "installments_payments.csv": ("SK_ID_CURR", "SK_ID_PREV"),
}

