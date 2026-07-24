"""Reusable first-pass cleaning for application_train and application_test.

The pipeline never overwrites raw CSVs, never uses TARGET to derive cleaning
rules, and records every transformation and validation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DOCS_DIR, PROJECT_ROOT, RAW_DIR

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

TRAIN_INPUT = RAW_DIR / "application_train.csv"
TEST_INPUT = RAW_DIR / "application_test.csv"
TRAIN_OUTPUT = INTERIM_DIR / "application_train_cleaned.parquet"
TEST_OUTPUT = INTERIM_DIR / "application_test_cleaned.parquet"

TIME_FEATURES = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYMENT_YEARS",
    "DAYS_REGISTRATION": "REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
    "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
}
MISSING_INDICATORS = (
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE",
    "OCCUPATION_TYPE", "AMT_ANNUITY", "AMT_GOODS_PRICE",
)
HOUSING_FIELDS = (
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG",
    "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE", "COMMONAREA_MODE",
    "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", "FLOORSMIN_MODE",
    "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI",
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "TOTALAREA_MODE",
    "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
)
CONTACT_FLAGS = (
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
)


def load_application_data(path: Path) -> pd.DataFrame:
    """Load one original CSV and fail clearly if it is unavailable."""
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    print(f"Loading {path.name}")
    return pd.read_csv(path, low_memory=False)


def validate_schema(train_columns: list[str], test_columns: list[str]) -> None:
    """Require identical feature schemas, with TARGET only in train."""
    if "TARGET" not in train_columns:
        raise ValueError("Training data is missing TARGET")
    if "TARGET" in test_columns:
        raise ValueError("Test data unexpectedly contains TARGET")
    train_features = [c for c in train_columns if c != "TARGET"]
    if train_features != test_columns:
        missing_test = sorted(set(train_features) - set(test_columns))
        extra_test = sorted(set(test_columns) - set(train_features))
        raise ValueError(
            f"Train/test feature schema mismatch; missing in test={missing_test}, "
            f"extra in test={extra_test}, or column order differs"
        )


def record(changes: list[dict[str, Any]], dataset: str, rule: str, field: str,
           before: Any, after: Any, count: int, note: str = "") -> None:
    changes.append({
        "dataset": dataset, "rule": rule, "field": field,
        "before": before, "after": after, "affected_count": int(count), "note": note,
    })


def clean_special_values(df: pd.DataFrame, dataset: str,
                         changes: list[dict[str, Any]]) -> pd.DataFrame:
    """Handle only explicitly approved sentinel and categorical values."""
    anomaly = df["DAYS_EMPLOYED"].eq(365243)
    df["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("int8")
    record(changes, dataset, "create_anomaly_indicator", "DAYS_EMPLOYED_ANOMALY",
           "absent", "int8", anomaly.sum(), "1 where DAYS_EMPLOYED equals 365243")
    before_missing = int(df["DAYS_EMPLOYED"].isna().sum())
    df.loc[anomaly, "DAYS_EMPLOYED"] = np.nan
    record(changes, dataset, "replace_sentinel_with_nan", "DAYS_EMPLOYED",
           before_missing, int(df["DAYS_EMPLOYED"].isna().sum()), anomaly.sum(),
           "365243 replaced; no other value changed")

    categorical = list(df.select_dtypes(include=["object", "category"]).columns)
    for col in categorical:
        missing_output = f"{col}_MISSING"
        original_missing = df[col].isna()
        if missing_output not in df:
            df[missing_output] = original_missing.astype("int8")
            record(changes, dataset, "create_categorical_missing_indicator",
                   missing_output, "absent", "int8", original_missing.sum(),
                   f"Created before filling missing values in {col}")
        xna = df[col].eq("XNA").fillna(False)
        if xna.any():
            df.loc[xna, col] = "Missing_or_Unknown"
            record(changes, dataset, "replace_field_specific_category", col,
                   "XNA", "Missing_or_Unknown", xna.sum())
        missing = df[col].isna()
        if missing.any():
            df.loc[missing, col] = "Missing"
            record(changes, dataset, "fill_categorical_missing", col,
                   "NaN", "Missing", missing.sum(),
                   "Unknown, Not specified and Other preserved")
    return df


def create_time_features(df: pd.DataFrame, dataset: str,
                         changes: list[dict[str, Any]]) -> pd.DataFrame:
    for source, output in TIME_FEATURES.items():
        df[output] = -df[source] / 365.25
        record(changes, dataset, "create_relative_year_feature", output,
               "absent", "float64", df[output].notna().sum(),
               f"Calculated as -{source}/365.25; source retained")
    return df


def create_missing_indicators(df: pd.DataFrame, dataset: str,
                              changes: list[dict[str, Any]]) -> pd.DataFrame:
    for source in MISSING_INDICATORS:
        output = f"{source}_MISSING"
        if output not in df:
            df[output] = df[source].isna().astype("int8")
            record(changes, dataset, "create_missing_indicator", output,
                   "absent", "int8", df[output].sum(), f"Missingness of {source}")
    missing_housing = df[list(HOUSING_FIELDS)].isna().sum(axis=1)
    df["HOUSING_MISSING_COUNT"] = pd.to_numeric(missing_housing, downcast="integer")
    record(changes, dataset, "create_missing_count", "HOUSING_MISSING_COUNT",
           "absent", str(df["HOUSING_MISSING_COUNT"].dtype),
           int(missing_housing.gt(0).sum()), f"Count across {len(HOUSING_FIELDS)} housing fields")
    return df


def optimize_dtypes(df: pd.DataFrame, dataset: str,
                    changes: list[dict[str, Any]]) -> pd.DataFrame:
    """Reduce memory without changing values or filling numeric nulls."""
    binary = [c for c in df if c.startswith("FLAG_DOCUMENT_")]
    binary += list(CONTACT_FLAGS)
    binary += [f"{c}_MISSING" for c in MISSING_INDICATORS]
    binary += [c for c in df if c.endswith("_MISSING")]
    binary += ["DAYS_EMPLOYED_ANOMALY"]
    for col in dict.fromkeys(binary):
        old = str(df[col].dtype)
        if df[col].isna().any():
            df[col] = df[col].astype("Int8")
        else:
            df[col] = df[col].astype("int8")
        if str(df[col].dtype) != old:
            record(changes, dataset, "optimize_dtype", col, old, str(df[col].dtype), len(df))

    for col in df.select_dtypes(include=["object", "category"]).columns:
        old = str(df[col].dtype)
        df[col] = df[col].astype("category")
        record(changes, dataset, "optimize_dtype", col, old, "category", len(df))

    for col in list(df.select_dtypes(include=["int64", "int32", "int16"]).columns):
        if col == "SK_ID_CURR":
            df[col] = df[col].astype("int64")
            continue
        if col == "TARGET":
            old = str(df[col].dtype)
            df[col] = df[col].astype("int8")
            if old != "int8":
                record(changes, dataset, "optimize_dtype", col, old, "int8", len(df))
            continue
        old = str(df[col].dtype)
        converted = pd.to_numeric(df[col], downcast="integer")
        df[col] = converted
        if str(df[col].dtype) != old:
            record(changes, dataset, "optimize_dtype", col, old, str(df[col].dtype), len(df))
    return df


def legality_checks(df: pd.DataFrame, dataset: str) -> list[dict[str, Any]]:
    checks: list[tuple[str, pd.Series, str]] = []
    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"):
        checks.append((f"{col}_le_0", df[col].le(0), "Report only; no rows changed"))
    checks += [
        ("AGE_YEARS_lt_18", df["AGE_YEARS"].lt(18), "Report only"),
        ("AGE_YEARS_gt_100", df["AGE_YEARS"].gt(100), "Report only"),
        ("CNT_CHILDREN_lt_0", df["CNT_CHILDREN"].lt(0), "Report only"),
        ("CNT_FAM_MEMBERS_le_0", df["CNT_FAM_MEMBERS"].le(0), "Report only"),
    ]
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        checks.append((f"{col}_outside_0_1",
                       df[col].notna() & ~df[col].between(0, 1), "Null excluded"))
    for col in [c for c in df if c.startswith("FLAG_DOCUMENT_")] + list(CONTACT_FLAGS):
        checks.append((f"{col}_outside_binary", df[col].notna() & ~df[col].isin([0, 1]), "Null excluded"))
    return [{
        "dataset": dataset, "check_type": "legality", "check": name,
        "status": "PASS" if int(mask.sum()) == 0 else "REVIEW",
        "value": int(mask.sum()), "details": note,
    } for name, mask, note in checks]


def validate_cleaned_data(
    original: pd.DataFrame, cleaned: pd.DataFrame, dataset: str,
    original_target: pd.Series | None,
) -> list[dict[str, Any]]:
    rows = []
    def add(check, ok, value, details=""):
        rows.append({"dataset": dataset, "check_type": "integrity", "check": check,
                     "status": "PASS" if ok else "FAIL", "value": value, "details": details})
    add("row_count_unchanged", len(original) == len(cleaned), len(cleaned))
    add("SK_ID_CURR_no_missing", cleaned.SK_ID_CURR.notna().all(), int(cleaned.SK_ID_CURR.isna().sum()))
    add("SK_ID_CURR_unique", cleaned.SK_ID_CURR.is_unique, int(cleaned.SK_ID_CURR.duplicated().sum()))
    add("SK_ID_CURR_unchanged", original.SK_ID_CURR.equals(cleaned.SK_ID_CURR), "exact ordered equality")
    add("AGE_YEARS_nonnegative_or_missing",
        bool((cleaned.AGE_YEARS.dropna() >= 0).all()), int(cleaned.AGE_YEARS.lt(0).sum()))
    add("EMPLOYMENT_YEARS_nonnegative_or_missing",
        bool((cleaned.EMPLOYMENT_YEARS.dropna() >= 0).all()),
        int(cleaned.EMPLOYMENT_YEARS.lt(0).sum()))
    add("DAYS_EMPLOYED_sentinel_removed",
        not cleaned.DAYS_EMPLOYED.eq(365243).any(), int(cleaned.DAYS_EMPLOYED.eq(365243).sum()))
    if original_target is not None:
        add("TARGET_present_train_only", "TARGET" in cleaned, True)
        add("TARGET_exactly_unchanged",
            original_target.astype("int8").equals(cleaned.TARGET), "exact ordered equality")
        add("TARGET_distribution_unchanged",
            original_target.value_counts().sort_index().to_dict() ==
            cleaned.TARGET.value_counts().sort_index().to_dict(),
            str(cleaned.TARGET.value_counts().sort_index().to_dict()))
    else:
        add("TARGET_absent_in_test", "TARGET" not in cleaned, "TARGET" in cleaned)
    rows.extend(legality_checks(cleaned, dataset))
    failures = [r for r in rows if r["status"] == "FAIL"]
    if failures:
        raise ValueError(f"{dataset} validation failed: {failures}")
    return rows


def clean_application(df: pd.DataFrame, dataset: str,
                      changes: list[dict[str, Any]]) -> pd.DataFrame:
    """Shared train/test cleaning function."""
    df = df.copy()
    df = clean_special_values(df, dataset, changes)
    df = create_time_features(df, dataset, changes)
    df = create_missing_indicators(df, dataset, changes)
    df = optimize_dtypes(df, dataset, changes)
    return df


def save_cleaned_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    df.to_parquet(temporary, engine="pyarrow", compression="snappy", index=False)
    temporary.replace(path)
    print(f"Saved {path} ({path.stat().st_size:,} bytes)")


def process(path: Path, output: Path, dataset: str,
            changes: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    original = load_application_data(path)
    target = original["TARGET"].copy() if "TARGET" in original else None
    before = {
        "dataset": dataset, "rows_before": len(original), "columns_before": len(original.columns),
        "memory_before": int(original.memory_usage(index=True, deep=True).sum()),
        "dtypes_before": original.dtypes.astype(str).value_counts().to_dict(),
    }
    cleaned = clean_application(original, dataset, changes)
    validations = validate_cleaned_data(original, cleaned, dataset, target)
    before.update({
        "rows_after": len(cleaned), "columns_after": len(cleaned.columns),
        "memory_after": int(cleaned.memory_usage(index=True, deep=True).sum()),
        "dtypes_after": cleaned.dtypes.astype(str).value_counts().to_dict(),
    })
    columns = list(cleaned.columns)
    save_cleaned_data(cleaned, output)
    return before, validations, columns


def write_report(summaries: list[dict[str, Any]], changes: pd.DataFrame,
                 validations: pd.DataFrame, schemas_match: bool) -> None:
    lines = [
        "# Application train/test 基础清洗报告", "",
        "本流程对 train/test 调用同一个核心清洗函数；没有使用 TARGET 生成规则、阈值或转换，也没有覆盖原始 CSV。", "",
        "## 1. 清洗规则", "",
        "- DAYS_EMPLOYED=365243：创建异常标志后替换为 NaN。",
        "- 从五个相对天数字段创建 years 特征，同时保留原 DAYS 字段。",
        "- 仅将逐字段出现的 XNA 替换为 Missing_or_Unknown；Unknown、Not specified、Other 保留。",
        "- object/category 原始缺失替换为 Missing；数值缺失不填补。",
        "- 创建七个指定缺失标志和 HOUSING_MISSING_COUNT。",
        "- 优化二元标志、小整数和类别 dtype；金额及连续值保持浮点。",
        "- 合法性问题只记录，不删除或修改记录。", "",
        "## 2. 清洗前后规模与内存", "",
        "| 数据集 | 清洗前 | 清洗后 | 新增列 | 清洗前内存 | 清洗后内存 | 变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        delta = s["memory_after"] / s["memory_before"] - 1
        lines.append(
            f"| {s['dataset']} | {s['rows_before']:,}×{s['columns_before']} | "
            f"{s['rows_after']:,}×{s['columns_after']} | {s['columns_after']-s['columns_before']} | "
            f"{s['memory_before']/2**20:.2f} MiB | {s['memory_after']/2**20:.2f} MiB | {delta:.2%} |"
        )
    lines += ["", "## 3. 新增字段", "",
              ", ".join(f"`{x}`" for x in (
                  "DAYS_EMPLOYED_ANOMALY", *TIME_FEATURES.values(),
                  *(f"{x}_MISSING" for x in MISSING_INDICATORS),
                  "HOUSING_MISSING_COUNT",
              )), "",
              "此外，为全部16个原始类别字段创建对应的 `<FIELD>_MISSING` 标志；与上列重复的 OCCUPATION_TYPE_MISSING 只创建一次。", "",
              "## 4. 特殊值与缺失变化", ""]
    selected = changes[changes["rule"].isin([
        "replace_sentinel_with_nan", "replace_field_specific_category",
        "fill_categorical_missing", "create_missing_indicator",
        "create_categorical_missing_indicator", "create_missing_count",
    ])]
    lines += ["| 数据集 | 规则 | 字段 | 修改前 | 修改后 | 数量 |",
              "|---|---|---|---|---|---:|"]
    for _, r in selected.iterrows():
        lines.append(f"| {r.dataset} | {r.rule} | {r.field} | {r.before} | {r.after} | {r.affected_count:,} |")
    lines += ["", "## 5. 数据类型与内存", "",
              "逐字段 dtype 变化见 application_cleaning_changes.csv。所有 object 字段已转为 category；二元标志优先使用 int8。", "",
              "## 6. 合法性检查", "",
              "| 数据集 | 检查 | 状态 | 异常数 |",
              "|---|---|---|---:|"]
    legal = validations[validations.check_type.eq("legality")]
    for _, r in legal.iterrows():
        lines.append(f"| {r.dataset} | {r.check} | {r.status} | {r.value} |")
    lines += ["", "## 7. 完整性验证", "",
              f"- train/test 清洗后特征字段名、顺序和 dtype 一致：{'是' if schemas_match else '否'}。",
              "- TARGET 仅保留在 train，并执行逐行值及分布完全一致校验。",
              "- SK_ID_CURR 执行非空、唯一和逐行完全一致校验。",
              "- 所有完整性检查明细见 application_cleaning_validation.csv。", "",
              "## 8. 尚未处理的问题", "",
              "- 未删除高缺失字段或异常记录。",
              "- 未填补任何数值缺失。",
              "- Unknown、Not specified、Other 保持原样，等待业务确认。",
              "- 金额极端值、年龄范围外记录和合法性 REVIEW 项仅报告。",
              "- 未编码类别、标准化、选择特征、使用其他历史表或训练模型。"]
    (DOCS_DIR / "application_cleaning_report.md").write_text(
        "\n".join(lines).replace("`", "`") + "\n", encoding="utf-8"
    )


def main() -> int:
    try:
        print("[1/5] Validating raw schemas")
        train_header = list(pd.read_csv(TRAIN_INPUT, nrows=0).columns)
        test_header = list(pd.read_csv(TEST_INPUT, nrows=0).columns)
        validate_schema(train_header, test_header)
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        changes: list[dict[str, Any]] = []
        print("[2/5] Cleaning train")
        train_summary, train_validations, train_columns = process(
            TRAIN_INPUT, TRAIN_OUTPUT, "train", changes
        )
        print("[3/5] Cleaning test")
        test_summary, test_validations, test_columns = process(
            TEST_INPUT, TEST_OUTPUT, "test", changes
        )
        train_features = [c for c in train_columns if c != "TARGET"]
        schemas_match = train_features == test_columns

        print("[4/5] Re-reading parquet outputs for persisted validation")
        train_saved = pd.read_parquet(TRAIN_OUTPUT, engine="pyarrow")
        test_saved = pd.read_parquet(TEST_OUTPUT, engine="pyarrow")
        persisted_match = (
            [c for c in train_saved if c != "TARGET"] == list(test_saved.columns)
            and [str(train_saved[c].dtype) for c in train_saved if c != "TARGET"]
            == [str(test_saved[c].dtype) for c in test_saved]
        )
        schemas_match = schemas_match and persisted_match
        persisted = [
            {"dataset":"combined","check_type":"integrity",
             "check":"persisted_train_test_feature_schema_and_dtypes_match",
             "status":"PASS" if schemas_match else "FAIL",
             "value":schemas_match,"details":"Verified after reading both parquet files"},
            {"dataset":"train","check_type":"integrity","check":"persisted_row_count",
             "status":"PASS" if len(train_saved)==train_summary["rows_before"] else "FAIL",
             "value":len(train_saved),"details":"Re-read parquet"},
            {"dataset":"test","check_type":"integrity","check":"persisted_row_count",
             "status":"PASS" if len(test_saved)==test_summary["rows_before"] else "FAIL",
             "value":len(test_saved),"details":"Re-read parquet"},
        ]
        if any(x["status"] == "FAIL" for x in persisted):
            raise ValueError(f"Persisted output validation failed: {persisted}")

        changes_df = pd.DataFrame(changes)
        validations_df = pd.DataFrame(train_validations + test_validations + persisted)
        changes_df.to_csv(TABLES_DIR / "application_cleaning_changes.csv", index=False)
        validations_df.to_csv(TABLES_DIR / "application_cleaning_validation.csv", index=False)
        write_report([train_summary, test_summary], changes_df, validations_df, schemas_match)
        print("[5/5] Cleaning and validation complete")
        return 0
    except Exception as exc:
        print(f"ERROR: application cleaning failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
