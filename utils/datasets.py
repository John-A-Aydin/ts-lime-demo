from typing import List, Literal, TypedDict
import pandas as pd
import numpy as np
import os


class SeriesData(TypedDict):
    train: List[List[float]]
    test: List[List[float]]


def load_and_standardize(file_path):
    df = pd.read_csv(file_path)
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "time" in col_lower:
            col_map[col] = "timestamp"
        elif "series" in col_lower or "id" in col_lower or "part" in col_lower:
            col_map[col] = "series_id"
        elif "value" in col_lower or "data" in col_lower or "target" in col_lower:
            col_map[col] = "value"
    return df.rename(columns=col_map)


def normalize_timestamps(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def prepare_model_data(data):
    model_data = {"train": {}, "test": {}}
    full_df = pd.concat([data["train"], data["test"]])
    for series_id, group in full_df.groupby("series_id"):
        series = group.sort_values("timestamp")
        timestamps = series["timestamp"]
        values = series["value"].astype(np.float32).values
        train_len = len(data["train"][data["train"]["series_id"] == series_id])
        train_vals = values[:train_len]
        test_vals = values[train_len:]
        train_ts = timestamps[:train_len]
        test_ts = timestamps[train_len:]
        if len(train_vals) > 0:
            model_data["train"][series_id] = {
                "timestamps": train_ts,
                "values": train_vals,
            }
        else:
            print(f"⚠️ Skipping {series_id}: Empty train data")
        model_data["test"][series_id] = {
            "timestamps": test_ts,
            "values": test_vals,
        }
    return model_data


def verify_train_test_split(data):
    for series_id in data["train"]["series_id"].unique():
        train_series = data["train"][data["train"]["series_id"] == series_id]
        test_series = data["test"][data["test"]["series_id"] == series_id]
        if not train_series.empty and not test_series.empty:
            last_train = train_series["timestamp"].max()
            first_test = test_series["timestamp"].min()
            if first_test < last_train:
                print(f"⚠️ {series_id}: test starts before train ends")
        elif train_series.empty:
            print(f"⚠️ {series_id}: train empty")
        elif test_series.empty:
            print(f"⚠️ {series_id}: test empty")


def validate_dataset(domain_data):
    for mode in ["train", "test"]:
        print(f"\nValidating {mode} data")
        for series_id, series in domain_data[mode].items():
            if np.isnan(series["values"]).any():
                print(f"❗ NaNs in {series_id} ({mode})")
            if len(series["timestamps"]) > 0 and not np.all(
                np.diff(series["timestamps"].astype("int64")) > 0
            ):
                print(f"❗ Non-monotonic timestamps in {series_id} ({mode})")


def load_dataset(
    selected_dataset: Literal["finance", "power", "pedestrian", "car"],
) -> List[SeriesData]:
    # Load datasets
    dataset: dict | None = None
    try:
        train_df = load_and_standardize(f"./data/{selected_dataset}_train.csv")
        test_df = load_and_standardize(f"./data/{selected_dataset}_test.csv")
        train_df = normalize_timestamps(train_df)
        test_df = normalize_timestamps(test_df)
        dataset = {"train": train_df, "test": test_df}

        print(f"✅ Loaded {selected_dataset}")
    except Exception as e:
        print(f"Failed to load {selected_dataset}: {e}")

    # Sanity check
    # assert dataset, "❌ No datasets loaded. Check dataset selections."

    # Summary
    print("\nSummary:")
    print(f"\n{selected_dataset.upper()}")
    print(f"Train shape: {dataset['train'].shape}")
    print(f"Test shape: {dataset['test'].shape}")
    print(f"Train columns: {list(dataset['train'].columns)}")
    print(f"Unique train series: {dataset['train']['series_id'].nunique()}")
    print(f"Unique test series: {dataset['test']['series_id'].nunique()}")

    # Transform to model-ready format
    verify_train_test_split(dataset)

    train = dataset["train"].copy()
    test = dataset["test"].copy()
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    result = []
    for series_id in train["series_id"].unique():
        # keep rows for this series and sort by timestamp
        s_train = train[train["series_id"] == series_id].sort_values("timestamp")
        s_test = test[test["series_id"] == series_id].sort_values("timestamp")

        # convert the 'value' column to list-of-singleton-lists and drop other columns
        train_values = s_train["value"].apply(lambda x: [x]).tolist()
        test_values = s_test["value"].apply(lambda x: [x]).tolist()

        result.append({"train": train_values, "test": test_values})

    return result
