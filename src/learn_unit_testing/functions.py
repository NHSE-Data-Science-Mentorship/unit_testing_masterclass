import pandas as pd
import numpy as np
import warnings
import time


def train_test_split(df, test_frac):
    # Get the number of test rows
    n_test = int(len(df) * test_frac)

    # Make a shuffled copy of the df
    shuffled = df.sample(frac=1.0)

    # Let's be explicit about which is which to avoid confusion
    test = shuffled.iloc[-n_test:]
    train = shuffled.iloc[:-n_test]

    return train, test


def add_group_mean(df):
    group_means = (
        df.groupby("group")[["value"]]
        .mean()
        .rename(columns={"value": "group_mean"})
    )
    return df.merge(group_means, on="group")


def add_group_median(df):
    group_medians = (
        df.groupby("group")[["value"]]
        .median()
        .rename(columns={"value": "group_median"})
    )
    return df.merge(group_medians, on="group")


def preprocess(df, median_weight=1):

    df = add_group_mean(df)
    df = add_group_median(df)
    df["value_normalised"] = (
        df["value"] + df["group_mean"] - (median_weight * df["group_median"])
    )
    return df.drop(columns=["group_median", "group_mean"])


def add_norm_last_value(df):
    # If the value column is all nans, we want to warn the user
    if df["value"].isna().any():
        warnings.warn("Some values are missing!", UserWarning)

    # If the value column is all nans, we want to raise an exception
    if df["value"].isna().all():
        raise ValueError("All values are missing!")

    _df = df.sort_values("time")
    shifted_cumsum = _df["value"].cumsum().shift()
    _df["norm_last_value"] = _df["value"].shift(fill_value=0)
    _df["norm_last_value"] = _df["norm_last_value"].values / shifted_cumsum

    # Fix the divide by 0 problems.
    _df.loc[shifted_cumsum == 0, "norm_last_value"] = 0

    # The first value is always 0
    _df.at[0, "norm_last_value"] = 0.0

    return _df


class Pipeline:
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        pd.read_csv("some_very_large_file.csv")

    def slow_computation(self, input_data):
        time.sleep(100000)
        return np.random.uniform()

    def run(self):
        df = preprocess(self.data)
        return self.slow_computation(df)
