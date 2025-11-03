import pandas as pd
from datetime import datetime
from settings import DATA_FOLDER
from typing import Literal, List


def get_experiment_data():
    clicked_df = pd.read_csv(DATA_FOLDER / "processed/clicked.csv")
    non_clicked_df = pd.read_csv(DATA_FOLDER / "processed/non_clicked_large.csv")

    variations_df = (
        pd.read_csv(DATA_FOLDER / "processed/feats_df.csv")
        .rename(columns={"id": "VARIATION_ID"})
        .fillna("UNK")
    )
    variations_df = variations_df[~variations_df["error"].isna()].drop(
        columns=["error"]
    )
    users_df = (
        pd.concat([clicked_df, non_clicked_df], axis=0)
        .assign(
            CLICK=lambda x: (x["CLICK_COUNT"] > 0).astype(int),
            EXPERIMENT_DATE=lambda x: pd.to_datetime(
                {
                    "year": 2025,
                    "month": x["MONTH"],
                    "day": x["DAY"],
                }
            ),
        )
        .dropna(subset=["CLICK_COUNT"])
        .drop(columns=["RN"])
        .fillna(
            value={
                "TOTAL_ORDERS_VALUE": 0,
                "AVG_ORDER_VALUE": 0,
                "LAST_ORDER_VALUE": 0,
                "COUNTRY": "UNK",
                "REGION": "UNK",
                "LATEST_CLICK_CLIENT_TYPE": "UNK",
                "LATEST_CLICK_CLIENT_NAME": "UNK",
                "LATEST_CLICK_CLIENT_OS_FAMILY": "UNK",
                "FIRST_UTM_SOURCE": "UNK",
                "FIRST_UTM_CONTENT": "UNK",
                "FIRST_UTM_CAMPAIGN": "UNK",
                "LAST_UTM_SOURCE": "UNK",
                "LAST_UTM_CONTENT": "UNK",
                "LAST_UTM_CAMPAIGN": "UNK",
                "CITY": "UNK",
                "TIMEZONE": "UNK",
            }
        )
    )
    # Convert FIRST_ACTIVE_TS to datetime
    users_df["FIRST_ACTIVE_TS_dt"] = pd.to_datetime(users_df["FIRST_ACTIVE_TS"])

    # Compute months between today and FIRST_ACTIVE_TS
    today = pd.Timestamp(datetime.today())

    # Compute years and months difference and convert to total months
    users_df["MONTHS_SINCE_FIRST_ACTIVE"] = (
        today.year - users_df["FIRST_ACTIVE_TS_dt"].dt.year
    ) * 12 + (today.month - users_df["FIRST_ACTIVE_TS_dt"].dt.month)

    users_df = users_df[users_df["VARIATION_ID"].isin(variations_df["VARIATION_ID"])]
    users_df = users_df.drop_duplicates()

    # Print the size of users_df before removal
    print(f"users_df size before removing small experiments: {users_df.shape[0]} rows")
    # Remove experiments with less than 100 participants
    experiment_counts = users_df.groupby("EXPERIMENT_ID")["RECIPIENT_ID"].nunique()
    valid_experiments = experiment_counts[experiment_counts >= 100].index
    users_df = users_df[users_df["EXPERIMENT_ID"].isin(valid_experiments)]
    # Print the size of users_df after removal
    print(f"users_df size after removing small experiments: {users_df.shape[0]} rows")

    variations_per_experimen_df = users_df[
        ["EXPERIMENT_ID", "VARIATION_ID"]
    ].drop_duplicates()

    users_all_variations = pd.merge(
        users_df.drop(columns=["VARIATION_ID"]),
        variations_per_experimen_df,
        how="left",
        left_on="EXPERIMENT_ID",
        right_on="EXPERIMENT_ID",
    )

    # Assign the click to the correct variation
    users_all_variations["CLICK"] = (
        users_all_variations.set_index(
            ["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"]
        )
        .index.map(
            users_df.drop_duplicates(
                ["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"]
            ).set_index(["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"])["CLICK"]
        )
        .fillna(0.5)
    )

    users_all_variations = users_all_variations.merge(
        variations_df,
        left_on=["VARIATION_ID"],
        right_on=["VARIATION_ID"],
        how="left",
    )
    users_all_variations["EXPERIMENT_DATE"] = pd.to_datetime(
        users_all_variations["EXPERIMENT_DATE"]
    )
    return users_all_variations


def split_experiment_train_test_val_data(
    data: pd.DataFrame,
    n_last_test: int = 4,
    n_last_val: int = 2,
    n_last_train: int = None,
):
    data["EXPERIMENT_DATE"] = pd.to_datetime(data["EXPERIMENT_DATE"])

    # Sort unique experiments by date
    experiment_order = (
        data[["EXPERIMENT_ID", "EXPERIMENT_DATE"]]
        .sort_values("EXPERIMENT_DATE")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    test_tail_idx = -n_last_test
    val_tail_idx = -(n_last_test + n_last_val)
    train_tail_idx = (
        -(n_last_test + n_last_val + n_last_train)
        if n_last_train
        else -len(experiment_order)
    )
    # Get last two for test, others for train
    test_experiments = experiment_order.tail(n_last_test)["EXPERIMENT_ID"]
    val_experiments = experiment_order.iloc[val_tail_idx:test_tail_idx]["EXPERIMENT_ID"]

    train_experiments = experiment_order.iloc[train_tail_idx:val_tail_idx][
        "EXPERIMENT_ID"
    ]

    # Select rows for train/test
    train_data = data[data["EXPERIMENT_ID"].isin(train_experiments)]
    # For validation set
    val_data_raw = data[data["EXPERIMENT_ID"].isin(val_experiments)]
    val_data = val_data_raw.groupby(["EXPERIMENT_ID", "RECIPIENT_ID"]).filter(
        lambda g: g["CLICK"].max() == 1
    )

    # For test set
    test_data_raw = data[data["EXPERIMENT_ID"].isin(test_experiments)]
    test_data = test_data_raw.groupby(["EXPERIMENT_ID", "RECIPIENT_ID"]).filter(
        lambda g: g["CLICK"].max() == 1
    )
    return train_data, val_data, test_data

