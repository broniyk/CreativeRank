from typing import List

import pandas as pd
import mlflow
from catboost import CatBoostClassifier, CatBoostRanker, Pool


def get_pooled_dataset(data_df: pd.DataFrame, pos_neg_ratio: float = None, cols: List[str] = None, cat_cols: List[str] = None):
    if pos_neg_ratio is not None:
        data_df_pos = data_df[data_df["CLICK"] == 1][["EXPERIMENT_ID", "RECIPIENT_ID"]]
        pos_sz = len(data_df_pos)
        neg_sz = len(data_df[data_df["CLICK"] == 0])
        data_df_neg = data_df[data_df["CLICK"] == 0][["EXPERIMENT_ID", "RECIPIENT_ID"]].sample(frac=pos_sz/neg_sz*pos_neg_ratio, random_state=42)
        pos_neg_df = pd.concat([data_df_pos, data_df_neg]).set_index(["EXPERIMENT_ID", "RECIPIENT_ID"])
        data_df = data_df.set_index(["EXPERIMENT_ID", "RECIPIENT_ID"]).loc[pos_neg_df.index].reset_index()
    data_df_sorted = data_df.sort_values(["EXPERIMENT_ID", "RECIPIENT_ID"])
    group_ids = (
        data_df_sorted[["EXPERIMENT_ID", "RECIPIENT_ID"]]
        .astype(str)
        .agg("_".join, axis=1)
        .astype("category").cat.codes.values
    )
    X, y = data_df_sorted[cols], data_df_sorted["CLICK"]
    cat_features = [
        X.columns.get_loc(col) for col in cat_cols if col in X.columns
    ]
   
    pool = Pool(
        X, label=y, cat_features=cat_features, group_id=group_ids
    )
    return pool, group_ids, X, y

def get_model(type: str, cat_features: List[int], params: dict = None):
    if type == "classifier":
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=cat_features,
            random_seed=42,
            verbose=100,
            # early_stopping_rounds=500,
            use_best_model=True,
        )
        return model
    elif type == "ranker":
        params.update(
            {
                "loss_function": "YetiRank",
                "eval_metric": "MRR",
                "early_stopping_rounds": 500,
                "random_seed": 42,
            }
        )
        model = CatBoostRanker(cat_features=cat_features, verbose=False, **params)
        mlflow.log_params(params)
        return model
