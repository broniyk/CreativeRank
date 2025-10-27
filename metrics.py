import mlflow
import numpy as np
import pandas as pd


def hit_rate_at_k(preds: pd.DataFrame, y_true: pd.DataFrame, k: int, **kwargs):
    top_k_preds = (
        preds.sort_values(
            ["EXPERIMENT_ID", "RECIPIENT_ID", "PRED"], ascending=[True, True, False]
        )
        .groupby(["EXPERIMENT_ID", "RECIPIENT_ID"])
        .head(k)
    )
    # Merge y_true with top_k_preds to find hits
    hits = pd.merge(
        y_true,
        top_k_preds,
        on=["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"],
        how="inner",
    ) 

    if mlflow.active_run() is not None:
        prefix = ''
        if 'prefix' in kwargs:
            prefix = kwargs["prefix"]
        mlflow.log_metric(f"{prefix}hit_rate_at_{k}", hits.shape[0] / y_true.shape[0])
    return hits.shape[0] / y_true.shape[0]

def hit_rate_at_k_per_experiment(preds: pd.DataFrame, y_true: pd.DataFrame, k: int, **kwargs):
    top_k_preds = (
        preds.sort_values(
            ["EXPERIMENT_ID", "RECIPIENT_ID", "PRED"], ascending=[True, True, False]
        )
        .groupby(["EXPERIMENT_ID", "RECIPIENT_ID"])
        .head(k)
    )
    # Merge y_true with top_k_preds to find hits
    hits = pd.merge(
        y_true,
        top_k_preds,
        on=["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"],
        how="inner",
    )
    # Calculate hit rate per experiment
    total_per_experiment = y_true.groupby("EXPERIMENT_ID").size()
    hits_per_experiment = hits.groupby("EXPERIMENT_ID").size()
    # Fill missing experiments (with no hits) with 0
    hits_per_experiment = hits_per_experiment.reindex(
        total_per_experiment.index, fill_value=0
    )
    hit_rate_per_experiment = hits_per_experiment / total_per_experiment

    test_variations_per_experiment = preds.groupby("EXPERIMENT_ID")[
        "VARIATION_ID"
    ].nunique()
    base_hit_rate = 1 / test_variations_per_experiment
    uplift_hit_rate = (hit_rate_per_experiment - base_hit_rate) / base_hit_rate * 100

    if mlflow.active_run() is not None:
        step = None
        prefix = ''
        if 'step' in kwargs:
            step = kwargs["step"]
        if 'prefix' in kwargs:
            prefix = kwargs["prefix"]
        mlflow.log_metric(f"{prefix}avg_hit_rate_at_{k}", hit_rate_per_experiment.mean(), step=step)
        mlflow.log_metric(f"{prefix}std_hit_rate_at_{k}", hit_rate_per_experiment.std(), step=step)
        mlflow.log_metric(f"{prefix}avg_uplift_of_hit_rate_at_{k}", uplift_hit_rate.mean(), step=step)
    # Return mean hit rate over all experiments
    return hit_rate_per_experiment.mean(), uplift_hit_rate.mean()


def compute_mrr_values(preds: pd.DataFrame, y_true: pd.DataFrame, k: int):
    # Sort and get top k per EXPERIMENT_ID, RECIPIENT_ID
    preds_sorted = (
        preds.sort_values(
            ["EXPERIMENT_ID", "RECIPIENT_ID", "PRED"], ascending=[True, True, False]
        )
        .groupby(["EXPERIMENT_ID", "RECIPIENT_ID"])
        .head(k)
    )

    # Compute rank for each RECIPIENT_ID in each EXPERIMENT_ID
    preds_sorted["RANK"] = (
        preds_sorted.groupby(["EXPERIMENT_ID", "RECIPIENT_ID"]).cumcount() + 1
    )

    # Only keep relevant columns for merge
    temp = preds_sorted[["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID", "RANK"]]

    # Join with y_true (ground truth click) to get the relevant rank for each true event
    gt_ranks = pd.merge(
        y_true[["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"]],
        temp,
        on=["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"],
        how="left",
    )

    # Rank will be nan if not in top-k
    gt_ranks["RECIPROCAL_RANK"] = gt_ranks["RANK"].apply(
        lambda x: 1 / x if pd.notnull(x) else 0
    )
    return gt_ranks


def mrr_at_k(preds: pd.DataFrame, y_true: pd.DataFrame, k: int, **kwargs):
    gt_ranks = compute_mrr_values(preds, y_true, k)
    if mlflow.active_run() is not None:
        step = None
        prefix = ''
        if 'prefix' in kwargs:
            prefix = kwargs["prefix"]
        if 'step' in kwargs:
            step = kwargs["step"]
        mlflow.log_metric(f"{prefix}_mrr_at_{k}", gt_ranks["RECIPROCAL_RANK"].mean(), step=step)
    return gt_ranks["RECIPROCAL_RANK"].mean()


def mrr_at_k_per_experiment(preds: pd.DataFrame, y_true: pd.DataFrame, k: int, **kwargs):
    """
    Compute mean reciprocal rank (MRR) at k for each experiment,
    and return the mean MRR across all experiments.
    """
    gt_ranks = compute_mrr_values(preds, y_true, k)

    # Compute mean MRR for each experiment
    mrr_per_experiment = gt_ranks.groupby("EXPERIMENT_ID")["RECIPROCAL_RANK"].mean()

    mrr_base = (
        preds.groupby(["EXPERIMENT_ID"])["VARIATION_ID"]
        .nunique()
        .apply(lambda x: sum(1 / i for i in range(1, x + 1)) / x if x > 0 else 0)
    )
    uplift_mrr = (mrr_per_experiment - mrr_base) / mrr_base * 100
    # Return the mean across all experiments (or you could return the per-experiment series)
    # Only log to mlflow if there's an active run
    if mlflow.active_run() is not None:
        step = None
        prefix = ''
        if 'step' in kwargs:
            step = kwargs["step"]
        if 'prefix' in kwargs:
            prefix = kwargs["prefix"]
        mlflow.log_metric(f"{prefix}avg_mrr_at_{k}", mrr_per_experiment.mean(), step=step)
        mlflow.log_metric(f"{prefix}std_mrr_at_{k}", mrr_per_experiment.std(), step=step)
        mlflow.log_metric(f"{prefix}avg_uplift_of_mrr_at_{k}", uplift_mrr.mean(), step=step)

    return mrr_per_experiment.mean(), mrr_per_experiment.std(), uplift_mrr.mean()


def bootstrap_mrr_at_k(
    preds: pd.DataFrame, y_true: pd.DataFrame, k: int, bootstrap_samples: int = 100, random_state: int = 42, **kwargs
):
    all_mrr_values = compute_mrr_values(preds, y_true, k)
    mrr_values = []
    for _ in range(bootstrap_samples):
        bootstraped_mrr = all_mrr_values.sample(frac=1, replace=True, random_state=random_state)
        mrr_values.append(bootstraped_mrr["RECIPROCAL_RANK"].mean())

    avg_mrr = np.array(mrr_values).mean()
    std_mrr = np.array(mrr_values).std()

    if mlflow.active_run() is not None:
        step = None
        if 'step' in kwargs:
            step = kwargs["step"]
        mlflow.log_metric(f"bootstrapped_mrr_at_{k}", avg_mrr, step=step)
        mlflow.log_metric(f"bootstrapped_std_mrr_at_{k}", std_mrr, step=step)

    return avg_mrr, std_mrr
