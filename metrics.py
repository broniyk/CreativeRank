import pandas as pd

def hit_rate_at_k(preds: pd.DataFrame, y_true: pd.DataFrame, k: int):
    top_k_preds = (
        preds
        .sort_values(
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
        how="inner"
    )
    # Calculate hit rate per experiment
    total_per_experiment = y_true.groupby("EXPERIMENT_ID").size()
    hits_per_experiment = hits.groupby("EXPERIMENT_ID").size()
    # Fill missing experiments (with no hits) with 0
    hits_per_experiment = hits_per_experiment.reindex(total_per_experiment.index, fill_value=0)
    hit_rate_per_experiment = hits_per_experiment / total_per_experiment
    
    test_variations_per_experiment = preds.groupby("EXPERIMENT_ID")["VARIATION_ID"].nunique()
    base_hit_rate = 1 / test_variations_per_experiment
    uplift_hit_rate = (hit_rate_per_experiment - base_hit_rate) / base_hit_rate * 100
    
    # Return mean hit rate over all experiments
    return hit_rate_per_experiment.mean(), uplift_hit_rate.mean()

def mrr_at_k(preds: pd.DataFrame, y_true: pd.DataFrame, k: int):
    """
    Compute mean reciprocal rank (MRR) at k for each experiment,
    and return the mean MRR across all experiments.
    """
    # Sort and get top k per EXPERIMENT_ID, RECIPIENT_ID
    preds_sorted = (
        preds
        .sort_values(["EXPERIMENT_ID", "RECIPIENT_ID", "PRED"], ascending=[True, True, False])
        .groupby(["EXPERIMENT_ID", "RECIPIENT_ID"])
        .head(k)
    )

    # Compute rank for each RECIPIENT_ID in each EXPERIMENT_ID
    preds_sorted["RANK"] = preds_sorted.groupby(["EXPERIMENT_ID", "RECIPIENT_ID"]).cumcount() + 1

    # Only keep relevant columns for merge
    temp = preds_sorted[["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID", "RANK"]]

    # Join with y_true (ground truth click) to get the relevant rank for each true event
    gt_ranks = pd.merge(
        y_true[["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"]],
        temp,
        on=["EXPERIMENT_ID", "RECIPIENT_ID", "VARIATION_ID"],
        how="left"
    )

    # Rank will be nan if not in top-k
    gt_ranks["RECIPROCAL_RANK"] = gt_ranks["RANK"].apply(lambda x: 1 / x if pd.notnull(x) else 0)

    # Compute mean MRR for each experiment
    mrr_per_experiment = gt_ranks.groupby("EXPERIMENT_ID")["RECIPROCAL_RANK"].mean()

    mrr_base = preds.groupby(["EXPERIMENT_ID"])["VARIATION_ID"].nunique().apply(lambda x: sum(1/i for i in range(1, x + 1))/x if x > 0 else 0)
    uplift_mrr = (mrr_per_experiment - mrr_base) / mrr_base * 100
    # Return the mean across all experiments (or you could return the per-experiment series)
    return mrr_per_experiment.mean(), uplift_mrr.mean()