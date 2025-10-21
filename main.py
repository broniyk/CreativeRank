import asyncio
import pandas as pd
import os
from utils import read_prompt
from features.creative_features import extract_creative_features
from pipelines.features_pipeline import run_feature_extraction_pipeline


def main():
    ds = pd.read_csv("/Users/broniy/Desktop/CreativeRank/data/dataset.csv")
    res = run_feature_extraction_pipeline(
        ds,
        creative_prompt="prompt_v3.txt",
        subject_line_prompt="subjectline_v2.txt",
        creative_model="gpt-4o",
        subject_line_model="gpt-5",
        output_file="temp_feats.csv",
    )
    print(res)


if __name__ == "__main__":
    main()
