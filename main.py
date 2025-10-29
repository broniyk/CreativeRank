import pandas as pd
from pipelines.features_pipeline import run_feature_extraction_pipeline


def main():
    ds = pd.read_csv("/Users/broniy/Desktop/CreativeRank/data/dataset.csv")
    res = run_feature_extraction_pipeline(
        ds,
        creative_prompt="prompt_v3.txt",
        subject_line_prompt="subjectline_v2.txt",
        creative_model="gpt-4o",
        subject_line_model="gpt-5",
        save_output="run_20251029",
    )
    print(res)


if __name__ == "__main__":
    main()
