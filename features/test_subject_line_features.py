import pandas as pd
from subject_line_features import extract_subject_line_features


def test_extract_subject_line_features():
    df = pd.read_csv("/Users/broniy/Desktop/creative/data/subject_lines.csv").head(20)
    extract_subject_line_features(
        df,
        "/Users/broniy/Desktop/creative/prompts/subjectline_prompt.txt",
        output_path="/Users/broniy/Desktop/creative/outputs/sbl_results_test.csv",
    )


if __name__ == "__main__":
    test_extract_subject_line_features()
