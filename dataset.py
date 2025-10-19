import os

import pandas as pd


def create_dataset(creative_links_path, subject_lines_path, output_path):
    """
    Create a joined dataset from creative_links.csv and subject_lines.csv, and save it to the given output path.

    Args:
        creative_links_path: Path to the creative_links.csv file
        subject_lines_path: Path to the subject_lines.csv file
        output_path: Path to save the resulting joined CSV file

    Returns:
        pd.DataFrame: Joined dataframe with image_name extracted from cdn_link
    """
    # Read both CSV files
    creative_links_df = pd.read_csv(creative_links_path)
    subject_lines_df = pd.read_csv(subject_lines_path)

    # Join the dataframes on the 'id' field
    joined_df = pd.merge(
        creative_links_df,
        subject_lines_df,
        on="id",
        how="inner",
        suffixes=("_creative", "_subject"),
    )

    # Extract image name from cdn_link
    # The image name is the last part of the URL after the final '/'
    joined_df["image_name"] = joined_df["cdn_link"].apply(
        lambda x: x.split("/")[-1] + ".jpg" if pd.notna(x) else None
    )

    # Save the resulting DataFrame to the specified output path
    joined_df.to_csv(output_path, index=False)

    return joined_df
