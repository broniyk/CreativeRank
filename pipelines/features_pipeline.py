"""
Complete pipeline for image analysis and feature encoding.

This pipeline:
1. Analyzes all images using GPT-4 Vision to extract features
2. Converts JSON results to a pandas DataFrame
3. Applies OrdinalEncoder to transform categorical features into numeric values
"""

import os
from typing import Dict, List

import asyncio
import pandas as pd
from dotenv import load_dotenv

from settings import IMAGES_FOLDER, OUTPUTS_FOLDER
from features.creative_features import extract_creative_features
from features.subject_line_features import extract_subject_line_features
from utils import read_prompt

load_dotenv()


def run_feature_extraction_pipeline(
    ds: pd.DataFrame,
    creative_prompt: str,
    subject_line_prompt: str,
    creative_model: str = "gpt-4o",
    subject_line_model: str = "gpt-4o",
    output_file: str = None,
):
    image_paths = dict(zip(ds["id"], ds["image_name"]))
    image_paths = {k: os.path.join(IMAGES_FOLDER, v) for k, v in image_paths.items()}
    creative_prompt = read_prompt(creative_prompt)
    creative_feats = asyncio.run(
        extract_creative_features(
            creatives=image_paths,
            prompt=creative_prompt,
            models=creative_model,
        )
    )[creative_model]

    subject_line_prompt = read_prompt(subject_line_prompt)
    subject_lines = dict(zip(ds["id"], ds["subject_line"]))
    sbl_feats = asyncio.run(
        extract_subject_line_features(
            subject_lines=subject_lines,
            prompt=subject_line_prompt,
            model=subject_line_model,
        )
    )

    feats_df = pd.DataFrame(index=ds["id"])
    feats_df = feats_df.join(creative_feats, how="left")
    feats_df = feats_df.join(sbl_feats, how="left", lsuffix="_CREATIVE", rsuffix="_SBL")
    
    if output_file:
        feats_df.to_csv(OUTPUTS_FOLDER / output_file, index=False)
    return feats_df

