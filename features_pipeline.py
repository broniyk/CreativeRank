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

from features.creative_features import extract_creative_features

load_dotenv()


def run_creative_feature_extraction_pipeline(
    image_paths: List[str],
    prompt_path: str,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
    save_results: bool = True,
    output_file: str = "./outputs/image_analysis_results.csv",
    max_concurrent: int = 10,
) :

    print("=" * 80)
    print("🚀 STARTING IMAGE ANALYSIS PIPELINE")
    print("=" * 80)

    # Step 1: Analyze all images
    print("\n📸 STEP 1: Analyzing images with GPT-4 Vision...")
    print(f"   Output YAML: {output_file}")

    results = asyncio.run(extract_creative_features(
        image_paths=image_paths,
        prompt_path=prompt_path,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        save_results=save_results,
        output_file=output_file,
        max_concurrent=max_concurrent,
    ))

    return results

def run_subject_line_feature_extraction_pipeline(
    subject_lines: List[str],
    prompt_path: str,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
    save_results: bool = True,
    output_file: str = "./outputs/subject_line_analysis_results.csv",
):
    pass