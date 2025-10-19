"""Feature extraction for email subject lines using GPT-4."""

import os
import asyncio
from typing import Dict, Optional
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm


load_dotenv()


async def analyze_subject_line_with_gpt(
    subject_line: str,
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> Dict[str, any]:
    """
    Analyzes a single subject line using GPT-4 API asynchronously.

    Parameters:
        subject_line (str): The email subject line to analyze
        prompt (str): The prompt/instruction for categorization
        client (AsyncOpenAI): The async OpenAI client
        model (str): Model to use (default: gpt-4o)
        max_tokens (int): Maximum tokens in response
        temperature (float): Response creativity (0-1, default 0 for consistency)

    Returns:
        dict: Contains subject line, category, and status
    """
    try:
        # Combine the base prompt with the subject line
        full_prompt = f"{prompt}\n{subject_line}"

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        category = response.choices[0].message.content.strip()

        return {
            "subject_line": subject_line,
            "category": category,
            "status": "success",
        }

    except Exception as e:
        return {
            "subject_line": subject_line,
            "category": None,
            "error": str(e),
            "status": "error",
        }


async def _extract_subject_line_features_async(
    df: pd.DataFrame,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    subject_line_col: str,
    id_col: str,
) -> pd.DataFrame:
    """
    Internal async function to process subject lines concurrently.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create tasks for all subject lines
    tasks = []
    rows_data = []
    
    for _, row in df.iterrows():
        subject_line = row[subject_line_col]
        tasks.append(
            analyze_subject_line_with_gpt(
                subject_line=subject_line,
                prompt=prompt,
                client=client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        rows_data.append(row)
    
    # Process all tasks concurrently with progress bar
    results_list = []
    for result in await async_tqdm.gather(*tasks, desc="Categorizing"):
        results_list.append(result)
    
    # Combine results with row data
    results = []
    for i, result in enumerate(results_list):
        row = rows_data[i]
        results.append({
            id_col: row[id_col],
            subject_line_col: row[subject_line_col],
            "category": result.get("category"),
            "status": result.get("status"),
            "error": result.get("error", None),
        })
    
    return pd.DataFrame(results)


def extract_subject_line_features(
    df: pd.DataFrame,
    prompt_path: str,
    output_path: Optional[str] = None,
    model: str = "gpt-4o",
    max_tokens: int = 50,
    temperature: float = 0.0,
    subject_line_col: str = "subject_line",
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Categorizes email subject lines from a DataFrame using GPT-4 asynchronously.

    Parameters:
        df (pd.DataFrame): DataFrame containing subject lines with columns for id and subject_line
        prompt_path (str): Path to the prompt file (e.g., 'prompts/subjectline_prompt.txt')
        output_path (str, optional): Path to save the results CSV file. If None, doesn't save.
        model (str): Model to use (default: gpt-4o)
        max_tokens (int): Maximum tokens in response
        temperature (float): Response creativity (0-1, default 0 for consistency)
        subject_line_col (str): Name of the column containing subject lines
        id_col (str): Name of the column containing IDs

    Returns:
        pd.DataFrame: DataFrame with columns [id, subject_line, category]
    """

    # Check if it's an absolute path or relative path
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Filter out empty subject lines
    df_filtered = df[df[subject_line_col].notna() & (df[subject_line_col] != "")].copy()
    
    print(f"Analyzing {len(df_filtered)} subject lines with {model}...")

    # Run async processing
    results_df = asyncio.run(
        _extract_subject_line_features_async(
            df=df_filtered,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            subject_line_col=subject_line_col,
            id_col=id_col,
        )
    )
    
    # Print summary
    success_count = (results_df["status"] == "success").sum()
    error_count = (results_df["status"] == "error").sum()
    print(f"\nResults:")
    print(f"  Successful: {success_count}/{len(results_df)}")
    print(f"  Errors: {error_count}/{len(results_df)}")
    
    if error_count > 0:
        print("\nError details:")
        error_rows = results_df[results_df["status"] == "error"]
        for _, row in error_rows.iterrows():
            print(f"  ID {row[id_col]}: {row['error']}")

    # Save to CSV if output path is provided
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save only the essential columns
        output_df = results_df[[id_col, subject_line_col, "category"]].copy()
        output_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    return results_df[[id_col, subject_line_col, "category"]]

