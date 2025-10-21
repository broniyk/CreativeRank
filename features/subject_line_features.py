"""Feature extraction for email subject lines using GPT-4."""

import os
import asyncio
from typing import Dict, Optional
from pathlib import Path
import json

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from utils import json_markdown_to_dict


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
        args = dict()
        if not model.startswith("gpt-5"):
            args["max_tokens"] = max_tokens
            args["temperature"] = temperature
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            **args
        )

        response_text = response.choices[0].message.content.strip()

        return response_text

    except Exception as e:
        return str(e)

async def extract_subject_line_features(
    subject_lines: Dict[str, str],
    prompt: str,
    model: str,
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> pd.DataFrame:
    """
    Internal async function to process subject lines concurrently.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create tasks for all subject lines
    tasks = []
    ids = list(subject_lines.keys())
    
    for sbl_id in ids:
        tasks.append(
            analyze_subject_line_with_gpt(
                subject_line=subject_lines[sbl_id],
                prompt=prompt,
                client=client,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
       
    
    # Process all tasks concurrently with progress bar
    results_list = []
    for result in await async_tqdm.gather(*tasks, desc="Categorizing"):
        results_list.append(result)
    
    # Each item in results_list is a JSON string with feature names and their values.

    results_dict = {}
    # Use the same ordering as subject_lines.keys()
    for sbl_id, result_json in zip(ids, results_list):
        try:
            features = json_markdown_to_dict(result_json)
        except Exception as e:
            # If parsing fails, assign error row
            features = {"error": str(e)}
        results_dict[sbl_id] = features

    # Now, convert to DataFrame with id as index and features as columns
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    return df
