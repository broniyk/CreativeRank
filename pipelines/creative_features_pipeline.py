import asyncio
import os

from typing import Dict
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

from utils import detect_image_media_type, encode_image, json_markdown_to_dict

load_dotenv()


async def analyze_image_with_gpt(
    image_path: str,
    prompt: str,
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Analyzes a single image using GPT-4 Vision API asynchronously.

    Parameters:
        image_path (str): Path to the image file
        prompt (str): The prompt/question to ask about the image
        api_key (str): OpenAI API key
        model (str): Model to use (default: gpt-4o)
        max_tokens (int): Maximum tokens in response
        temperature (float): Response creativity (0-1)

    Returns:
        dict: Contains image path, prompt, and GPT response
    """
    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Create the message
        # If the model is GPT-5 (contains 'gpt-5' in its name), use max_completion_tokens instead of max_tokens
        args = dict()
        if not model.startswith("gpt-5"):
            args["max_tokens"] = max_tokens
            args["temperature"] = temperature

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            **args,
        )

        return response.choices[0].message.content

    except Exception as e:
        return str(e)


async def analyze_image_with_claude(
    image_path: str,
    prompt: str,
    client: Anthropic,
    model: str = "claude-sonnet-4-5",
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Analyzes a single image using Claude Vision API asynchronously.

    Parameters:
        image_path (str): Path to the image file
        prompt (str): The prompt/question to ask about the image
        model (str): Model to use (default: claude-3-5-sonnet-20241022)
        max_tokens (int): Maximum tokens in response
        temperature (float): Response creativity (0-1)

    Returns:
        dict: Contains image path, prompt, and Claude response
    """

    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Detect actual media type from file content
        media_type = detect_image_media_type(image_path)

        # Create the message
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text

    except Exception as e:
        return str(e)


async def analyze_image(
    image_path: str,
    prompt: str,
    client: AsyncOpenAI | Anthropic,  # Accept client as parameter
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
) -> str:
    """
    Analyzes a single image using either GPT or Claude based on the model parameter asynchronously.

    Parameters:
        image_path (str): Path to the image file
        prompt (str): The prompt/question to ask about the image
        model (str): Model to use (e.g., 'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229')
        max_tokens (int): Maximum tokens in response
        temperature (float): Response creativity (0-1)

    Returns:
        dict: Contains image path, prompt, model used, and response
    """
    result = None

    if model.startswith("claude"):
        result = await analyze_image_with_claude(
            image_path, prompt, client, model, max_tokens, temperature
        )
    elif model.startswith("gpt"):
        result = await analyze_image_with_gpt(
            image_path, prompt, client, model, max_tokens, temperature
        )

    return result


async def extract_creative_features(
    creatives: Dict[str, str],
    prompt: str,
    models: str | Dict[str, Dict] = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
    max_concurrent: int = 10,
) -> pd.DataFrame:
    """
    Analyzes a provided list of image paths using GPT-4 Vision API with asynchronous calls.

    Parameters:
        image_paths (List[str]): List of image file paths to analyze
        prompt_path (str): Path to the prompt file to use for each image
        model (str): Model to use (default: gpt-4o)
        max_tokens (int): Maximum tokens per response
        temperature (float): Response creativity (0-1)
        save_results (bool): Whether to save results to file
        output_file (str): Path to save results YAML or CSV
        max_concurrent (int): Maximum number of concurrent API calls (default: 10)

    Returns:
        List[dict]: List of analysis results for each image
    """

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_image(
        image_path: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        """Process a single image with semaphore for rate limiting."""
        async with semaphore:
            response = await analyze_image(
                image_path=image_path,
                prompt=prompt,
                client=client,  # Add this line
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response

    all_models = dict()
    if isinstance(models, str):
        model_config = {
            "model": models,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        all_models[models] = model_config
    else:
        all_models = models

    all_models_results = dict()
    for model_name, model_config in all_models.items():
        print(f"Analyzing images with {model_name}...")
        model = model_config["model"]
        temperature = model_config.get("temperature", 0.7)
        max_tokens = model_config.get("max_tokens", 300)
        
        # Create client once per model type
        if model.startswith("claude"):
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        try:
            # Process all images concurrently with progress bar
            tasks = [
                process_image(image_path, model, temperature, max_tokens)
                for _, image_path in creatives.items()
            ]
            creative_ids = list(creatives.keys())

            results_list = []
            for result in await async_tqdm.gather(*tasks, desc="Analyzing images"):
                results_list.append(result)

            results_dict = {}
            for creative_id, result_json in zip(creative_ids, results_list):
                try:
                    features = json_markdown_to_dict(result_json)
                except Exception as e:
                    features = {"error": str(e)}
                results_dict[creative_id] = features
            
            df = pd.DataFrame.from_dict(results_dict, orient="index")
            df.index.name = "id"
            all_models_results[model_name] = df
        finally:
            # Close the client after processing all images for this model
            if hasattr(client, 'close'):
                await client.close()

    return all_models_results
