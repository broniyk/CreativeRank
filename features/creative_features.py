import asyncio
import base64
import glob
import json
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from anthropic import Anthropic
from dotenv import load_dotenv
from joblib import Parallel, delayed
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
from collections import defaultdict, Counter

load_dotenv()


def detect_image_media_type(image_path):
    """
    Detects the actual media type of an image by reading its magic bytes (file signature).

    Parameters:
        image_path (str): Path to the image file

    Returns:
        str: The detected media type (e.g., 'image/jpeg', 'image/png', etc.)
    """
    with open(image_path, "rb") as f:
        header = f.read(12)  # Read first 12 bytes

    # Check magic bytes for different image formats
    if header[:2] == b"\xff\xd8":
        return "image/jpeg"
    elif header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    elif header[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    else:
        # Fallback to extension-based detection
        extension = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_type_map.get(extension, "image/jpeg")


def encode_image(image_path):
    """
    Encodes an image to base64 format for API consumption.

    Parameters:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def analyze_image_with_gpt(
        image_path: str,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        temperature: float = 0.7,
) -> Dict[str, any]:
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
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            **args
        )

        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "response": response.choices[0].message.content,
            "status": "success",
        }

    except Exception as e:
        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "response": None,
            "error": str(e),
            "status": "error",
        }


async def analyze_image_with_claude(
        image_path: str,
        prompt: str,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 300,
        temperature: float = 0.7,
) -> Dict[str, any]:
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
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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

        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "response": response.content[0].text,
            "status": "success",
        }

    except Exception as e:
        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "response": None,
            "error": str(e),
            "status": "error",
        }


async def analyze_image(
        image_path: str,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        temperature: float = 0.7,
) -> Dict[str, any]:
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
    if model.startswith("claude"):
        result = await analyze_image_with_claude(
            image_path, prompt, model, max_tokens, temperature
        )
    elif model.startswith("gpt"):
        result = await analyze_image_with_gpt(
            image_path, prompt, model, max_tokens, temperature
        )
    else:
        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "response": None,
            "error": f"Unknown model type: {model}",
            "status": "error",
        }

    result["model"] = model
    return result


async def extract_creative_features(
        image_paths: List[str],
        prompt_path: str,
        model: str = "gpt-4o",
        max_tokens: int = 300,
        temperature: float = 0.7,
        save_results: bool = True,
        output_file: str = "image_analysis_results.yaml",
        max_concurrent: int = 10,
) -> List[Dict[str, any]]:
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
    # Read the prompt from file
    with open(prompt_path, "r", encoding="utf-8") as pf:
        prompt = pf.read()

    # Remove duplicates and sort
    all_images = sorted(list(set(image_paths)))

    print(f"🖼️ Found {len(all_images)} images to analyze")

    if len(all_images) == 0:
        print("❌ No images found in the specified list")
        return []

    # Process images
    results_dict = dict(
        [
            ("prompt_path", prompt_path),
            ("model", model),
            ("max_tokens", max_tokens),
            ("temperature", temperature),
        ]
    )

    images_to_process = all_images

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_image(image_path: str) -> Dict[str, any]:
        """Process a single image with semaphore for rate limiting."""
        async with semaphore:
            result = await analyze_image(
                image_path=image_path,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # The response is a string that may contain markdown code block formatting.

            # Check if the response is valid JSON before attempting to parse
            try:
                response_text = result["response"]
                # Remove markdown code block markers if present
                if response_text is not None and response_text.startswith("```json"):
                    response_text = response_text[len("```json"):].strip()
                if response_text is not None and response_text.endswith("```"):
                    response_text = response_text[:-3].strip()
                if response_text is not None:
                    response_text = response_text.encode("utf-8").decode("unicode_escape")
                    parsed_json = json.loads(response_text)
                    result["response"] = parsed_json
            except json.JSONDecodeError as e:
                print("❌ The response is not valid JSON. Error:", e)
                print("Response text was:\n", response_text)
                print(f"Skipping image: {os.path.basename(image_path)}")
                return None

            # Print progress for errors
            if result["status"] == "error":
                print(
                    f"❌ Error processing {os.path.basename(image_path)}: {result['error']}"
                )
            
            return result

    # Process all images concurrently with progress bar
    tasks = [process_image(image_path) for image_path in images_to_process]
    results = []
    
    # Use atqdm for async progress tracking
    for coro in atqdm.as_completed(tasks, desc="Analyzing images", total=len(tasks)):
        result = await coro
        if result is not None:
            results.append(result)

    gpt_results = results
    results_dict["results"] = gpt_results

    # Save results
    if save_results:
        if output_file.lower().endswith(".csv"):
            records = []

            for r in gpt_results:
                base_name = r["filename"] if "filename" in r else os.path.basename(r.get("image_path", ""))
                answer_dict = {}
                resp = r.get("response", {})
                # If resp is a dict (from JSON), flatten, otherwise store directly
                if isinstance(resp, dict):
                    for k, v in resp.items():
                        answer_dict[k] = v
                elif resp is not None:
                    # For simple string answer (for single Q etc)
                    answer_dict["answer"] = resp
                # Add error/status info if error
                if r.get("status") == "error":
                    answer_dict["error"] = r.get("error", "")
                answer_dict["status"] = r.get("status", "")
                answer_dict["filename"] = base_name
                answer_dict["image_path"] = r.get("image_path", "")
                records.append(answer_dict)
            # Remove duplicates columns for 'filename' and 'image_path'
            df = pd.DataFrame(records)
            if "filename" in df.columns:
                df = df.set_index("filename")
            # Order columns: put answers first, then status, error, image_path
            main_cols = [c for c in df.columns if c not in ["status", "error", "image_path"]]
            ordered_cols = main_cols + [c for c in ["status", "error", "image_path"] if c in df.columns]
            df = df[ordered_cols]
            df.to_csv(output_file, index=True, encoding="utf-8")
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    results_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
    # Print summary
    successful = sum(1 for r in gpt_results if r["status"] == "success")
    failed = sum(1 for r in gpt_results if r["status"] == "error")
    print(
        f"\n📊 Summary: {successful} successful, {failed} failed out of {len(gpt_results)} total"
    )

    return results_dict



# def analyze_all_images_multi_model(
#         image_directory: str,
#         prompt_path: str,
#         models: Dict[str, Dict],  # Model name -> {"model": ..., "temperature": ..., "max_tokens": ...}
#         output_folder: str = "outputs",
#         output_file: str = "combined_majority.yaml",
#         file_patterns: List[str] = None,
#         batch_size: Optional[int] = None,
#         n_jobs: int = -1,
# ) -> Dict[str, Dict]:
#     """
#     Analyzes all images in a directory using multiple models in parallel with joblib.

#     Parameters:
#         image_directory (str): Directory containing images
#         prompt_path (str): Path to the prompt file to use for each image
#         models (Dict[str, Dict]): Dictionary of model configs,
#             e.g., {'gpt-4o': {'temperature': 0.7, 'max_tokens': 300},
#                    'claude-3-5-sonnet': {'temperature': 0.5, 'max_tokens': 350}}
#         output_folder (str): Folder to save results (default: 'outputs')
#         file_patterns (List[str]): File patterns to match (default: common image formats)
#         batch_size (int): Process images in batches (None = process all)
#         n_jobs (int): Number of parallel jobs (-1 = use all CPUs, 1 = sequential)

#     Returns:
#         Dict[str, Dict]: Dictionary mapping model names to their analysis results
#     """
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     print(f"🚀 Starting multi-model analysis with {len(models)} model(s)")
#     print(f"📁 Results will be saved to: {output_folder}")
#     print(f"🔧 Parallel jobs: {n_jobs if n_jobs > 0 else 'all available CPUs'}")

#     def process_model(name: str, config: Dict) -> tuple:
#         """Process a single model and return the model name and results."""
#         print(f"\n{'=' * 60}")
#         print(f"🤖 Processing {name}")
#         print(f"{'=' * 60}")

#         # Extract per-model settings
#         model = config.get("model", "gpt-4o")
#         temperature = config.get("temperature", 0.7)
#         max_tokens = config.get("max_tokens", 300)

#         # Generate output filename based on model name
#         output_file = os.path.join(output_folder, f"{name}_results.yaml")

#         # Run analysis for this model
#         results = analyze_all_images(
#             image_directory=image_directory,
#             prompt_path=prompt_path,
#             model=model,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             file_patterns=file_patterns,
#             save_results=True,
#             output_file=output_file,
#             batch_size=batch_size,
#         )

#         print(f"✅ Completed analysis with {model}")
#         print(f"💾 Results saved to: {output_file}")

#         return name, results

#     # Process all models in parallel using joblib
#     results_list = Parallel(n_jobs=n_jobs, verbose=10)(
#         delayed(process_model)(name, models[name]) for name in models
#     )

#     # Convert list of tuples to dictionary
#     all_results = dict(results_list)

#     # Print final summary
#     print(f"\n{'=' * 60}")
#     print("🎉 Multi-model analysis complete!")
#     print(f"{'=' * 60}")
#     print(f"📊 Processed {len(models)} model(s):")
#     for name in models:
#         model_results = all_results[name]["results"]
#         successful = sum(1 for r in model_results if r["status"] == "success")
#         print(f"  • {name}: {successful}/{len(model_results)} successful")

#     combine_outputs_with_majority(output_folder, output_file)
#     return all_results


# def combine_outputs_with_majority(outputs_folder="outputs", output_file="combined_majority.yaml"):
#     """
#     Combines the outputs of multiple models into a single output file using majority voting.
#     """

#     model_results_dict = dict()
#     combined_answers = dict()
#     final_majority_results = []

#     # Collect all results for each model
#     for file in os.listdir(outputs_folder):
#         if file.endswith("_results.yaml"):
#             with open(os.path.join(outputs_folder, file), "r") as f:
#                 data = yaml.safe_load(f)
#             model_name = file.split("_results.yaml")[0]
#             results = data["results"]
#             model_results_dict[model_name] = results
#             # Prepare answer lists per image
#             for res in results:
#                 img_name = res["filename"]
#                 if combined_answers.get(img_name, None) is None:
#                     combined_answers[img_name] = defaultdict(list)
#                 for q, a in res["response"].items():
#                     combined_answers[img_name][q].append(a)

#     # Majority voting: select the most common answer for each question per image
#     for img_name, q_dict in combined_answers.items():
#         combined_response = {}
#         for q, values in q_dict.items():
#             if values:
#                 most_common = Counter(values).most_common(1)[0][0]
#                 combined_response[q] = most_common
#             else:
#                 combined_response[q] = None
#         # Compose the full combined result for this image
#         final_majority_results.append({
#             "filename": img_name,
#             "response": combined_response,
#             "status": "success"
#         })

#     # Write to file
#     with open(os.path.join(outputs_folder, output_file), "w") as f:
#         yaml.dump({"results": final_majority_results}, f, allow_unicode=True, sort_keys=False)
#     print(f"✅ Combined majority results written to: {output_file}")