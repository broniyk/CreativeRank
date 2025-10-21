"""Utility functions for the creative analysis project."""

import json
import os
import re
import shutil
from pathlib import Path
from typing import List
import base64
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from settings import PROMPTS_FOLDER


def read_prompt(prompt_name: str) -> str:
    """
    Read a prompt file by name and return its contents as a string.

    Args:
        prompt_name: Name of the prompt file (with or without .txt extension)
        prompts_dir: Directory where prompt files are stored (default: "prompts")

    Returns:
        str: Contents of the prompt file

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file

    Examples:
        >>> prompt = read_prompt("base_prompt")
        >>> prompt = read_prompt("base_prompt.txt")
    """
    # Add .txt extension if not provided
    if not prompt_name.endswith(".txt"):
        prompt_name = f"{prompt_name}.txt"

    # Construct the full path
    prompt_file = PROMPTS_FOLDER / prompt_name

    # Check if file exists
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file '{prompt_name}' not found in '{PROMPTS_FOLDER}' directory"
        )

    # Read and return the file contents
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except IOError as e:
        raise IOError(f"Error reading prompt file '{prompt_name}': {e}")


def display_image_by_path(image_path):
    """
    Display a single image given its file path and return the matplotlib figure.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        matplotlib.figure.Figure: The figure object displaying the image, or None if error.
    """
    try:
        img = plt.imread(image_path)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.set_title(os.path.basename(image_path), fontsize=10)
        ax.axis("off")
        plt.close(fig)  # Prevent the figure from displaying immediately
        return fig
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_validation_image_paths(
    yaml_file: str = "validation_ground_truth.yaml",
) -> List[str]:
    """
    Read the validation_ground_truth.yaml file and extract the image filenames.

    Args:
        yaml_file: Path to the YAML file containing validation ground truth data

    Returns:
        List of image filenames from the validation set
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Extract filenames from the validation_images list
    image_filenames = [item["filename"] for item in data["validation_images"]]

    return image_filenames


def copy_validation_images(
    yaml_file: str = "validation_ground_truth.yaml",
    source_dir: str = "data/all_images",
    dest_dir: str = "data/validation",
) -> None:
    """
    Copy validation images from source directory to destination directory.

    Args:
        yaml_file: Path to the YAML file containing validation ground truth data
        source_dir: Source directory containing all images
        dest_dir: Destination directory for validation images
    """
    # Get the list of validation image filenames
    image_filenames = get_validation_image_paths(yaml_file)

    # Create destination directory if it doesn't exist
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy each image
    source_path = Path(source_dir)
    copied_count = 0
    missing_files = []

    for filename in image_filenames:
        src_file = source_path / filename
        dest_file = dest_path / filename

        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            copied_count += 1
            print(f"Copied: {filename}")
        else:
            missing_files.append(filename)
            print(f"Warning: File not found - {filename}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"Total validation images: {len(image_filenames)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing files: {len(missing_files)}")

    if missing_files:
        print(f"\nMissing files:")
        for filename in missing_files:
            print(f"  - {filename}")
    print(f"{'='*60}")

def json_markdown_to_dict(json_markdown: str):
    """
    Takes a string (markdown block containing JSON) and returns a Python dictionary.
    Strips codeblock markers (e.g., ```json ... ```) if present, then parses JSON.

    Args:
        json_markdown (str): The markdown string containing JSON.

    Returns:
        dict: The parsed Python dictionary.
    """

    # Remove any leading/trailing whitespace
    s = json_markdown.strip()

    # Remove markdown code block "```json" and "```", or just "```"
    codeblock_pattern = r"^```(?:json)?\s*\n(.+?)\n?```$"
    match = re.match(codeblock_pattern, s, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        json_str = s

    return json.loads(json_str)

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
