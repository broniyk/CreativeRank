"""Utility functions for the creative analysis project."""

import os
import yaml
import shutil
from pathlib import Path
from typing import List
from PIL import Image
from pathlib import Path
from settings import ROOT_DIR
import matplotlib.pyplot as plt


def read_prompt(prompt_name: str, prompts_dir: str = "prompts") -> str:
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
    if not prompt_name.endswith('.txt'):
        prompt_name = f"{prompt_name}.txt"
    
    # Construct the full path
    prompts_path = ROOT_DIR / Path(prompts_dir)
    prompt_file = prompts_path / prompt_name
    
    # Check if file exists
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file '{prompt_name}' not found in '{prompts_dir}' directory"
        )
    
    # Read and return the file contents
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
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
        ax.axis('off')
        plt.close(fig)  # Prevent the figure from displaying immediately
        return fig
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def get_validation_image_paths(yaml_file: str = "validation_ground_truth.yaml") -> List[str]:
    """
    Read the validation_ground_truth.yaml file and extract the image filenames.
    
    Args:
        yaml_file: Path to the YAML file containing validation ground truth data
        
    Returns:
        List of image filenames from the validation set
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract filenames from the validation_images list
    image_filenames = [item['filename'] for item in data['validation_images']]
    
    return image_filenames


def copy_validation_images(
    yaml_file: str = "validation_ground_truth.yaml",
    source_dir: str = "data/all_images",
    dest_dir: str = "data/validation"
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
