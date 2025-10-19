"""
Validation script for feature extraction accuracy.

This script:
1. Loads ground truth manually labeled features
2. Runs the feature extraction pipeline on validation images
3. Compares extracted features with ground truth
4. Calculates accuracy metrics for each feature
5. Generates a detailed accuracy report
"""

import yaml
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from creative_features import analyze_image_with_gpt
from dotenv import load_dotenv

load_dotenv()



def load_ground_truth(yaml_file: str) -> pd.DataFrame:
    """
    Load ground truth validation data from YAML file (validation_ground_truth.yaml).
    
    Parameters:
        yaml_file (str): Path to the ground truth YAML file
        
    Returns:
        pd.DataFrame: DataFrame with ground truth features, indexed by filename
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    records = []
    for item in data['validation_images']:
        record = {'filename': item['filename']}
        record.update(item['ground_truth'])
        records.append(record)
    
    df = pd.DataFrame(records)
    df.set_index('filename', inplace=True)
    return df


def extract_features_for_validation(
    image_directory: str,
    validation_filenames: List[str],
    prompt: str,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7
) -> pd.DataFrame:
    """
    Extract features for validation images using the pipeline.
    
    Parameters:
        image_directory (str): Directory containing images
        validation_filenames (List[str]): List of filenames to process
        prompt (str): Prompt for GPT-4 Vision
        model (str): Model to use
        max_tokens (int): Maximum tokens per response
        temperature (float): Response creativity
        
    Returns:
        pd.DataFrame: DataFrame with extracted features, indexed by filename
    """
    results = []
    
    print(f"\n🔍 Extracting features for {len(validation_filenames)} validation images...")
    
    for filename in validation_filenames:
        image_path = os.path.join(image_directory, filename)
        
        if not os.path.exists(image_path):
            print(f"⚠️  Warning: {filename} not found at {image_path}")
            continue
        
        print(f"   Processing: {filename}")
        
        result = analyze_image_with_gpt(
            image_path=image_path,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if result['status'] == 'success':
            # Parse the response (remove markdown if present)
            response_text = result['response']
            if response_text.startswith("```json"):
                response_text = response_text[len("```json"):].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            try:
                parsed_json = json.loads(response_text)
                record = {'filename': filename}
                record.update(parsed_json)
                results.append(record)
            except json.JSONDecodeError as e:
                print(f"   ❌ Failed to parse JSON for {filename}: {e}")
        else:
            print(f"   ❌ Error extracting features for {filename}: {result.get('error', 'Unknown error')}")
    
    if not results:
        print("❌ No features extracted successfully")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df.set_index('filename', inplace=True)
    return df


def normalize_value(value: str) -> str:
    """
    Normalize a feature value for comparison.
    Handles common variations and formatting differences.
    
    Parameters:
        value (str): Value to normalize
        
    Returns:
        str: Normalized value
    """
    if pd.isna(value):
        return ""
    
    # Convert to string and lowercase
    value = str(value).lower().strip()
    
    # Remove percentage signs and normalize
    value = value.replace('%', '').strip()
    
    # Normalize yes/no variations
    if value in ['yes', 'y', 'true']:
        return 'yes'
    if value in ['no', 'n', 'false']:
        return 'no'
    
    # Normalize unknown variations
    if value in ['unknown', 'unclear', 'not visible', 'n/a', 'na']:
        return 'unknown'
    
    return value


def calculate_exact_match_accuracy(ground_truth: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate exact match accuracy for a single feature.
    
    Parameters:
        ground_truth (pd.Series): Ground truth values
        predicted (pd.Series): Predicted values
        
    Returns:
        float: Accuracy as a percentage
    """
    # Align the series
    aligned_gt, aligned_pred = ground_truth.align(predicted, join='inner')
    
    if len(aligned_gt) == 0:
        return 0.0
    
    # Normalize values
    gt_normalized = aligned_gt.apply(normalize_value)
    pred_normalized = aligned_pred.apply(normalize_value)
    
    # Calculate exact matches
    matches = (gt_normalized == pred_normalized).sum()
    total = len(gt_normalized)
    
    return (matches / total) * 100 if total > 0 else 0.0


def calculate_partial_match_accuracy(ground_truth: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate partial match accuracy for a single feature.
    Useful for features with multiple components (e.g., "brown, long" hair).
    
    Parameters:
        ground_truth (pd.Series): Ground truth values
        predicted (pd.Series): Predicted values
        
    Returns:
        float: Partial match accuracy as a percentage
    """
    # Align the series
    aligned_gt, aligned_pred = ground_truth.align(predicted, join='inner')
    
    if len(aligned_gt) == 0:
        return 0.0
    
    partial_matches = 0
    total = len(aligned_gt)
    
    for gt_val, pred_val in zip(aligned_gt, aligned_pred):
        gt_normalized = normalize_value(str(gt_val))
        pred_normalized = normalize_value(str(pred_val))
        
        if gt_normalized == pred_normalized:
            partial_matches += 1
        else:
            # Check for partial matches (any common words)
            gt_words = set(gt_normalized.replace(',', ' ').split())
            pred_words = set(pred_normalized.replace(',', ' ').split())
            
            # If there's any overlap, count as partial match
            if gt_words & pred_words:
                partial_matches += 0.5
    
    return (partial_matches / total) * 100 if total > 0 else 0.0


def calculate_confusion_matrix(ground_truth: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """
    Create a confusion matrix for a categorical feature.
    
    Parameters:
        ground_truth (pd.Series): Ground truth values
        predicted (pd.Series): Predicted values
        
    Returns:
        pd.DataFrame: Confusion matrix
    """
    # Align the series
    aligned_gt, aligned_pred = ground_truth.align(predicted, join='inner')
    
    # Normalize values
    gt_normalized = aligned_gt.apply(normalize_value)
    pred_normalized = aligned_pred.apply(normalize_value)
    
    # Create confusion matrix
    confusion = pd.crosstab(
        gt_normalized, 
        pred_normalized, 
        rownames=['Ground Truth'], 
        colnames=['Predicted'],
        dropna=False
    )
    
    return confusion


def generate_accuracy_report(
    ground_truth_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
    output_file: str = "validation_report.json"
) -> Dict:
    """
    Generate a comprehensive accuracy report comparing ground truth with extracted features.
    
    Parameters:
        ground_truth_df (pd.DataFrame): Ground truth features
        extracted_df (pd.DataFrame): Extracted features
        output_file (str): Path to save the report
        
    Returns:
        Dict: Accuracy report
    """
    print("\n" + "="*80)
    print("📊 GENERATING ACCURACY REPORT")
    print("="*80)
    
    # Get common columns (questions)
    common_columns = list(set(ground_truth_df.columns) & set(extracted_df.columns))
    common_columns.sort()
    
    # Get common images
    common_images = list(set(ground_truth_df.index) & set(extracted_df.index))
    
    print(f"\n✅ Common images: {len(common_images)}")
    print(f"✅ Common features: {len(common_columns)}")
    
    # Calculate accuracy for each feature
    feature_accuracies = []
    
    for col in common_columns:
        exact_acc = calculate_exact_match_accuracy(
            ground_truth_df[col], 
            extracted_df[col]
        )
        partial_acc = calculate_partial_match_accuracy(
            ground_truth_df[col], 
            extracted_df[col]
        )
        
        # Get unique values for this feature
        all_values = set(ground_truth_df[col].apply(normalize_value)) | \
                     set(extracted_df[col].apply(normalize_value))
        num_categories = len(all_values)
        
        feature_accuracies.append({
            'feature': col,
            'exact_match_accuracy': round(exact_acc, 2),
            'partial_match_accuracy': round(partial_acc, 2),
            'num_categories': num_categories,
            'num_samples': len(common_images)
        })
    
    # Sort by exact match accuracy
    feature_accuracies.sort(key=lambda x: x['exact_match_accuracy'], reverse=True)
    
    # Calculate overall statistics
    overall_exact_acc = np.mean([f['exact_match_accuracy'] for f in feature_accuracies])
    overall_partial_acc = np.mean([f['partial_match_accuracy'] for f in feature_accuracies])
    
    # Create detailed comparison for each image
    image_comparisons = []
    for img in common_images:
        comparison = {
            'filename': img,
            'features': {}
        }
        
        for col in common_columns:
            gt_val = normalize_value(str(ground_truth_df.loc[img, col]))
            pred_val = normalize_value(str(extracted_df.loc[img, col]))
            match = gt_val == pred_val
            
            comparison['features'][col] = {
                'ground_truth': gt_val,
                'predicted': pred_val,
                'match': match
            }
        
        # Calculate per-image accuracy
        matches = sum(1 for f in comparison['features'].values() if f['match'])
        comparison['accuracy'] = round((matches / len(common_columns)) * 100, 2)
        
        image_comparisons.append(comparison)
    
    # Create the report
    report = {
        'summary': {
            'total_images': len(common_images),
            'total_features': len(common_columns),
            'overall_exact_match_accuracy': round(overall_exact_acc, 2),
            'overall_partial_match_accuracy': round(overall_partial_acc, 2)
        },
        'feature_accuracies': feature_accuracies,
        'image_comparisons': image_comparisons
    }
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n💾 Report saved to: {output_file}")
    
    return report


def print_accuracy_summary(report: Dict):
    """
    Print a formatted summary of the accuracy report.
    
    Parameters:
        report (Dict): Accuracy report
    """
    print("\n" + "="*80)
    print("📈 ACCURACY SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Images: {summary['total_images']}")
    print(f"   Total Features: {summary['total_features']}")
    print(f"   Overall Exact Match Accuracy: {summary['overall_exact_match_accuracy']:.2f}%")
    print(f"   Overall Partial Match Accuracy: {summary['overall_partial_match_accuracy']:.2f}%")
    
    print(f"\n📋 Per-Feature Accuracy (Top 10 Best):")
    print(f"{'Feature':<10} {'Exact Match':<15} {'Partial Match':<15} {'Categories':<12}")
    print("-" * 60)
    
    for feature_data in report['feature_accuracies'][:10]:
        print(f"{feature_data['feature']:<10} "
              f"{feature_data['exact_match_accuracy']:>6.2f}%        "
              f"{feature_data['partial_match_accuracy']:>6.2f}%        "
              f"{feature_data['num_categories']:>6}")
    
    print(f"\n📋 Per-Feature Accuracy (Bottom 10 Worst):")
    print(f"{'Feature':<10} {'Exact Match':<15} {'Partial Match':<15} {'Categories':<12}")
    print("-" * 60)
    
    for feature_data in report['feature_accuracies'][-10:]:
        print(f"{feature_data['feature']:<10} "
              f"{feature_data['exact_match_accuracy']:>6.2f}%        "
              f"{feature_data['partial_match_accuracy']:>6.2f}%        "
              f"{feature_data['num_categories']:>6}")
    
    print(f"\n📷 Per-Image Accuracy:")
    print(f"{'Filename':<50} {'Accuracy':<10}")
    print("-" * 60)
    
    for img_data in sorted(report['image_comparisons'], 
                           key=lambda x: x['accuracy'], 
                           reverse=True):
        print(f"{img_data['filename']:<50} {img_data['accuracy']:>6.2f}%")
    
    print("\n" + "="*80)


def run_validation(
    image_directory: str,
    ground_truth_file: str,
    prompt_file: str,
    model: str = "gpt-4o",
    max_tokens: int = 300,
    temperature: float = 0.7,
    output_report: str = "validation_report.json",
    save_extracted: str = "validation_extracted.json"
) -> Dict:
    """
    Run the complete validation pipeline.
    
    Parameters:
        image_directory (str): Directory containing validation images
        ground_truth_file (str): Path to ground truth JSON file
        prompt_file (str): Path to prompt text file
        model (str): Model to use for extraction
        max_tokens (int): Maximum tokens per response
        temperature (float): Response creativity
        output_report (str): Path to save validation report
        save_extracted (str): Path to save extracted features
        
    Returns:
        Dict: Validation report
    """
    print("="*80)
    print("🔬 STARTING FEATURE EXTRACTION VALIDATION")
    print("="*80)
    
    # Step 1: Load ground truth
    print("\n📖 STEP 1: Loading ground truth data...")
    ground_truth_df = load_ground_truth(ground_truth_file)
    print(f"   Loaded {len(ground_truth_df)} validation samples")
    print(f"   Features: {list(ground_truth_df.columns)}")
    
    # Step 2: Load prompt
    print("\n📝 STEP 2: Loading prompt...")
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    print(f"   Prompt length: {len(prompt)} characters")
    
    # Step 3: Extract features
    print("\n🤖 STEP 3: Extracting features with GPT-4 Vision...")
    validation_filenames = list(ground_truth_df.index)
    extracted_df = extract_features_for_validation(
        image_directory=image_directory,
        validation_filenames=validation_filenames,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if extracted_df.empty:
        print("❌ No features extracted. Validation cannot proceed.")
        return None
    
    print(f"   Successfully extracted features for {len(extracted_df)} images")
    
    # Save extracted features
    extracted_df.to_json(save_extracted, orient='index', indent=4)
    print(f"   Saved extracted features to: {save_extracted}")
    
    # Step 4: Compare and generate report
    print("\n📊 STEP 4: Comparing features and calculating accuracy...")
    report = generate_accuracy_report(
        ground_truth_df=ground_truth_df,
        extracted_df=extracted_df,
        output_file=output_report
    )
    
    # Step 5: Print summary
    print_accuracy_summary(report)
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    print(f"📁 Validation report: {output_report}")
    print(f"📁 Extracted features: {save_extracted}")
    print("="*80)
    
    return report


if __name__ == "__main__":
    # Run validation
    report = run_validation(
        image_directory="images",
        ground_truth_file="validation_ground_truth.json",
        prompt_file="prompts/base_prompt.txt",
        model="gpt-4o",
        max_tokens=300,
        temperature=0.7,
        output_report="outputs/validation_report.json",
        save_extracted="outputs/validation_extracted.json"
    )

