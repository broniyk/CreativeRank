import pytest

from creative_pipeline import (
    run_image_analysis_pipeline,
    get_category_mappings,
    decode_features,
)

def test_image_analysis_pipeline_and_decoding():
    # Load prompt
    with open("./prompts/base_prompt.txt", "r") as f:
        prompt = f.read()

    # Run the complete pipeline
    original_df, encoded_df, encoder = run_image_analysis_pipeline(
        image_directory="./images/",
        prompt=prompt,
        output_json= "./outputs/image_analysis_results.json",
        output_csv= "./outputs/creative_pipeline_encoded.csv",
        model="gpt-4o",
        max_tokens=300,
        temperature=0.7,
        batch_size=5,  # Process first 5 images for testing; set to None for all
    )

    # Optional: Get category mappings
    assert encoder is not None, "Encoder should not be None"
    mappings = get_category_mappings(encoder, original_df.columns)
    assert isinstance(mappings, dict)
    assert len(mappings) > 0

    # Check first few mappings as example
    for feature, mapping in list(mappings.items())[:3]:
        assert isinstance(mapping, dict)
        assert len(mapping) > 0

    # Test decoding
    decoded_df = decode_features(encoded_df, encoder, original_df.columns)
    assert decoded_df.equals(original_df), "Decoded DataFrame does not match the original"
