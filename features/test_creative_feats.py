from creative_features import analyze_all_images, analyze_all_images_multi_model, combine_outputs_with_majority


# Test that claude is working
def test_analyze_gpt_5():
    analyze_all_images("/Users/broniy/Desktop/creative/data/validation",
                       "/Users/broniy/Desktop/creative/prompts/prompt_v2.txt",
                       model="gpt-5",
                       max_tokens=800,
                       output_file="../outputs/gpt5_validation_results.yaml",
                       )


def test_analyze_multi_model():
    analyze_all_images_multi_model(
        "/Users/broniy/Desktop/creative/data/validation",
        "/Users/broniy/Desktop/creative/prompts/prompt_v2.txt",
        models={
            "gpt-40": {"model": "gpt-4o", "temperature": 0.2},
            "claude-4-5-temp-0-7": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 500,
                "temperature": 0.7,
            },
            "claude-4-5-temp-0-5": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 500,
                "temperature": 0.5,
            },
        },
    )

def test_analyze_multi_model():
    analyze_all_images_multi_model(
        "/Users/broniy/Desktop/creative/data/validation",
        "/Users/broniy/Desktop/creative/prompts/prompt_v2.txt",
        models={
            "gpt-40": {"model": "gpt-4o", "temperature": 0.2},
            "claude-4-5-temp-0-7": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 500,
                "temperature": 0.7,
            },
            "claude-4-5-temp-0-5": {
                "model": "claude-sonnet-4-5",
                "max_tokens": 500,
                "temperature": 0.5,
            },
        },
    )

def test_analyze_single_model():
    analyze_all_images_multi_model(
        "/Users/broniy/Desktop/creative/data/validation",
        "/Users/broniy/Desktop/creative/prompts/prompt_v2.txt",
        models={
            "gpt-5": {"model": "gpt-5", "temperature": 0.2},
        },
    )

def test_combine_outputs_with_majority():
    combine_outputs_with_majority("/Users/broniy/Desktop/creative/outputs")

if __name__ == "__main__":
    test_analyze_single_model()