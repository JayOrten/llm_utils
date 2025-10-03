#!/usr/bin/env python3
"""
Example script demonstrating model loading and inference.

Usage:
    python examples/run_inference.py --config configs/gpt2_example.yaml --input "Hello, world!"
    python examples/run_inference.py --config configs/gpt2_example.yaml --input-file input.txt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import llm_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_utils.model_loader import load_model, run_inference
from llm_utils.config_loader import load_config, validate_model_config


def main():
    parser = argparse.ArgumentParser(description="Run inference with a HuggingFace model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text for inference"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to file containing input texts (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_model_config(config)
    
    model_config = config['model']
    inference_config = config.get('inference', {})
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model(
        model_name=model_config['model_name'],
        model_type=model_config.get('model_type', 'causal'),
        device=model_config.get('device'),
        load_in_8bit=model_config.get('load_in_8bit', False),
        load_in_4bit=model_config.get('load_in_4bit', False),
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    
    # Get input texts
    if args.input:
        texts = [args.input]
    elif args.input_file:
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Either --input or --input-file must be provided")
    
    print(f"\nRunning inference on {len(texts)} input(s)...")
    
    # Run inference
    outputs = run_inference(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=inference_config.get('max_length', 512),
        max_new_tokens=inference_config.get('max_new_tokens'),
        temperature=inference_config.get('temperature', 1.0),
        top_p=inference_config.get('top_p', 1.0),
        top_k=inference_config.get('top_k', 50),
        do_sample=inference_config.get('do_sample', False),
        num_return_sequences=inference_config.get('num_return_sequences', 1),
        batch_size=inference_config.get('batch_size', 1)
    )
    
    # Display/save results
    print("\nResults:")
    print("=" * 80)
    for i, (input_text, output_text) in enumerate(zip(texts, outputs), 1):
        print(f"\nInput {i}: {input_text}")
        print(f"Output {i}: {output_text}")
        print("-" * 80)
    
    if args.output:
        with open(args.output, 'w') as f:
            for output in outputs:
                f.write(output + '\n')
        print(f"\nOutputs saved to {args.output}")


if __name__ == "__main__":
    main()
