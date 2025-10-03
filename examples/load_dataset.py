#!/usr/bin/env python3
"""
Example script demonstrating dataset loading and preparation.

Usage:
    # Load from HuggingFace
    python examples/load_dataset.py --config configs/gpt2_example.yaml
    
    # Load local dataset
    python examples/load_dataset.py --config configs/local_dataset_example.yaml
    
    # Save prepared dataset
    python examples/load_dataset.py --config configs/gpt2_example.yaml --output ./prepared_data
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import llm_utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_utils.dataset_loader import load_dataset, prepare_dataset, save_dataset
from llm_utils.model_loader import load_model
from llm_utils.config_loader import load_config, validate_dataset_config


def main():
    parser = argparse.ArgumentParser(description="Load and prepare datasets")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save prepared dataset"
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Skip dataset preparation (tokenization)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="arrow",
        choices=["arrow", "csv", "json", "parquet"],
        help="Format to save dataset in"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    validate_dataset_config(config)
    
    dataset_config = config['dataset']
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        dataset_name=dataset_config.get('name'),
        dataset_path=dataset_config.get('path'),
        split=dataset_config.get('split'),
        streaming=dataset_config.get('streaming', False),
        # Pass additional config if present
        **({k: v for k, v in dataset_config.items() 
            if k in ['config_name', 'data_dir', 'data_files']})
    )
    
    print(f"\nDataset loaded: {dataset}")
    
    # Show sample
    if hasattr(dataset, 'column_names'):
        print(f"Columns: {dataset.column_names}")
        print(f"\nSample item:")
        print(dataset[0] if len(dataset) > 0 else "Empty dataset")
    elif hasattr(dataset, 'keys'):
        print(f"Splits: {list(dataset.keys())}")
        for split_name in dataset.keys():
            print(f"\n{split_name} split:")
            print(f"  Columns: {dataset[split_name].column_names}")
            print(f"  Size: {len(dataset[split_name])}")
    
    # Prepare dataset if requested
    if not args.no_prepare:
        print("\nPreparing dataset (tokenizing)...")
        
        # Load tokenizer from model config
        model_config = config.get('model', {})
        if 'model_name' not in model_config:
            print("Warning: No model configuration found, skipping preparation")
            return
        
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
        
        # Prepare dataset
        prepared_dataset = prepare_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            text_column=dataset_config.get('text_column', 'text'),
            max_length=config.get('inference', {}).get('max_length', 512)
        )
        
        print(f"\nPrepared dataset: {prepared_dataset}")
        
        # Use prepared dataset for saving
        dataset = prepared_dataset
    
    # Save dataset if output path provided
    if args.output:
        print(f"\nSaving dataset to {args.output} in {args.format} format...")
        save_dataset(dataset, args.output, format=args.format)
        print("Dataset saved successfully!")


if __name__ == "__main__":
    main()
