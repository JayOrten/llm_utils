"""
Dataset loading and preparation utilities
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datasets import load_dataset as hf_load_dataset, Dataset, DatasetDict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: Optional[str] = None,
    dataset_path: Optional[str] = None,
    split: Optional[str] = None,
    streaming: bool = False,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset from HuggingFace Hub or local files.
    
    Args:
        dataset_name: Name of dataset on HuggingFace Hub (e.g., 'wikitext')
        dataset_path: Path to local dataset files
        split: Dataset split to load ('train', 'test', 'validation', or None for all)
        streaming: Whether to stream the dataset
        **kwargs: Additional arguments for dataset loading
        
    Returns:
        Dataset or DatasetDict object
    """
    if dataset_name:
        logger.info(f"Loading dataset from HuggingFace Hub: {dataset_name}")
        dataset = hf_load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            **kwargs
        )
    elif dataset_path:
        logger.info(f"Loading dataset from local path: {dataset_path}")
        dataset = _load_local_dataset(dataset_path, split=split, **kwargs)
    else:
        raise ValueError("Either dataset_name or dataset_path must be provided")
    
    logger.info(f"Dataset loaded successfully")
    return dataset


def _load_local_dataset(
    path: str,
    split: Optional[str] = None,
    data_format: Optional[str] = None,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Load dataset from local files.
    
    Args:
        path: Path to dataset directory or file
        split: Specific split to load
        data_format: Format of data files ('json', 'csv', 'text', 'parquet', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Dataset or DatasetDict object
    """
    path_obj = Path(path)
    
    # Auto-detect format if not specified
    if data_format is None:
        if path_obj.is_file():
            data_format = path_obj.suffix.lstrip('.')
        else:
            # Look for common file extensions in directory
            for ext in ['json', 'jsonl', 'csv', 'txt', 'parquet']:
                if list(path_obj.glob(f"*.{ext}")):
                    data_format = ext
                    break
    
    if data_format is None:
        raise ValueError(f"Could not auto-detect data format for path: {path}")
    
    logger.info(f"Loading dataset with format: {data_format}")
    
    # Handle different file formats
    if data_format in ['json', 'jsonl']:
        if path_obj.is_file():
            dataset = hf_load_dataset('json', data_files=str(path), split=split, **kwargs)
        else:
            data_files = _get_split_files(path_obj, 'json', split)
            dataset = hf_load_dataset('json', data_files=data_files, **kwargs)
    
    elif data_format == 'csv':
        if path_obj.is_file():
            dataset = hf_load_dataset('csv', data_files=str(path), split=split, **kwargs)
        else:
            data_files = _get_split_files(path_obj, 'csv', split)
            dataset = hf_load_dataset('csv', data_files=data_files, **kwargs)
    
    elif data_format in ['txt', 'text']:
        if path_obj.is_file():
            dataset = hf_load_dataset('text', data_files=str(path), split=split, **kwargs)
        else:
            data_files = _get_split_files(path_obj, 'txt', split)
            dataset = hf_load_dataset('text', data_files=data_files, **kwargs)
    
    elif data_format == 'parquet':
        if path_obj.is_file():
            dataset = hf_load_dataset('parquet', data_files=str(path), split=split, **kwargs)
        else:
            data_files = _get_split_files(path_obj, 'parquet', split)
            dataset = hf_load_dataset('parquet', data_files=data_files, **kwargs)
    
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    return dataset


def _get_split_files(
    directory: Path,
    extension: str,
    split: Optional[str] = None
) -> Union[str, Dict[str, str], Dict[str, List[str]]]:
    """
    Get data files organized by split.
    
    Args:
        directory: Directory containing data files
        extension: File extension to look for
        split: Specific split to get files for
        
    Returns:
        Dictionary mapping split names to file paths
    """
    # Handle jsonl as json
    if extension == 'json':
        patterns = ['*.json', '*.jsonl']
    else:
        patterns = [f'*.{extension}']
    
    all_files = []
    for pattern in patterns:
        all_files.extend(directory.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No {extension} files found in {directory}")
    
    # Try to organize by split
    splits = {}
    for file in all_files:
        file_name = file.stem.lower()
        if 'train' in file_name:
            splits.setdefault('train', []).append(str(file))
        elif 'valid' in file_name or 'val' in file_name:
            splits.setdefault('validation', []).append(str(file))
        elif 'test' in file_name:
            splits.setdefault('test', []).append(str(file))
        else:
            splits.setdefault('train', []).append(str(file))
    
    # Return based on split parameter
    if split:
        if split in splits:
            return splits[split][0] if len(splits[split]) == 1 else splits[split]
        else:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(splits.keys())}")
    
    return {k: v[0] if len(v) == 1 else v for k, v in splits.items()}


def prepare_dataset(
    dataset: Union[Dataset, DatasetDict],
    tokenizer,
    text_column: str = "text",
    max_length: int = 512,
    remove_columns: Optional[List[str]] = None,
    batched: bool = True,
    num_proc: Optional[int] = None,
    **kwargs
) -> Union[Dataset, DatasetDict]:
    """
    Prepare dataset for model training/inference by tokenizing.
    
    Args:
        dataset: Dataset or DatasetDict to prepare
        tokenizer: Tokenizer to use for preprocessing
        text_column: Name of the column containing text
        max_length: Maximum sequence length
        remove_columns: Columns to remove after preprocessing
        batched: Whether to process in batches
        num_proc: Number of processes for multiprocessing
        **kwargs: Additional arguments for map function
        
    Returns:
        Preprocessed dataset
    """
    logger.info("Preparing dataset with tokenization")
    
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    # Handle both Dataset and DatasetDict
    if isinstance(dataset, DatasetDict):
        tokenized = {}
        for split_name, split_dataset in dataset.items():
            logger.info(f"Tokenizing {split_name} split")
            cols_to_remove = remove_columns if remove_columns else split_dataset.column_names
            tokenized[split_name] = split_dataset.map(
                tokenize_function,
                batched=batched,
                num_proc=num_proc,
                remove_columns=cols_to_remove,
                **kwargs
            )
        result = DatasetDict(tokenized)
    else:
        cols_to_remove = remove_columns if remove_columns else dataset.column_names
        result = dataset.map(
            tokenize_function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=cols_to_remove,
            **kwargs
        )
    
    logger.info("Dataset preparation complete")
    return result


def save_dataset(
    dataset: Union[Dataset, DatasetDict],
    output_path: str,
    format: str = "arrow"
) -> None:
    """
    Save dataset to disk.
    
    Args:
        dataset: Dataset to save
        output_path: Path to save the dataset
        format: Format to save in ('arrow', 'csv', 'json', 'parquet')
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset to {output_path} in {format} format")
    
    if format == "arrow":
        dataset.save_to_disk(output_path)
    elif format == "csv":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_dataset.to_csv(output_dir / f"{split_name}.csv")
        else:
            dataset.to_csv(output_dir / "dataset.csv")
    elif format == "json":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_dataset.to_json(output_dir / f"{split_name}.json")
        else:
            dataset.to_json(output_dir / "dataset.json")
    elif format == "parquet":
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_dataset.to_parquet(output_dir / f"{split_name}.parquet")
        else:
            dataset.to_parquet(output_dir / "dataset.parquet")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info("Dataset saved successfully")
