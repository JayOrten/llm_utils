# llm_utils
Utilities and examples for training/inference of language models using HuggingFace and PyTorch Lightning

## Overview

This repository provides generic utilities for working with HuggingFace language models and datasets. It includes:

- **Model Loading**: Generic scripts for loading HuggingFace models (GPT, T5, BART, etc.)
- **Inference**: Run inference with configurable parameters
- **Dataset Loading**: Load datasets from HuggingFace Hub or local files
- **Configuration**: YAML-based configuration for reproducible experiments

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JayOrten/llm_utils.git
cd llm_utils
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
llm_utils/
├── llm_utils/              # Main package
│   ├── __init__.py
│   ├── config_loader.py    # Configuration loading utilities
│   ├── model_loader.py     # Model loading and inference
│   └── dataset_loader.py   # Dataset loading and preparation
├── configs/                # Example configuration files
│   ├── gpt2_example.yaml
│   ├── t5_example.yaml
│   └── local_dataset_example.yaml
├── examples/               # Example scripts
│   ├── run_inference.py
│   └── load_dataset.py
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### 1. Model Inference

Run inference with a pre-configured model:

```bash
python examples/run_inference.py \
    --config configs/gpt2_example.yaml \
    --input "Once upon a time"
```

Or use an input file:

```bash
python examples/run_inference.py \
    --config configs/gpt2_example.yaml \
    --input-file inputs.txt \
    --output outputs.txt
```

### 2. Load and Prepare Datasets

Load a dataset from HuggingFace:

```bash
python examples/load_dataset.py --config configs/gpt2_example.yaml
```

Load a local dataset and save prepared version:

```bash
python examples/load_dataset.py \
    --config configs/local_dataset_example.yaml \
    --output ./prepared_data \
    --format arrow
```

## Configuration

All scripts use YAML configuration files. Here's an example configuration:

```yaml
model:
  model_name: "gpt2"           # HuggingFace model name
  model_type: "causal"          # 'causal' or 'seq2seq'
  device: null                  # null=auto, 'cuda', or 'cpu'
  load_in_8bit: false          # 8-bit quantization
  load_in_4bit: false          # 4-bit quantization

inference:
  max_length: 512
  max_new_tokens: 100
  temperature: 1.0
  top_p: 0.95
  top_k: 50
  do_sample: true
  batch_size: 4

dataset:
  name: "wikitext"             # HuggingFace dataset name
  path: null                   # Or local path
  split: "test"
  text_column: "text"
```

See the `configs/` directory for more examples.

## Using as a Library

You can also import and use the utilities directly in your Python code:

### Model Loading and Inference

```python
from llm_utils import load_model, run_inference

# Load model
model, tokenizer = load_model(
    model_name="gpt2",
    model_type="causal",
    device="cuda"
)

# Run inference
outputs = run_inference(
    model=model,
    tokenizer=tokenizer,
    texts=["Hello, world!"],
    max_new_tokens=50,
    temperature=0.8
)

print(outputs[0])
```

### Dataset Loading

```python
from llm_utils import load_dataset, prepare_dataset
from transformers import AutoTokenizer

# Load dataset from HuggingFace
dataset = load_dataset(dataset_name="wikitext", split="test")

# Or load from local files
dataset = load_dataset(dataset_path="./data", split="train")

# Prepare dataset with tokenization
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prepared = prepare_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    text_column="text",
    max_length=512
)
```

## Supported Model Types

- **Causal LM** (GPT-2, GPT-Neo, LLaMA, etc.): Use `model_type: "causal"`
- **Seq2Seq** (T5, BART, etc.): Use `model_type: "seq2seq"`

## Supported Dataset Formats

- HuggingFace Hub datasets (by name)
- Local files:
  - JSON/JSONL
  - CSV
  - Text files
  - Parquet

## Advanced Features

### Quantization

For large models, use quantization to reduce memory usage:

```yaml
model:
  model_name: "meta-llama/Llama-2-7b-hf"
  load_in_8bit: true  # or load_in_4bit: true
```

### Batch Processing

Process multiple inputs efficiently:

```yaml
inference:
  batch_size: 8  # Process 8 inputs at once
```

### Streaming Datasets

For very large datasets:

```yaml
dataset:
  streaming: true  # Stream instead of loading into memory
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.12+
- PyTorch Lightning 2.0+

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and research purposes.
