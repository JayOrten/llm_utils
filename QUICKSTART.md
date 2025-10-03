# Quick Reference Guide

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Quick Start

### 1. Basic Inference

```python
from llm_utils import load_model, run_inference

model, tokenizer = load_model("gpt2")
outputs = run_inference(
    model, tokenizer, 
    texts=["Hello, world!"],
    max_new_tokens=50
)
print(outputs[0])
```

### 2. Using Configuration Files

```bash
python examples/run_inference.py \
    --config configs/gpt2_example.yaml \
    --input "Your text here"
```

### 3. Load HuggingFace Dataset

```python
from llm_utils import load_dataset

dataset = load_dataset(
    dataset_name="wikitext",
    split="test"
)
```

### 4. Load Local Dataset

```python
from llm_utils import load_dataset

dataset = load_dataset(
    dataset_path="./data",
    split="train"
)
```

### 5. Prepare Dataset with Tokenization

```python
from llm_utils import load_dataset, prepare_dataset
from transformers import AutoTokenizer

dataset = load_dataset(dataset_name="wikitext", split="test")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prepared = prepare_dataset(
    dataset=dataset,
    tokenizer=tokenizer,
    text_column="text",
    max_length=512
)
```

## Configuration File Template

```yaml
model:
  model_name: "gpt2"
  model_type: "causal"  # or "seq2seq"
  device: null          # auto-detect
  load_in_8bit: false
  load_in_4bit: false

inference:
  max_length: 512
  max_new_tokens: 100
  temperature: 1.0
  top_p: 0.95
  top_k: 50
  do_sample: true
  batch_size: 4

dataset:
  name: null           # HuggingFace dataset name
  path: null           # or local path
  split: "train"
  text_column: "text"
  streaming: false
```

## Command Line Examples

### Run Inference
```bash
# Single input
python examples/run_inference.py \
    --config configs/gpt2_example.yaml \
    --input "Once upon a time"

# From file
python examples/run_inference.py \
    --config configs/gpt2_example.yaml \
    --input-file inputs.txt \
    --output outputs.txt
```

### Load Dataset
```bash
# Load and display
python examples/load_dataset.py \
    --config configs/gpt2_example.yaml

# Load and save prepared version
python examples/load_dataset.py \
    --config configs/gpt2_example.yaml \
    --output ./prepared_data \
    --format arrow
```

## Supported Features

### Model Types
- **Causal LM**: GPT-2, GPT-Neo, LLaMA, Mistral, etc.
- **Seq2Seq**: T5, BART, FLAN-T5, etc.

### Dataset Formats
- HuggingFace Hub (by name)
- JSON/JSONL files
- CSV files
- Text files
- Parquet files

### Advanced Features
- 8-bit and 4-bit quantization for large models
- Batch processing for efficiency
- Streaming for large datasets
- Custom tokenization parameters
- Flexible generation parameters (temperature, top-p, top-k)

## API Reference

### `load_model(model_name, model_type='causal', device=None, ...)`
Load a HuggingFace model and tokenizer.

### `run_inference(model, tokenizer, texts, max_length=512, ...)`
Run inference on text inputs.

### `load_dataset(dataset_name=None, dataset_path=None, split=None, ...)`
Load a dataset from HuggingFace or local files.

### `prepare_dataset(dataset, tokenizer, text_column='text', ...)`
Tokenize and prepare a dataset for training/inference.

### `load_config(config_path)`
Load a YAML configuration file.

## Tips

1. **Memory Management**: Use `load_in_8bit=true` or `load_in_4bit=true` for large models
2. **Batch Processing**: Set appropriate `batch_size` for your hardware
3. **Streaming**: Enable `streaming=true` for very large datasets
4. **Sampling**: Use `do_sample=true` with temperature/top_p for creative generation
5. **Greedy Decoding**: Set `do_sample=false` for deterministic outputs
