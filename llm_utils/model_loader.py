"""
Model loading and inference utilities
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
    pipeline
)
from typing import Dict, Any, Optional, List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    model_type: str = "causal",
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    **kwargs
) -> tuple:
    """
    Load a HuggingFace model and tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        model_type: Type of model - 'causal' for GPT-like, 'seq2seq' for T5-like
        device: Device to load the model on ('cuda', 'cpu', or None for auto)
        load_in_8bit: Whether to load model in 8-bit precision
        load_in_4bit: Whether to load model in 4-bit precision
        trust_remote_code: Whether to trust remote code for custom models
        **kwargs: Additional arguments to pass to model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Set up model loading arguments
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        **kwargs
    }
    
    # Add quantization if specified
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["device_map"] = "auto"
    else:
        # Only set device if not using quantization (which requires device_map)
        if "device_map" not in model_kwargs:
            model_kwargs["device_map"] = None
    
    # Load model based on type
    if model_type.lower() in ["causal", "gpt", "llm"]:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    elif model_type.lower() in ["seq2seq", "t5", "bart"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'causal' or 'seq2seq'")
    
    # Move to device if not using quantization
    if not (load_in_8bit or load_in_4bit) and model_kwargs["device_map"] is None:
        model = model.to(device)
    
    logger.info(f"Model loaded successfully")
    
    return model, tokenizer


def run_inference(
    model,
    tokenizer,
    texts: Union[str, List[str]],
    max_length: int = 512,
    max_new_tokens: Optional[int] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    do_sample: bool = False,
    num_return_sequences: int = 1,
    batch_size: int = 1,
    **kwargs
) -> List[str]:
    """
    Run inference on text(s) using a loaded model.
    
    Args:
        model: Loaded HuggingFace model
        tokenizer: Loaded tokenizer
        texts: Input text or list of texts
        max_length: Maximum length of generated sequence
        max_new_tokens: Maximum number of new tokens to generate (overrides max_length)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate per input
        batch_size: Batch size for processing multiple texts
        **kwargs: Additional generation arguments
        
    Returns:
        List of generated texts
    """
    # Convert single text to list
    if isinstance(texts, str):
        texts = [texts]
    
    device = next(model.parameters()).device
    model.eval()
    
    all_outputs = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        # Set up generation config
        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            **kwargs
        }
        
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        else:
            gen_kwargs["max_length"] = max_length
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode outputs
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(decoded)
    
    return all_outputs


def create_pipeline(
    model_name: str,
    task: str = "text-generation",
    device: Optional[int] = None,
    **kwargs
):
    """
    Create a HuggingFace pipeline for easy inference.
    
    Args:
        model_name: Name or path of the model
        task: Task type (e.g., 'text-generation', 'text2text-generation')
        device: Device index (-1 for CPU, None for auto)
        **kwargs: Additional arguments for pipeline creation
        
    Returns:
        HuggingFace pipeline object
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    
    logger.info(f"Creating {task} pipeline with model: {model_name}")
    
    pipe = pipeline(
        task,
        model=model_name,
        device=device,
        **kwargs
    )
    
    return pipe
