"""
LLM Utilities - Utilities for working with HuggingFace models and datasets
"""

__version__ = "0.1.0"

# Lazy imports to avoid requiring all dependencies at import time
def __getattr__(name):
    if name == "load_model":
        from .model_loader import load_model
        return load_model
    elif name == "run_inference":
        from .model_loader import run_inference
        return run_inference
    elif name == "load_dataset":
        from .dataset_loader import load_dataset
        return load_dataset
    elif name == "prepare_dataset":
        from .dataset_loader import prepare_dataset
        return prepare_dataset
    elif name == "load_config":
        from .config_loader import load_config
        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "load_model",
    "run_inference",
    "load_dataset",
    "prepare_dataset",
    "load_config",
]
