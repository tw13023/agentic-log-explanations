"""
config_loader.py — Load and access configs/config.yaml.

Usage:
    from src.config_loader import load_config, get_llm_kwargs
    from src.llm_client import LLMClient

    # Load full config
    cfg = load_config()

    # Get LLMClient kwargs directly
    llm_client = LLMClient(**get_llm_kwargs())

    # Or with an explicit path
    llm_client = LLMClient(**get_llm_kwargs(config_path="/path/to/config.yaml"))
"""

from pathlib import Path
from typing import Dict, Optional

import yaml

# Resolve default config path relative to this file (src/ -> project root -> configs/)
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configs/config.yaml and return as a dict.

    Args:
        config_path: Override path to config file.  If None, uses
                     PROJECT_ROOT/configs/config.yaml.

    Returns:
        Config dict with top-level keys: datasets, model, rag, llm,
        gating, output, seed.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected at: {_DEFAULT_CONFIG_PATH}"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_llm_kwargs(
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """Return keyword arguments for LLMClient from the llm section of config.

    Args:
        config:      Pre-loaded config dict (skips file I/O if provided).
        config_path: Override config file path (used only when config is None).

    Returns:
        Dict with keys: provider, model, temperature, max_tokens, timeout.

    Example:
        from src.config_loader import get_llm_kwargs
        from src.llm_client import LLMClient
        llm_client = LLMClient(**get_llm_kwargs())
    """
    cfg = config if config is not None else load_config(config_path)
    llm = cfg.get("llm", {})
    return {
        "provider":    llm.get("provider",    "openai"),
        "model":       llm.get("model",       "gpt-5.1"),
        "temperature": llm.get("temperature", 0.1),
        "max_tokens":  llm.get("max_tokens",  2048),
        "timeout":     llm.get("timeout",     120),
    }
