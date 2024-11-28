import tomllib
from pathlib import Path


def load_config(config_path: str = "config.toml") -> dict:
    with open(config_path, 'rb') as f:  # Note: TOML needs binary mode
        return tomllib.load(f)
