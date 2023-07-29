import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pytorch_pfn_extras.config import Config


def load_config(path: Path, types: Optional[dict] = None) -> Config:
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict, types=types)


def load_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)

def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity matrix between two arrays.

    Args:
        x (np.ndarray): The first array.
        y (np.ndarray): The second array.

    Returns:
        np.ndarray: The cosine similarity matrix.
    """

    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    x = x / x_norm[:, np.newaxis]
    y = y / y_norm[:, np.newaxis]
    return np.dot(x, y.T)
