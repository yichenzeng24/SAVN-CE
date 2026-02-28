# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import logging
import os
import pickle
import numpy as np
import torch
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def convert_semantic_object_to_rgb(x):
    semantic_img = Image.new("P", (x.shape[1], x.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((x.flatten() % 40).astype(np.uint8))
    semantic_img = np.array(semantic_img.convert("RGB"))
    return semantic_img


def get_logger(name, level=logging.INFO, filename=None, mode="a", formatter=None, propagate=False, rank=0):
    logger = logging.getLogger(name)
    logger.propagate = propagate
    logger.setLevel(level)
    if formatter is None:
        # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    
    if not logger.hasHandlers():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == rank:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            if filename is not None:
                dir_path = os.path.dirname(filename)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                file_handler = logging.FileHandler(filename, mode=mode, encoding="utf-8")
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
    return logger

def wrap_to_pi(angle):
    """
    Wraps an angle in radians to the range [-pi, pi].
    :param angle: Angle in radians.
    :return: Wrapped angle in radians.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def wrap_to_half_pi(angle):
    """
    Wraps an angle in radians to the range [-pi/2, pi/2].
    :param angle: Angle in radians.
    :return: Wrapped angle in radians.
    """
    return (angle + np.pi / 2) % (np.pi) - np.pi / 2    

def normalize_distance(distance, min_distance: int, max_distance: int): 
    """
    Normalize distance to the range [0, 1].
    :param distance: Distance.
    :param min_distance: Minimum distance.
    :param max_distance: Maximum distance.
    :return: Normalized distance.
    """
    if isinstance(distance, np.ndarray): 
        distance = np.clip(distance, min_distance, max_distance)
    elif isinstance(distance, torch.Tensor):
        distance = torch.clamp(distance, min_distance, max_distance)
    else: 
        assert isinstance(distance, (int, float)), "distance must be a numpy array or a torch tensor, or a single number."
    distance = (distance - min_distance) / (max_distance - min_distance)
    return distance

def to_real_distance(distance, min_distance: int, max_distance: int): 
    """
    Convert normalized distance to real distance.
    :param distance: Normalized distance.
    :param min_distance: Minimum distance.
    :param max_distance: Maximum distance.
    :return: Real distance.
    """
    distance = distance * (max_distance - min_distance) + min_distance
    return distance