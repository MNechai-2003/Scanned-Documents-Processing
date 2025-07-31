import os
import cv2
import numpy as np
from typing import Dict


def number_of_channels(image_dir: str) -> Dict[int, int]:
    """Return a dictionary with the count of images for each number of channels."""
    channels = dict()

    for file in os.listdir(image_dir):
        if file.lower().endswith('.png'):
            image = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: {file} could not be read.")
                continue

            if len(image.shape) == 2:
                # Grayscale image (no channel dimension)
                num_channels = 1
            else:
                num_channels = image.shape[2]

            channels[num_channels] = channels.get(num_channels, 0) + 1

    return channels


def calculate_main_statistic_for_images(image_dir: str) -> Dict[str, int]:
    """Return a dictionary with main statitics about aspect ratio of images."""
    aspect_ratios = []

    for file in os.listdir(image_dir):
        if file.lower().endswith('.png'):
            image = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            height, width = image.shape[:2]
            aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)

    return {
        "mean": np.mean(aspect_ratios),
        "median": np.median(aspect_ratios),
        "min": np.min(aspect_ratios),
        "max": np.max(aspect_ratios)
    }