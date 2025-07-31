import os
import re
import json
from typing import List, Tuple, Dict, Any, Set


def get_data_paths(data_dir: str, data_type: str, flag: str) -> Tuple[str, str]:
    """Return tuple from two data paths (annotation directory and image directory)"""
    image_dir = os.path.join(data_dir, data_type, 'dataset', flag, 'images')
    annotation_dir = os.path.join(data_dir, data_type, 'dataset', flag, 'annotations')

    return annotation_dir, image_dir


def check_data_types(image_dir: str, annotation_dir: str) -> Set:
    """Check all existing data types."""
    image_types = set([os.path.splitext(file.lower())[1] for file in os.listdir(image_dir)])
    annotation_types = set([os.path.splitext(file.lower())[1] for file in os.listdir(annotation_dir)])

    return image_types.union(annotation_types)


def check_data_on_completeness(image_dir: str, annotation_dir: str) -> Tuple[list, list, list, list]:
    """Validate data completeness: number of images equals number of annotations."""

    images = {
        os.path.splitext(file.lower())[0]
        for file in os.listdir(image_dir)
        if file.lower().endswith('.png')
    }

    annotations = {
        os.path.splitext(file.lower())[0]
        for file in os.listdir(annotation_dir)
        if file.lower().endswith('.json')
    }

    all_file_names = images.union(annotations)
    paired_names = images.intersection(annotations)
    images_only = images.difference(annotations)
    annotations_only = annotations.difference(images)

    return (
        list(all_file_names),
        list(paired_names),
        list(images_only),
        list(annotations_only)
    )