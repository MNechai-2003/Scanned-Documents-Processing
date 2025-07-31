import os
import json
import numpy as np
from langdetect import detect
from typing import Dict, List


def language_detection(annotation_dir: str, min_text_len: int = 200) -> Dict[str, str]:
    """Detect all unique languages in JSON annotations."""

    languages = dict()

    for file in os.listdir(annotation_dir):
        if file.lower().endswith('.json'):
            try:
                with open(os.path.join(annotation_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue

            for entry in data.get('form', []):
                text = entry.get('text')
                if text and isinstance(text, str) and len(text) >= min_text_len:
                    try:
                        languages[detect(text)] = text
                    except Exception as e:
                        print(f"Error detecting language in file {file}: {e}")

    return languages


def output_all_classes_of_textual_data(annotation_dir: str) -> List[str]:
    """Output all unique classes of textual data."""
    classes: set = set()

    for file_name in os.listdir(annotation_dir):
        if file_name.lower().endswith('.json'):
            try:
                with open(os.path.join(annotation_dir, file_name), 'r', encoding='utf-8') as file:
                    data = json.load(file)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
                continue

            for entry in data.get('form', []):
                classes.add(entry.get('label'))

    return list(classes)


def main_statistics_for_textual_data(annotation_dir: str) -> Dict[str, int]:
    """Calculate main statistics for textual annotations."""
    length = []

    for file in os.listdir(annotation_dir):
        if file.lower().endswith('.json'):
            try:
                with open(os.path.join(annotation_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file: {file} {e} ")
                continue

            for entry in data.get('form', []):
                text = entry.get('text')
                if isinstance(text, str):
                    length.append(len(text))

    return {
        "mean": np.mean(length),
        "median": np.median(length),
        "min": np.min(length),
        "max": np.max(length)
    }