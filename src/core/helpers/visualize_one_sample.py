import os
import cv2
import json
import random


def visualize_one_sample(
    sample_name: str,
    data_dir: str,
    data_type: str = 'raw',
    flag: str = 'training_data'
) -> None:
    """Visualize one sample from dataset as scanned document with printed bounding box and target text."""
    annotation_path = os.path.join(
        data_dir, data_type, 'dataset', flag, 'annotations', f'{sample_name}.json'
    )
    image_path = os.path.join(
        data_dir, data_type, 'dataset', flag, 'images', f'{sample_name}.png'
    )

    with open(annotation_path, 'r', encoding='utf-8') as file:
        annotations = json.load(file)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    unique_labels = set(entry['label'] for entry in annotations.get('form', []))

    def random_color():
        return tuple(random.randint(50, 255) for _ in range(3))

    label_colors = {label: random_color() for label in unique_labels}

    for entry in annotations.get('form', []):
        label = entry['label']
        color = label_colors[label]
        for word in entry.get('words', []):
            bbox = word['box']  # [x1, y1, x2, y2]
            text = word['text']

            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
            )

    cv2.imshow('Sample {}'.format(sample_name), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()