import os
import cv2
import random
import numpy as np
from collections import deque


def connected_component_analysis(image_path: str, output_path: str, connectivity: int = 4) -> None:
    """
        Method for segment characters from binary cut scanned documents.
        This method implement connected component analysis algorithm by using BFS search algorithm.
    """

    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary_image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY_INV)

    rows = binary_image.shape[0]
    columns = binary_image.shape[1]

    labeled_image = np.zeros((rows, columns), dtype=np.int32)
    current_label = 1

    for row in range(rows):
        for column in range(columns):
            if binary_image[row, column] != 0 and labeled_image[row, column] == 0:
                labeled_image[row, column] = current_label

                new_collection = deque()
                new_collection.append((row, column))

                while new_collection:
                    pixel_row, pixel_column = new_collection.popleft()

                    for neighbor_row, neighbor_column in [
                        (pixel_row - 1, pixel_column),
                        (pixel_row + 1, pixel_column),
                        (pixel_row, pixel_column - 1),
                        (pixel_row, pixel_column + 1)
                    ]:
                        if (
                            0 <= neighbor_row <= rows - 1
                            and 0 <= neighbor_column <= columns - 1
                            and binary_image[neighbor_row, neighbor_column] != 0
                            and labeled_image[neighbor_row, neighbor_column] == 0
                        ):
                            labeled_image[neighbor_row, neighbor_column] = current_label
                            new_collection.append((neighbor_row, neighbor_column))

                current_label += 1

    colored_output_image = np.zeros((rows, columns, 3), dtype=np.uint8)
    color_dict = {0: (255, 255, 255)}

    num_clasters = current_label - 1

    for i in range(1, num_clasters + 1):
        b = random.randint(50, 255)
        g = random.randint(50, 255)
        r = random.randint(50, 255)
        color_dict[i] = (b, g, r)

    for row in range(rows):
        for column in range(columns):
            label = labeled_image[row, column]
            colored_output_image[row, column] = color_dict[label]

    cv2.imwrite(output_path, colored_output_image)      
    print(f'Number of clasters: {num_clasters}')
