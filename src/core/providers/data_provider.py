import os
import kaggle
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleDataProvider:
    def __init__(self, dataset: str, data_dir: str = "Scanned-Documents-Processing/data/raw") -> None:
        self.dataset = dataset
        assert data_dir is not None, 'data_dir cannot be empty!!!'
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        """Download the dataset from Kaggle"""
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(self.dataset, path=self.data_dir, unzip=True)
        logger.info(f"Dataset {self.dataset} downloaded and extracted to {self.data_dir}")