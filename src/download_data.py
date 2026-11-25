"""Download and extract the Kaggle Traffic Prediction dataset."""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
import requests

KAGGLE_DATASET_SLUG = "hasibullahaman/traffic-prediction-dataset"
KAGGLE_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET_SLUG}"
)


def download_kaggle_dataset(
    data_dir: Path,
    force: bool = False,
    retries: int = 3,
    timeout: int = 30,
) -> Path:
    """Download Traffic.csv and TrafficTwoMonth.csv from Kaggle into data_dir.

    Args:
        data_dir: Directory where CSV files will be stored.
        force: Re-download even if TrafficTwoMonth.csv already exists.
        retries: Number of network retry attempts on transient errors.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the primary dataset file (TrafficTwoMonth.csv).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    primary_csv = data_dir / "TrafficTwoMonth.csv"

    if primary_csv.exists() and not force:
        print(f"[Data] Using cached Kaggle dataset at: {primary_csv}")
        return primary_csv

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[Data] Downloading '{KAGGLE_DATASET_SLUG}' from kaggle.com "
                f"(attempt {attempt}/{retries})..."
            )
            response = requests.get(KAGGLE_DOWNLOAD_URL, timeout=timeout)
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(data_dir)
                extracted = zf.namelist()

            if not primary_csv.exists():
                raise FileNotFoundError(
                    f"Expected {primary_csv.name} in archive, found: {extracted}"
                )

            print(f"[Data] Extracted {extracted} into {data_dir}")
            return primary_csv
        except (requests.RequestException, zipfile.BadZipFile, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Failed to download dataset from Kaggle after {retries} attempts: {last_error}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    download_kaggle_dataset(project_root / "data", force=True)
