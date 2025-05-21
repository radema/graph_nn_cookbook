"""
Usage example:
python 012_extract_contigui_logged_output.py --output_dir data/01_raw
"""
import argparse
import io
import logging
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.append(os.path.abspath("../../src"))  # noqa: E402
from config import URL_CONTIGUI

from logging_utils import setup_logger  # noqa: E402

logger = setup_logger(__name__)


def download_and_extract_excel(url):
    logger.info(f"Starting download from: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        logger.error("Failed to download file")
        raise Exception("Failed to download file")

    dataframes = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for file_name in z.namelist():
            if file_name.endswith(".xlsx"):
                logger.info(f"Reading file: {file_name}")
                with z.open(file_name) as f:
                    df = pd.read_excel(f, dtype=str)
                    df["source_file"] = file_name
                    dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def process_dataframe(df):
    logger.info("Starting dataframe transformation")
    if "COD_REG COMUNE" in df.columns and "COD_REG" in df.columns:
        df["COD_REG"] = df.apply(
            lambda row: row["COD_REG COMUNE"]
            if pd.notna(row["COD_REG COMUNE"])
            else row["COD_REG"],
            axis=1,
        )
    if "LUNGHEZZA CONFINE KM" in df.columns and "LUNGHEZZA CONFINE.1" in df.columns:
        df["LUNGHEZZA CONFINE KM"] = df.apply(
            lambda row: row["LUNGHEZZA CONFINE KM"]
            if pd.notna(row["LUNGHEZZA CONFINE KM"])
            else row["LUNGHEZZA CONFINE.1"],
            axis=1,
        )
    if (
        "COD_PRO COMUNE ADIACENTE" in df.columns
        and "COD_PRO COMUNE ADIACENTE" in df.columns
    ):
        df["COD_PRO COMUNE ADIACENTE"] = df.apply(
            lambda row: row["COD_PROV COMUNE ADIACENTE"]
            if pd.notna(row["COD_PROV COMUNE ADIACENTE"])
            else row["COD_PRO COMUNE ADIACENTE"],
            axis=1,
        )
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract and process contigui data.")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the output Parquet file will be saved",
    )
    args = parser.parse_args()

    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    try:
        df = download_and_extract_excel(URL_CONTIGUI)
        df = process_dataframe(df)

        logger.info("Process completed successfully")
        output_file = output_path + "/contigui_data.parquet"
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    except Exception as e:
        logger.exception("An error occurred during execution")


if __name__ == "__main__":
    main()
