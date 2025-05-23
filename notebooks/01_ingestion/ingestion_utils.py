import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset


class BaseCSVProcessor:
    """
    A base class for processing CSV (or zipped-CSV) label files into
    a standardized parquet output via lazy evaluation.
    """

    # Fallback default schema if none is provided
    DEFAULT_SCHEMA = {
        "label_id": pl.Int64,
        "label_value": pl.Int64,
        "label_timestamp": pl.Datetime,
    }

    DEFAULT_SCHEMA_OVERRIDE = {"Value": pl.Float64}

    def __init__(
        self, input_file_path: str, output_file_path: str, show_head: bool = False
    ):
        self.input_file_path = Path(input_file_path)
        self.output_file_path = Path(output_file_path)

        self.schema_overrides = self.DEFAULT_SCHEMA_OVERRIDE
        self.show_head = show_head

    def load(self) -> pl.LazyFrame:
        """
        Lazily load CSV data, transparently handling .zip archives
        (takes the first .csv found inside). Other compressed files
        (.gz, .bz2, .xz, etc.) are inferred automatically by Polars.
        """
        suffix = self.input_file_path.suffix.lower()

        # Handle zip archives explicitly
        if suffix == ".zip":
            with zipfile.ZipFile(self.input_file_path) as zf:
                # pick first CSV in archive
                csv_candidates = [
                    n for n in zf.namelist() if n.lower().endswith(".csv")
                ]
                if not csv_candidates:
                    raise FileNotFoundError(f"No .csv inside {self.input_file_path!r}")
                # read bytes into memory and feed to scan_csv
                raw = zf.read(csv_candidates[0])
                return pl.scan_csv(
                    BytesIO(raw),
                    separator="|",
                    infer_schema_length=10000,
                    schema_overrides=self.DEFAULT_SCHEMA_OVERRIDE,
                )
        # Let Polars infer compression from file extension
        return pl.scan_csv(str(self.input_file_path))

    def rename_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Rename all columns in the DataFrame to lowercase and replace spaces with underscores.
        """
        columns = df.collect_schema().names()
        return df.rename({col: col.replace(" ", "_") for col in columns})

    def save(self, df: pl.LazyFrame) -> None:
        """
        Materialize to a Parquet file via a lazy sink.
        """
        df.sink_parquet(str(self.output_file_path))

    def show(self, df: pl.LazyFrame) -> None:
        """
        Show the first few rows of the DataFrame.
        """
        if self.show_head:
            print(df.head().collect().to_pandas())

    def run(self) -> None:
        """
        Full pipeline: load → clean → validate → save.
        """
        lf = self.load()
        lf = self.rename_columns(lf)
        self.show(lf)
        self.save(lf)
