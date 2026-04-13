"""Shared dataset preparation and runtime loading helpers."""

from rlvr_games.datasets.parquet import (
    ParquetScenarioDataset,
    ShardedParquetWriter,
    assign_split_from_key,
    download_file,
    open_text_input,
    read_parquet_records,
    sha256_file,
)
from rlvr_games.datasets.manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    DatasetShardManifest,
    DatasetSplit,
    SplitPercentages,
    load_dataset_manifest,
    write_dataset_manifest,
)

__all__ = [
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DatasetManifest",
    "DatasetShardManifest",
    "DatasetSplit",
    "ParquetScenarioDataset",
    "ShardedParquetWriter",
    "SplitPercentages",
    "assign_split_from_key",
    "download_file",
    "load_dataset_manifest",
    "open_text_input",
    "read_parquet_records",
    "sha256_file",
    "write_dataset_manifest",
]
