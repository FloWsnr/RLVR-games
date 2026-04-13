"""Dataset manifest types shared across game-specific corpora."""

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any

DATASET_MANIFEST_SCHEMA_VERSION = 1


class DatasetSplit(StrEnum):
    """Supported processed dataset splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitPercentages:
    """Deterministic split percentages used during preprocessing.

    Attributes
    ----------
    train : int
        Percentage of records assigned to the training split.
    val : int
        Percentage of records assigned to the validation split.
    test : int
        Percentage of records assigned to the test split.
    """

    train: int
    val: int
    test: int

    def __post_init__(self) -> None:
        """Validate that split percentages are coherent.

        Raises
        ------
        ValueError
            If any percentage is negative or if the percentages do not sum to
            exactly 100.
        """
        percentages = (self.train, self.val, self.test)
        if any(value < 0 for value in percentages):
            raise ValueError("Dataset split percentages must be non-negative.")
        if sum(percentages) != 100:
            raise ValueError("Dataset split percentages must sum to 100.")

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable mapping of split percentages.

        Returns
        -------
        dict[str, int]
            Mapping keyed by split name.
        """
        return {
            DatasetSplit.TRAIN.value: self.train,
            DatasetSplit.VAL.value: self.val,
            DatasetSplit.TEST.value: self.test,
        }


@dataclass(frozen=True, slots=True)
class DatasetShardManifest:
    """Metadata describing one processed dataset shard.

    Attributes
    ----------
    split : DatasetSplit
        Split served by this shard.
    path : str
        Relative path from the manifest directory to the shard file.
    record_count : int
        Number of serialized records stored in the shard.
    """

    split: DatasetSplit
    path: str
    record_count: int

    def __post_init__(self) -> None:
        """Validate shard metadata.

        Raises
        ------
        ValueError
            If the shard path is empty or the record count is not positive.
        """
        if not self.path:
            raise ValueError("Dataset shard paths must be non-empty.")
        if self.record_count <= 0:
            raise ValueError("Dataset shard record counts must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable shard mapping.

        Returns
        -------
        dict[str, object]
            Mapping containing split, relative path, and record count.
        """
        return {
            "split": self.split.value,
            "path": self.path,
            "record_count": self.record_count,
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Processed dataset manifest stored next to dataset shards.

    Attributes
    ----------
    schema_version : int
        Manifest schema version understood by the runtime loader.
    game : str
        Game owning the dataset, for example ``"chess"``.
    dataset : str
        Dataset family name, for example ``"lichess-puzzles"``.
    version : str
        Deterministic processed dataset version identifier.
    record_format : str
        Record storage format identifier.
    shards : tuple[DatasetShardManifest, ...]
        All split shards written for the processed dataset.
    source_url : str | None
        Upstream source URL when the dataset originated from a download.
    source_filename : str | None
        Original raw filename used to build the processed dataset.
    source_sha256 : str | None
        SHA-256 hash of the raw input file when available.
    license : str | None
        Human-readable license identifier for the source data.
    metadata : dict[str, Any]
        Additional builder metadata such as split ratios or record counts.
    """

    schema_version: int
    game: str
    dataset: str
    version: str
    record_format: str
    shards: tuple[DatasetShardManifest, ...]
    source_url: str | None
    source_filename: str | None
    source_sha256: str | None
    license: str | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate manifest fields.

        Raises
        ------
        ValueError
            If the schema version or required identifying fields are invalid.
        """
        if self.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset manifest schema version: {self.schema_version}."
            )
        if not self.game:
            raise ValueError("Dataset manifests require a non-empty game name.")
        if not self.dataset:
            raise ValueError("Dataset manifests require a non-empty dataset name.")
        if not self.version:
            raise ValueError("Dataset manifests require a non-empty version.")
        if not self.record_format:
            raise ValueError("Dataset manifests require a non-empty record format.")
        if not self.shards:
            raise ValueError("Dataset manifests require at least one shard.")

    def shards_for_split(
        self,
        *,
        split: DatasetSplit,
    ) -> tuple[DatasetShardManifest, ...]:
        """Return the shards assigned to a given split.

        Parameters
        ----------
        split : DatasetSplit
            Split whose shards should be returned.

        Returns
        -------
        tuple[DatasetShardManifest, ...]
            Ordered shard metadata for the requested split.
        """
        return tuple(shard for shard in self.shards if shard.split == split)

    def split_record_counts(self) -> dict[str, int]:
        """Return total record counts aggregated by split.

        Returns
        -------
        dict[str, int]
            Mapping keyed by split name with total counts per split.
        """
        counts = {
            DatasetSplit.TRAIN.value: 0,
            DatasetSplit.VAL.value: 0,
            DatasetSplit.TEST.value: 0,
        }
        for shard in self.shards:
            counts[shard.split.value] += shard.record_count
        return counts

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable manifest mapping.

        Returns
        -------
        dict[str, object]
            Serialized manifest mapping suitable for ``json.dump``.
        """
        return {
            "schema_version": self.schema_version,
            "game": self.game,
            "dataset": self.dataset,
            "version": self.version,
            "record_format": self.record_format,
            "shards": [shard.to_dict() for shard in self.shards],
            "source_url": self.source_url,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "license": self.license,
            "metadata": self.metadata,
        }


def write_dataset_manifest(
    *,
    path: Path,
    manifest: DatasetManifest,
) -> None:
    """Persist a processed dataset manifest as formatted JSON.

    Parameters
    ----------
    path : Path
        Manifest path to create or overwrite.
    manifest : DatasetManifest
        Manifest payload to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_dataset_manifest(*, path: Path) -> DatasetManifest:
    """Load a processed dataset manifest from disk.

    Parameters
    ----------
    path : Path
        Manifest file to load.

    Returns
    -------
    DatasetManifest
        Parsed and validated manifest instance.

    Raises
    ------
    FileNotFoundError
        If the manifest path does not exist.
    ValueError
        If the JSON payload is missing required fields or contains invalid
        values.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    shards_payload = payload["shards"]
    shards = tuple(
        DatasetShardManifest(
            split=DatasetSplit(shard_payload["split"]),
            path=shard_payload["path"],
            record_count=shard_payload["record_count"],
        )
        for shard_payload in shards_payload
    )
    return DatasetManifest(
        schema_version=payload["schema_version"],
        game=payload["game"],
        dataset=payload["dataset"],
        version=payload["version"],
        record_format=payload["record_format"],
        shards=shards,
        source_url=payload.get("source_url"),
        source_filename=payload.get("source_filename"),
        source_sha256=payload.get("source_sha256"),
        license=payload.get("license"),
        metadata=payload.get("metadata", {}),
    )
