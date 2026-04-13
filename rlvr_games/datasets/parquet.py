"""Parquet-backed dataset helpers for preprocessing and runtime sampling."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from random import Random
from typing import Any, Callable, Generic, TextIO, TypeVar
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

from rlvr_games.datasets.manifest import (
    DatasetManifest,
    DatasetShardManifest,
    DatasetSplit,
    SplitPercentages,
    load_dataset_manifest,
)

RecordT = TypeVar("RecordT")


def open_text_input(*, path: Path) -> TextIO:
    """Open a plain-text or ``.zst`` input file as UTF-8 text.

    Parameters
    ----------
    path : Path
        File path to open.

    Returns
    -------
    TextIO
        Text stream for reading UTF-8 content.
    """
    if path.suffix == ".zst":
        return zstandard.open(path, mode="rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_parquet_records(*, path: Path) -> tuple[dict[str, Any], ...]:
    """Load all records stored in a Parquet shard.

    Parameters
    ----------
    path : Path
        Parquet file containing normalized dataset records.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Parsed record mappings in file order.
    """
    table = pq.read_table(path, memory_map=True)
    return tuple(table.to_pylist())


def sha256_file(*, path: Path) -> str:
    """Return the SHA-256 digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    *,
    url: str,
    destination: Path,
) -> Path:
    """Download a URL to a local file path.

    Parameters
    ----------
    url : str
        Source URL to download.
    destination : Path
        File path to create.

    Returns
    -------
    Path
        The downloaded destination path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return destination


def assign_split_from_key(
    *,
    key: str,
    split_percentages: SplitPercentages,
) -> DatasetSplit:
    """Assign a deterministic split to a record key.

    Parameters
    ----------
    key : str
        Stable record identifier.
    split_percentages : SplitPercentages
        Split percentages used by the dataset build.

    Returns
    -------
    DatasetSplit
        Split chosen by hashing the key into the configured percentage ranges.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % 100
    if bucket < split_percentages.train:
        return DatasetSplit.TRAIN
    if bucket < split_percentages.train + split_percentages.val:
        return DatasetSplit.VAL
    return DatasetSplit.TEST


@dataclass(slots=True)
class _OpenShard:
    """Mutable buffered writer state for one active shard."""

    split: DatasetSplit
    shard_index: int
    path: Path
    records: list[dict[str, Any]]


class ShardedParquetWriter:
    """Write processed records into split-aware Parquet shards."""

    def __init__(
        self,
        *,
        output_dir: Path,
        chunk_size: int,
    ) -> None:
        """Create a split-aware sharded Parquet writer.

        Parameters
        ----------
        output_dir : Path
            Directory receiving the emitted shard files.
        chunk_size : int
            Maximum number of records per shard.

        Raises
        ------
        ValueError
            If ``chunk_size`` is not strictly positive.
        """
        if chunk_size <= 0:
            raise ValueError("Dataset chunk_size must be positive.")
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self._open_shards: dict[DatasetSplit, _OpenShard] = {}
        self._next_shard_index = {
            DatasetSplit.TRAIN: 0,
            DatasetSplit.VAL: 0,
            DatasetSplit.TEST: 0,
        }
        self._completed_shards: list[DatasetShardManifest] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        split: DatasetSplit,
        record: dict[str, Any],
    ) -> None:
        """Write one record into the shard buffer for a split.

        Parameters
        ----------
        split : DatasetSplit
            Target split receiving the record.
        record : dict[str, Any]
            JSON-like normalized record payload.
        """
        open_shard = self._open_shards.get(split)
        if open_shard is None:
            open_shard = self._create_open_shard(split=split)
            self._open_shards[split] = open_shard

        open_shard.records.append(record)
        if len(open_shard.records) >= self.chunk_size:
            self._finalize_open_shard(open_shard=open_shard)
            del self._open_shards[split]

    def close(self) -> tuple[DatasetShardManifest, ...]:
        """Close all active shards and return their manifests.

        Returns
        -------
        tuple[DatasetShardManifest, ...]
            Metadata describing all completed shards.
        """
        for split in tuple(self._open_shards):
            self._finalize_open_shard(open_shard=self._open_shards[split])
        self._open_shards.clear()
        return tuple(self._completed_shards)

    def _create_open_shard(
        self,
        *,
        split: DatasetSplit,
    ) -> _OpenShard:
        """Create a fresh buffered shard for the supplied split.

        Parameters
        ----------
        split : DatasetSplit
            Split that will receive records.

        Returns
        -------
        _OpenShard
            Mutable buffered state for the new shard.
        """
        shard_index = self._next_shard_index[split]
        self._next_shard_index[split] += 1
        path = self.output_dir / f"{split.value}-{shard_index:05d}.parquet"
        return _OpenShard(
            split=split,
            shard_index=shard_index,
            path=path,
            records=[],
        )

    def _finalize_open_shard(
        self,
        *,
        open_shard: _OpenShard,
    ) -> None:
        """Flush one buffered shard to a Parquet file and record its manifest.

        Parameters
        ----------
        open_shard : _OpenShard
            Buffered shard state to finalize.
        """
        if not open_shard.records:
            return

        table = pa.Table.from_pylist(open_shard.records)
        pq.write_table(table, open_shard.path, compression="snappy")
        self._completed_shards.append(
            DatasetShardManifest(
                split=open_shard.split,
                path=open_shard.path.name,
                record_count=len(open_shard.records),
            )
        )


class ParquetScenarioDataset(Generic[RecordT]):
    """Runtime sampler for processed Parquet-backed scenario datasets."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest: DatasetManifest,
        parser: Callable[[dict[str, Any]], RecordT],
        max_cached_shards: int,
    ) -> None:
        """Create a cached sampler over a processed dataset manifest.

        Parameters
        ----------
        manifest_path : Path
            Manifest file used to resolve relative shard paths.
        manifest : DatasetManifest
            Parsed processed dataset manifest.
        parser : Callable[[dict[str, Any]], RecordT]
            Game-specific record parser used to validate loaded records.
        max_cached_shards : int
            Maximum number of fully parsed shards kept in memory.

        Raises
        ------
        ValueError
            If the manifest record format is unsupported or the cache size is
            not strictly positive.
        """
        if manifest.record_format != "parquet":
            raise ValueError("ParquetScenarioDataset only supports parquet manifests.")
        if max_cached_shards <= 0:
            raise ValueError("Scenario dataset caches require a positive size.")

        self.manifest_path = manifest_path
        self.manifest = manifest
        self.parser = parser
        self.max_cached_shards = max_cached_shards
        self._cached_shards: OrderedDict[str, tuple[RecordT, ...]] = OrderedDict()
        self._split_cumulative_counts = self._build_split_cumulative_counts()

    @classmethod
    def from_manifest_path(
        cls,
        *,
        manifest_path: Path,
        parser: Callable[[dict[str, Any]], RecordT],
        max_cached_shards: int,
    ) -> "ParquetScenarioDataset[RecordT]":
        """Load a processed dataset manifest and create a sampler.

        Parameters
        ----------
        manifest_path : Path
            Manifest file to load.
        parser : Callable[[dict[str, Any]], RecordT]
            Game-specific record parser used to validate loaded records.
        max_cached_shards : int
            Maximum number of fully parsed shards kept in memory.

        Returns
        -------
        ParquetScenarioDataset[RecordT]
            Cached sampler over the processed dataset.
        """
        manifest = load_dataset_manifest(path=manifest_path)
        return cls(
            manifest_path=manifest_path,
            manifest=manifest,
            parser=parser,
            max_cached_shards=max_cached_shards,
        )

    def sample_record(
        self,
        *,
        split: DatasetSplit,
        seed: int,
    ) -> RecordT:
        """Sample one normalized record deterministically from a split.

        Parameters
        ----------
        split : DatasetSplit
            Split to sample from.
        seed : int
            Seed controlling deterministic record selection.

        Returns
        -------
        RecordT
            Parsed normalized record for the requested split.

        Raises
        ------
        ValueError
            If the requested split contains no records.
        """
        cumulative_counts = self._split_cumulative_counts[split]
        if not cumulative_counts:
            raise ValueError(
                f"Processed dataset split {split.value!r} contains no records."
            )

        total_record_count = cumulative_counts[-1]
        random_index = Random(seed).randrange(total_record_count)
        shard_position = 0
        while random_index >= cumulative_counts[shard_position]:
            shard_position += 1

        shards = self.manifest.shards_for_split(split=split)
        previous_cumulative_count = 0
        if shard_position > 0:
            previous_cumulative_count = cumulative_counts[shard_position - 1]
        record_index = random_index - previous_cumulative_count
        records = self._load_shard_records(shard=shards[shard_position])
        return records[record_index]

    def _build_split_cumulative_counts(self) -> dict[DatasetSplit, tuple[int, ...]]:
        """Precompute cumulative record counts for each split.

        Returns
        -------
        dict[DatasetSplit, tuple[int, ...]]
            Cumulative shard-end counts keyed by split.
        """
        cumulative_counts: dict[DatasetSplit, tuple[int, ...]] = {}
        for split in DatasetSplit:
            running_total = 0
            shard_counts: list[int] = []
            for shard in self.manifest.shards_for_split(split=split):
                running_total += shard.record_count
                shard_counts.append(running_total)
            cumulative_counts[split] = tuple(shard_counts)
        return cumulative_counts

    def _load_shard_records(
        self,
        *,
        shard: DatasetShardManifest,
    ) -> tuple[RecordT, ...]:
        """Load and cache all parsed records from a shard.

        Parameters
        ----------
        shard : DatasetShardManifest
            Shard whose records should be loaded.

        Returns
        -------
        tuple[RecordT, ...]
            Parsed records stored in the shard.
        """
        cached_records = self._cached_shards.get(shard.path)
        if cached_records is not None:
            self._cached_shards.move_to_end(shard.path)
            return cached_records

        shard_path = self.manifest_path.parent / shard.path
        records = tuple(
            self.parser(record_payload)
            for record_payload in read_parquet_records(path=shard_path)
        )
        self._cached_shards[shard.path] = records
        if len(self._cached_shards) > self.max_cached_shards:
            self._cached_shards.popitem(last=False)
        return records
