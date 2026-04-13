"""Stockfish installation and evaluator helpers for chess rewards."""

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen
import zipfile

from rlvr_games.games.chess.rewards import UciEngineEvaluator

STOCKFISH_RELEASE_API_URL = (
    "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
)
STOCKFISH_PATH_ENV_VAR = "RLVR_GAMES_STOCKFISH_PATH"


@dataclass(frozen=True, slots=True)
class StockfishAsset:
    """One downloadable Stockfish release asset.

    Attributes
    ----------
    name : str
        Archive filename published by the official Stockfish release.
    download_url : str
        Direct URL used to fetch the archive.
    sha256 : str
        Expected SHA-256 digest for the downloaded archive.
    """

    name: str
    download_url: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StockfishRelease:
    """Minimal release metadata needed for automated installation.

    Attributes
    ----------
    tag_name : str
        Git tag identifying the release, for example ``"sf_18"``.
    display_name : str
        Human-readable release name.
    assets : tuple[StockfishAsset, ...]
        Downloadable release archives published for the release.
    """

    tag_name: str
    display_name: str
    assets: tuple[StockfishAsset, ...]


class StockfishEvaluator(UciEngineEvaluator):
    """Concrete Stockfish-backed evaluator with binary resolution helpers."""

    @classmethod
    def from_engine_path(
        cls,
        *,
        engine_path: Path,
        depth: int,
        mate_score: int,
    ) -> "StockfishEvaluator":
        """Construct an evaluator from one explicit Stockfish binary path.

        Parameters
        ----------
        engine_path : Path
            Filesystem path to the Stockfish binary.
        depth : int
            Fixed analysis depth passed to the engine for each evaluation.
        mate_score : int
            Centipawn-equivalent scalar used to map mating lines to finite
            values.

        Returns
        -------
        StockfishEvaluator
            Evaluator configured to talk to the supplied Stockfish binary.
        """
        return cls(
            engine_path=validate_stockfish_binary_path(engine_path=engine_path),
            depth=depth,
            mate_score=mate_score,
        )

    @classmethod
    def from_installed_binary(
        cls,
        *,
        depth: int,
        mate_score: int,
    ) -> "StockfishEvaluator":
        """Construct an evaluator from the configured local Stockfish install.

        Parameters
        ----------
        depth : int
            Fixed analysis depth passed to the engine for each evaluation.
        mate_score : int
            Centipawn-equivalent scalar used to map mating lines to finite
            values.

        Returns
        -------
        StockfishEvaluator
            Evaluator configured to use the resolved Stockfish binary.
        """
        return cls(
            engine_path=resolve_stockfish_binary_path(),
            depth=depth,
            mate_score=mate_score,
        )


def repo_local_stockfish_root_dir() -> Path:
    """Return the repo-local directory reserved for Stockfish installs.

    Returns
    -------
    Path
        Directory stored next to the chess game implementation.
    """
    return Path(__file__).resolve().parent / ".stockfish"


def repo_local_stockfish_install_dir() -> Path:
    """Return the active repo-local Stockfish installation directory.

    Returns
    -------
    Path
        Directory expected to contain the active Stockfish binary and manifest.
    """
    return repo_local_stockfish_root_dir() / "current"


def repo_local_stockfish_binary_path() -> Path:
    """Return the expected repo-local Stockfish binary path.

    Returns
    -------
    Path
        Full path to the active repo-local Stockfish binary.
    """
    return repo_local_stockfish_install_dir() / _stockfish_binary_filename()


def validate_stockfish_binary_path(*, engine_path: Path) -> Path:
    """Validate that an explicit Stockfish binary path exists.

    Parameters
    ----------
    engine_path : Path
        Candidate Stockfish binary path to validate.

    Returns
    -------
    Path
        Normalized absolute path to the existing binary.

    Raises
    ------
    FileNotFoundError
        If `engine_path` does not exist.
    """
    normalized_path = engine_path.expanduser().resolve()
    if not normalized_path.exists():
        raise FileNotFoundError(f"Stockfish binary does not exist: {normalized_path}")
    return normalized_path


def resolve_stockfish_binary_path() -> Path:
    """Resolve a usable Stockfish binary path for the current host.

    Resolution order is:

    1. ``RLVR_GAMES_STOCKFISH_PATH``
    2. the repo-local install managed by ``rlvr-games-install-stockfish``
    3. ``stockfish`` on ``PATH``

    Returns
    -------
    Path
        Normalized absolute path to a usable Stockfish binary.

    Raises
    ------
    FileNotFoundError
        If no Stockfish binary can be resolved.
    """
    stockfish_path_from_env = os.environ.get(STOCKFISH_PATH_ENV_VAR)
    if stockfish_path_from_env is not None:
        return validate_stockfish_binary_path(engine_path=Path(stockfish_path_from_env))

    repo_local_binary_path = repo_local_stockfish_binary_path()
    if repo_local_binary_path.exists():
        return repo_local_binary_path.resolve()

    stockfish_path_from_path = shutil.which(_stockfish_binary_filename())
    if stockfish_path_from_path is not None:
        return Path(stockfish_path_from_path).resolve()

    raise FileNotFoundError(
        "Stockfish binary not found. Run `uv run rlvr-games-install-stockfish`, "
        f"set {STOCKFISH_PATH_ENV_VAR}, or pass --stockfish-path."
    )


def fetch_latest_stockfish_release() -> StockfishRelease:
    """Fetch the latest official Stockfish release metadata.

    Returns
    -------
    StockfishRelease
        Parsed release metadata derived from the GitHub release API.

    Raises
    ------
    ValueError
        If the API response is missing required release fields.
    """
    request = Request(
        STOCKFISH_RELEASE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "rlvr-games-stockfish-installer",
        },
    )
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))

    tag_name = payload.get("tag_name")
    display_name = payload.get("name")
    assets_payload = payload.get("assets")
    if not isinstance(tag_name, str) or not isinstance(display_name, str):
        raise ValueError("Stockfish release metadata is missing tag/name fields.")
    if not isinstance(assets_payload, list):
        raise ValueError("Stockfish release metadata is missing the assets list.")

    assets: list[StockfishAsset] = []
    for asset_payload in assets_payload:
        asset_name = asset_payload.get("name")
        download_url = asset_payload.get("browser_download_url")
        digest = asset_payload.get("digest")
        if not isinstance(asset_name, str) or not isinstance(download_url, str):
            continue
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            continue
        assets.append(
            StockfishAsset(
                name=asset_name,
                download_url=download_url,
                sha256=digest.removeprefix("sha256:"),
            )
        )

    if not assets:
        raise ValueError("Stockfish release metadata did not include SHA-256 assets.")

    return StockfishRelease(
        tag_name=tag_name,
        display_name=display_name,
        assets=tuple(assets),
    )


def detect_host_cpu_features() -> frozenset[str]:
    """Return a small set of CPU features relevant to Stockfish asset choice.

    Returns
    -------
    frozenset[str]
        Normalized feature flags such as ``"avx2"`` when they are detectable.
    """
    machine = platform.machine().lower()
    if machine not in ("amd64", "x86_64"):
        return frozenset()

    system = platform.system()
    raw_features = ""
    if system == "Linux":
        cpuinfo_path = Path("/proc/cpuinfo")
        if cpuinfo_path.exists():
            raw_features = cpuinfo_path.read_text(encoding="utf-8")
    elif system == "Darwin":
        process = subprocess.run(
            [
                "sysctl",
                "-n",
                "machdep.cpu.features",
                "machdep.cpu.leaf7_features",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        raw_features = process.stdout

    normalized_features = raw_features.lower().replace(".", "_").split()
    detected_features = {
        feature_name
        for feature_name in ("avx2",)
        if feature_name in normalized_features
    }
    return frozenset(detected_features)


def select_stockfish_asset_for_host(
    *,
    release: StockfishRelease,
    system: str,
    machine: str,
    cpu_features: frozenset[str],
) -> StockfishAsset:
    """Select the best Stockfish release asset for one host description.

    Parameters
    ----------
    release : StockfishRelease
        Release whose published assets should be considered.
    system : str
        Host operating system label, for example ``"Linux"``.
    machine : str
        Host machine architecture label, for example ``"x86_64"``.
    cpu_features : frozenset[str]
        Small set of normalized CPU capabilities used for target selection.

    Returns
    -------
    StockfishAsset
        Selected release asset for the host.

    Raises
    ------
    ValueError
        If no supported Stockfish asset is available for the supplied host.
    """
    normalized_system = system.lower()
    normalized_machine = machine.lower()
    available_assets = {asset.name: asset for asset in release.assets}

    preferred_asset_names: tuple[str, ...]
    if normalized_system == "linux" and normalized_machine in ("amd64", "x86_64"):
        preferred_asset_names = ("stockfish-ubuntu-x86-64.tar",)
        if "avx2" in cpu_features:
            preferred_asset_names = (
                "stockfish-ubuntu-x86-64-avx2.tar",
                *preferred_asset_names,
            )
    elif normalized_system == "darwin" and normalized_machine in ("arm64", "aarch64"):
        preferred_asset_names = ("stockfish-macos-m1-apple-silicon.tar",)
    elif normalized_system == "darwin" and normalized_machine in ("amd64", "x86_64"):
        preferred_asset_names = ("stockfish-macos-x86-64.tar",)
        if "avx2" in cpu_features:
            preferred_asset_names = (
                "stockfish-macos-x86-64-avx2.tar",
                *preferred_asset_names,
            )
    elif normalized_system == "windows" and normalized_machine in ("arm64", "aarch64"):
        preferred_asset_names = ("stockfish-windows-armv8.zip",)
    elif normalized_system == "windows" and normalized_machine in ("amd64", "x86_64"):
        preferred_asset_names = ("stockfish-windows-x86-64.zip",)
        if "avx2" in cpu_features:
            preferred_asset_names = (
                "stockfish-windows-x86-64-avx2.zip",
                *preferred_asset_names,
            )
    else:
        raise ValueError(
            "Automatic Stockfish installation is supported on Linux x86_64, "
            "macOS arm64/x86_64, and Windows arm64/x86_64. Use "
            f"{STOCKFISH_PATH_ENV_VAR} or --stockfish-path on other hosts."
        )

    for asset_name in preferred_asset_names:
        asset = available_assets.get(asset_name)
        if asset is not None:
            return asset

    raise ValueError(
        "Could not find a matching Stockfish release asset for "
        f"{system} {machine} in release {release.tag_name}."
    )


def install_latest_stockfish(
    *,
    install_dir: Path,
    force: bool,
) -> Path:
    """Download and install the latest Stockfish release for the local host.

    Parameters
    ----------
    install_dir : Path
        Destination directory for the active Stockfish install.
    force : bool
        Whether to overwrite an existing installation in `install_dir`.

    Returns
    -------
    Path
        Installed Stockfish binary path.
    """
    release = fetch_latest_stockfish_release()
    asset = select_stockfish_asset_for_host(
        release=release,
        system=platform.system(),
        machine=platform.machine(),
        cpu_features=detect_host_cpu_features(),
    )
    return install_stockfish_release_asset(
        release=release,
        asset=asset,
        install_dir=install_dir,
        force=force,
    )


def install_stockfish_release_asset(
    *,
    release: StockfishRelease,
    asset: StockfishAsset,
    install_dir: Path,
    force: bool,
) -> Path:
    """Download, verify, and extract one Stockfish release asset.

    Parameters
    ----------
    release : StockfishRelease
        Release that owns `asset`.
    asset : StockfishAsset
        Asset to download and install.
    install_dir : Path
        Destination directory for the active Stockfish install.
    force : bool
        Whether to overwrite an existing installation in `install_dir`.

    Returns
    -------
    Path
        Installed Stockfish binary path.
    """
    normalized_install_dir = install_dir.expanduser().resolve()
    existing_binary_path = normalized_install_dir / _stockfish_binary_filename()
    if existing_binary_path.exists() and not force:
        return existing_binary_path

    if normalized_install_dir.exists():
        shutil.rmtree(normalized_install_dir)
    normalized_install_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary_dir_string:
        temporary_dir = Path(temporary_dir_string)
        archive_path = temporary_dir / asset.name
        _download_stockfish_asset(asset=asset, archive_path=archive_path)
        installed_binary_path = _extract_stockfish_binary(
            archive_path=archive_path,
            install_dir=normalized_install_dir,
        )

    manifest_path = normalized_install_dir / "manifest.json"
    manifest_payload = {
        "release_tag": release.tag_name,
        "release_name": release.display_name,
        "asset_name": asset.name,
        "asset_sha256": asset.sha256,
        "download_url": asset.download_url,
        "binary_path": str(installed_binary_path),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return installed_binary_path


def build_stockfish_install_parser() -> ArgumentParser:
    """Build the command-line parser for Stockfish installation.

    Returns
    -------
    ArgumentParser
        Parser for the ``rlvr-games-install-stockfish`` console script.
    """
    parser = ArgumentParser(prog="rlvr-games-install-stockfish")
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=repo_local_stockfish_install_dir(),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def run_install_cli(argv: list[str]) -> int:
    """Run the Stockfish installer CLI for the supplied arguments.

    Parameters
    ----------
    argv : list[str]
        Argument vector excluding the executable name.

    Returns
    -------
    int
        Process-style exit code.
    """
    parser = build_stockfish_install_parser()
    args = parser.parse_args(argv)
    installed_binary_path = install_latest_stockfish(
        install_dir=args.install_dir,
        force=args.force,
    )
    print(f"Installed Stockfish binary: {installed_binary_path}")
    return 0


def main() -> int:
    """Run the Stockfish installer console script.

    Returns
    -------
    int
        Process-style exit code.
    """
    return run_install_cli(sys.argv[1:])


def _stockfish_binary_filename() -> str:
    """Return the Stockfish executable filename for the current host.

    Returns
    -------
    str
        Expected filename of the Stockfish binary on the current platform.
    """
    if os.name == "nt":
        return "stockfish.exe"
    return "stockfish"


def _download_stockfish_asset(*, asset: StockfishAsset, archive_path: Path) -> None:
    """Download one Stockfish archive and verify its SHA-256 digest.

    Parameters
    ----------
    asset : StockfishAsset
        Stockfish archive metadata describing the download.
    archive_path : Path
        Local path that should receive the downloaded archive.

    Raises
    ------
    ValueError
        If the downloaded archive does not match the expected digest.
    """
    request = Request(
        asset.download_url,
        headers={"User-Agent": "rlvr-games-stockfish-installer"},
    )
    digest = hashlib.sha256()
    with urlopen(request) as response, archive_path.open("wb") as archive_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            archive_file.write(chunk)
            digest.update(chunk)

    actual_digest = digest.hexdigest()
    if actual_digest != asset.sha256:
        raise ValueError(
            "Downloaded Stockfish archive failed SHA-256 verification: "
            f"expected {asset.sha256}, got {actual_digest}."
        )


def _extract_stockfish_binary(*, archive_path: Path, install_dir: Path) -> Path:
    """Extract the Stockfish executable from one downloaded archive.

    Parameters
    ----------
    archive_path : Path
        Downloaded Stockfish archive.
    install_dir : Path
        Destination directory receiving the extracted binary.

    Returns
    -------
    Path
        Installed Stockfish binary path.

    Raises
    ------
    ValueError
        If the archive does not contain a recognizable Stockfish executable.
    """
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = Path(member.filename)
                if member.is_dir() or not _looks_like_stockfish_binary(member_path):
                    continue
                extracted_binary_path = install_dir / _stockfish_binary_filename()
                with (
                    archive.open(member) as source,
                    extracted_binary_path.open("wb") as destination,
                ):
                    shutil.copyfileobj(source, destination)
                _make_executable(binary_path=extracted_binary_path)
                return extracted_binary_path
        raise ValueError("Downloaded Stockfish zip did not contain an engine binary.")

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if not member.isfile() or not _looks_like_stockfish_binary(member_path):
                continue
            extracted_binary_path = install_dir / _stockfish_binary_filename()
            extracted_file = archive.extractfile(member)
            if extracted_file is None:
                continue
            with extracted_file, extracted_binary_path.open("wb") as destination:
                shutil.copyfileobj(extracted_file, destination)
            _make_executable(binary_path=extracted_binary_path)
            return extracted_binary_path

    raise ValueError("Downloaded Stockfish tar archive did not contain a binary.")


def _looks_like_stockfish_binary(member_path: Path) -> bool:
    """Return whether one archive member looks like a Stockfish executable.

    Parameters
    ----------
    member_path : Path
        Archive member path to inspect.

    Returns
    -------
    bool
        `True` when the member name matches Stockfish executable naming.
    """
    candidate_name = member_path.name.lower()
    if candidate_name in {"stockfish", "stockfish.exe"}:
        return True
    if candidate_name.endswith(".exe") and candidate_name.startswith("stockfish-"):
        return True
    return "." not in candidate_name and candidate_name.startswith("stockfish-")


def _make_executable(*, binary_path: Path) -> None:
    """Ensure a downloaded Stockfish binary has execute permissions.

    Parameters
    ----------
    binary_path : Path
        Binary whose mode should be updated when the host supports it.
    """
    if os.name == "nt":
        return
    current_mode = binary_path.stat().st_mode
    binary_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


__all__ = [
    "STOCKFISH_PATH_ENV_VAR",
    "StockfishAsset",
    "StockfishEvaluator",
    "StockfishRelease",
    "fetch_latest_stockfish_release",
    "install_latest_stockfish",
    "repo_local_stockfish_binary_path",
    "repo_local_stockfish_install_dir",
    "repo_local_stockfish_root_dir",
    "resolve_stockfish_binary_path",
    "select_stockfish_asset_for_host",
    "validate_stockfish_binary_path",
]
