"""Small, safe download helpers for versioned external input datasets."""
from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


class DatasetDownloadError(RuntimeError):
    """A remote dataset could not be retrieved or did not have the expected form."""


def download_file(url: str, destination: str | Path, *, overwrite: bool = False) -> Path:
    """Download an HTTP(S) URL atomically and return its destination.

    Existing inputs are retained unless ``overwrite`` is explicitly requested.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DatasetDownloadError(f"Only http(s) dataset URLs are supported: {url!r}")
    dest = Path(destination)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f".{dest.name}.part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ecosystem-complexity-fetch"})
        with urllib.request.urlopen(request, timeout=900) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        if partial.stat().st_size == 0:
            raise DatasetDownloadError(f"Downloaded an empty file from {url}")
        partial.replace(dest)
    except DatasetDownloadError:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise DatasetDownloadError(f"Could not download {url}: {exc}") from exc
    return dest


def extract_named_zip_members(
    archive: str | Path,
    destination: str | Path,
    names: Iterable[str],
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract named members (by basename) from a zip, rejecting ambiguity."""
    dest = Path(destination)
    wanted = list(names)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            selected: dict[str, str] = {}
            # Validate the entire archive before changing any staged table.
            for name in wanted:
                matches = [member for member in members if Path(member).name == name]
                if len(matches) != 1:
                    raise DatasetDownloadError(
                        f"{Path(archive).name} contains {len(matches)} copies of {name!r}; "
                        "supply the official compiled dataset archive for this ISRaD version."
                    )
                selected[name] = matches[0]
            outputs: list[Path] = []
            for name in wanted:
                output = dest / name
                if not output.exists() or overwrite:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    partial = output.with_name(f".{output.name}.part")
                    with zf.open(selected[name]) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    partial.replace(output)
                outputs.append(output)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(f"{archive} is not a valid zip archive") from exc
    return outputs
