"""
Output-contract writers shared by every ``ecosys`` app dispatcher.

Every app writes under ``./outputs/{name}/{verb}/[{subverb}/]`` where
``{name}`` is either a ``site.id`` (single-site runs) or a site-set
``name:`` (site-set runs). Each run directory carries the same three
required files so downstream tools (``analyze``, ``report``, notebooks,
``make_slides.js``) can consume outputs without introspection:

* ``manifest.json``       — run metadata + declared output-file list
* ``config.snapshot.yaml`` — fully-resolved config that produced the run
* ``logs/run.log``         — captured stdout+stderr of the run

Payload files (parquet / npz / png) are verb-specific and documented in
the plan file. The helpers here just make it trivial to emit them
uniformly.

The output tree layout is a contract: consumers read ``manifest.json``
first and then trust the declared file list, so writers MUST record
every non-log file they produce via ``RunDir.record_output``.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from ecosystem_complexity.model.configuration import ModelConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUTS_ROOT = _REPO_ROOT / "outputs"

# Bump when the output-contract shape changes in a way consumers must handle.
OUTPUT_CONTRACT_VERSION = "1.1"


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------


@dataclass
class RunDir:
    """A single run's output directory, with a manifest under construction.

    A ``RunDir`` is created by :func:`open_run_dir` at the top of an app,
    used to place files under a fixed layout during the run, and closed
    with :meth:`finalize` which writes ``manifest.json`` and the resolved
    config snapshot. Missing ``finalize`` still leaves a directory readable
    by downstream tools but without the manifest — treat manifest absence
    as "run in progress or crashed."
    """

    root: Path
    verb: str
    subverb: str | None
    name: str  # site.id or site-set name
    started_at: float
    inputs: dict[str, Any] = field(default_factory=dict)
    _outputs: list[str] = field(default_factory=list)
    _extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        """Return an absolute path inside this run dir, mkdir'ing parents."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def record_output(self, relpath: str) -> None:
        """Declare that ``relpath`` (relative to ``root``) is a run output."""
        if relpath not in self._outputs:
            self._outputs.append(relpath)

    def add_manifest_field(self, key: str, value: Any) -> None:
        """Attach an extra top-level field to the manifest (e.g. metrics)."""
        self._extra[key] = value

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def snapshot_config(self, config: ModelConfig) -> None:
        """Dump the resolved ModelConfig as YAML at config.snapshot.yaml."""
        snap = self.path("config.snapshot.yaml")
        snap.write_text(yaml.safe_dump(_config_to_dict(config), sort_keys=False))
        self.record_output("config.snapshot.yaml")

    def finalize(self) -> dict[str, Any]:
        """Write the manifest and return all artifact paths for integration.

        The return value is the machine-facing completion contract.  Its
        ``files`` mapping has the manifest's relative artifact names as keys
        and absolute paths as values, so callers do not need to reconstruct
        paths or know a verb-specific filename.  A declared-but-missing file
        is a writer error and prevents a completed manifest from being made.
        """
        outputs = sorted(self._outputs)
        missing = [relpath for relpath in outputs if not self.root.joinpath(relpath).is_file()]
        if missing:
            raise RuntimeError(
                "cannot finalize output contract; declared artifacts are missing: "
                + ", ".join(missing)
            )
        manifest_path = self.path("manifest.json")
        manifest = {
            "verb": self.verb,
            "subverb": self.subverb,
            "name": self.name,
            "contract_version": OUTPUT_CONTRACT_VERSION,
            "status": "complete",
            "output_dir": str(self.root),
            "ecosys_version": _ecosys_version(),
            "git_sha": _git_sha(),
            "started_at": _fmt_time(self.started_at),
            "finished_at": _fmt_time(time.time()),
            "inputs": self.inputs,
            "outputs": outputs,
            **self._extra,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default))
        return {
            "output_dir": str(self.root),
            "manifest": str(manifest_path),
            "files": {relpath: str(self.root / relpath) for relpath in outputs},
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def resolve_run_name(
    *,
    config: ModelConfig | None = None,
    site_set_name: str | None = None,
) -> str:
    """Pick the ``{name}`` component of the output directory.

    A site-set name (from the site-set YAML's top-level ``name:`` field)
    takes precedence over a per-site ``config.site_id``. Both being absent
    is a programming error — every app has one or the other.
    """
    if site_set_name:
        return _sanitize(site_set_name)
    if config is not None and config.site_id:
        return _sanitize(config.site_id)
    raise ValueError(
        "resolve_run_name requires either a site_set_name or a ModelConfig "
        "with a non-empty site.id"
    )


def open_run_dir(
    *,
    verb: str,
    name: str,
    subverb: str | None = None,
    outdir: str | Path | None = None,
    outputs_root: str | Path | None = None,
    inputs: dict[str, Any] | None = None,
) -> RunDir:
    """Create ``outputs/{name}/{verb}/[{subverb}/]`` and return a RunDir.

    When ``outdir`` is given, it overrides the default layout wholesale —
    useful for one-off runs. Otherwise the run lands under
    ``outputs_root or DEFAULT_OUTPUTS_ROOT``. Prior contents are NOT
    deleted; a rerun overwrites files with the same name and appends to
    the outputs list on ``finalize``.
    """
    if outdir is not None:
        root = Path(outdir).resolve()
    else:
        base = Path(outputs_root).resolve() if outputs_root else DEFAULT_OUTPUTS_ROOT
        parts = [name, verb]
        if subverb:
            parts.append(subverb)
        root = base.joinpath(*parts)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    return RunDir(
        root=root,
        verb=verb,
        subverb=subverb,
        name=name,
        started_at=time.time(),
        inputs=dict(inputs or {}),
    )


def attach_file_logger(run_dir: RunDir, level: int = logging.INFO) -> logging.Handler:
    """Route stdlib logging into ``{run_dir}/logs/run.log`` (append mode)."""
    log_path = run_dir.path("logs", "run.log")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    run_dir.record_output("logs/run.log")
    return handler


# ---------------------------------------------------------------------------
# Payload writers
# ---------------------------------------------------------------------------


def write_parquet(
    run_dir: RunDir,
    relpath: str,
    rows: Iterable[dict] | Any,
    *,
    also_csv: bool = True,
) -> Path:
    """Write ``rows`` to ``{run_dir}/{relpath}`` as parquet (+ optional CSV mirror).

    Accepts an iterable of dicts or an already-built ``pandas.DataFrame``.
    A CSV mirror is written by default when the table is small enough to
    read by eye (< 10k rows); disable with ``also_csv=False``.
    """
    import pandas as pd

    if hasattr(rows, "to_parquet"):  # already a DataFrame
        df = rows
    else:
        df = pd.DataFrame(list(rows))
    p = run_dir.path(relpath)
    df.to_parquet(p, index=False)
    run_dir.record_output(relpath)
    if also_csv and len(df) < 10_000:
        csv_rel = relpath[: -len(".parquet")] + ".csv" if relpath.endswith(".parquet") else relpath + ".csv"
        csv_path = run_dir.path(csv_rel)
        df.to_csv(csv_path, index=False)
        run_dir.record_output(csv_rel)
    return p


def write_npz(run_dir: RunDir, relpath: str, **arrays: Any) -> Path:
    """Write named numpy arrays to a ``.npz`` file inside the run dir."""
    import numpy as np

    p = run_dir.path(relpath)
    np.savez(p, **arrays)
    run_dir.record_output(relpath)
    return p


def write_json(run_dir: RunDir, relpath: str, payload: dict[str, Any]) -> Path:
    """Write a JSON payload (e.g. diagnostics.json) inside the run dir."""
    p = run_dir.path(relpath)
    p.write_text(json.dumps(payload, indent=2, default=_json_default))
    run_dir.record_output(relpath)
    return p


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Make ``name`` safe as a single path component."""
    bad = '/\\ \t\n"\''
    out = "".join("_" if c in bad else c for c in name)
    return out or "unnamed"


def _fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _git_sha() -> str | None:
    """Return the current git SHA, or None outside a repo / on error."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _ecosys_version() -> str:
    try:
        from importlib.metadata import version
        return version("ecosystem-complexity")
    except Exception:  # noqa: BLE001
        return "unknown"


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for numpy scalars, Paths, dataclasses."""
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(o)


def _config_to_dict(config: ModelConfig) -> dict[str, Any]:
    """Serialize a ModelConfig back to a plain-dict tree for YAML snapshotting.

    Uses ``dataclasses.asdict`` for the typed portions and preserves the
    ``*_raw`` sections verbatim. Enough to reproduce a run — not
    byte-identical to the source YAML (defaults are filled in).
    """
    d = dataclasses.asdict(config)
    # Frozen tuples become lists in asdict; that's the right shape for YAML.
    return d


# stdout/stderr passthrough of Python's print() is not captured into
# logs/run.log unless the caller uses stdlib logging. That's intentional:
# apps already emit via ``logging.info`` and we don't want to swallow
# subprocess output. Use ``attach_file_logger`` at app start.
_ = sys  # keep for future capture hook
