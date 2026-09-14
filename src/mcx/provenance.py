"""Run provenance. Every results table carries these columns.

The point is falsifiability: given a results file, you can say exactly which code and
which config produced it, and re-run to a byte-identical result.
"""

from __future__ import annotations

import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from mcx.config import REPO_ROOT, Config

CODE_VERSION = "0.1.0"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_sha() -> str:
    return _git("rev-parse", "--short", "HEAD") or "uncommitted"


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    stage: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    code_version: str
    python_version: str
    platform: str
    started_utc: str

    @classmethod
    def create(cls, stage: str, cfg: Config) -> RunInfo:
        return cls(
            run_id=uuid.uuid4().hex[:12],
            stage=stage,
            git_sha=git_sha(),
            git_dirty=git_dirty(),
            config_hash=cfg.config_hash,
            code_version=CODE_VERSION,
            python_version=platform.python_version(),
            platform=platform.platform(),
            started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    def stamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach provenance columns to a results frame."""
        out = df.copy()
        for key in ("run_id", "stage", "git_sha", "config_hash", "code_version"):
            out[key] = getattr(self, key)
        return out


def write_table(df: pd.DataFrame, path: Path, run: RunInfo, *, also_csv: bool = True) -> Path:
    """Write a results table deterministically: stable column and row order.

    Sorting before writing is what makes the two-clean-runs-are-identical check in
    Tier 4 meaningful.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = run.stamp(df)
    key_cols = [c for c in stamped.columns if c not in {"run_id", "started_utc"}]
    stamped = stamped.sort_values(key_cols, kind="stable").reset_index(drop=True)
    stamped.to_parquet(path, index=False)
    if also_csv:
        stamped.to_csv(path.with_suffix(".csv"), index=False)
    return path


def run_manifest(run: RunInfo) -> dict[str, object]:
    return asdict(run)
