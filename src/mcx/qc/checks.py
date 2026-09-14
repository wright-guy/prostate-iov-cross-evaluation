"""QC findings and the gate that acts on them.

A finding is a row, not a log line: every check that runs contributes a row to
``results/qc_findings.parquet`` whether it passed or failed, so "was this checked?"
is answerable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from mcx.config import Config


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    WAIVED = "WAIVED"


@dataclass
class Finding:
    check: str
    status: Status
    message: str
    tier: int = 0
    patient: str | None = None
    contour_set: str | None = None
    structure: str | None = None
    value: float | None = None
    expected: str | None = None
    waiver_reason: str | None = None

    def scope(self) -> dict[str, str]:
        return {
            k: v
            for k, v in (
                ("patient", self.patient),
                ("contour_set", self.contour_set),
                ("structure", self.structure),
            )
            if v is not None
        }


class QCLog:
    """Collects findings, applies waivers, and decides whether the stage may pass."""

    def __init__(self, cfg: Config, stage: str) -> None:
        self.cfg = cfg
        self.stage = stage
        self.findings: list[Finding] = []

    # ------------------------------------------------------------------ record

    def add(self, finding: Finding) -> Finding:
        if finding.status is Status.FAIL:
            waiver = self.cfg.waiver_for(finding.check, **finding.scope())
            if waiver is not None:
                finding.status = Status.WAIVED
                finding.waiver_reason = " ".join(str(waiver["reason"]).split())
        self.findings.append(finding)
        return finding

    def check(
        self,
        check: str,
        ok: bool,
        message: str,
        *,
        tier: int = 0,
        warn_only: bool = False,
        **kw: Any,
    ) -> Finding:
        status = Status.PASS if ok else (Status.WARN if warn_only else Status.FAIL)
        return self.add(Finding(check=check, status=status, message=message, tier=tier, **kw))

    # ------------------------------------------------------------------ report

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=[
                    "stage", "tier", "check", "status", "patient", "contour_set",
                    "structure", "value", "expected", "message", "waiver_reason",
                ]
            )
        rows = []
        for f in self.findings:
            rows.append(
                {
                    "stage": self.stage,
                    "tier": f.tier,
                    "check": f.check,
                    "status": f.status.value,
                    "patient": f.patient,
                    "contour_set": f.contour_set,
                    "structure": f.structure,
                    "value": f.value,
                    "expected": f.expected,
                    "message": f.message,
                    "waiver_reason": f.waiver_reason,
                }
            )
        return pd.DataFrame(rows)

    def counts(self) -> dict[str, int]:
        df = self.to_frame()
        if df.empty:
            return {s.value: 0 for s in Status}
        c = df["status"].value_counts().to_dict()
        return {s.value: int(c.get(s.value, 0)) for s in Status}

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.status is Status.FAIL]

    def summary(self) -> str:
        c = self.counts()
        return (
            f"{self.stage}: {c['PASS']} pass, {c['WARN']} warn, "
            f"{c['WAIVED']} waived, {c['FAIL']} FAIL"
        )


class GateFailure(RuntimeError):
    """Raised when a stage produced unwaived failures."""


def enforce(log: QCLog) -> None:
    """The gate. Unwaived failures stop the pipeline."""
    fails = log.failures
    if not fails:
        return
    lines = [f"{log.stage} gate failed on {len(fails)} check(s):", ""]
    for f in fails[:40]:
        scope = " ".join(f"{k}={v}" for k, v in f.scope().items())
        lines.append(f"  [{f.check}] {scope} — {f.message}")
    if len(fails) > 40:
        lines.append(f"  ... and {len(fails) - 40} more (see results/qc_findings.parquet)")
    lines += [
        "",
        "Fix the data, fix the code, or add a justified entry to config/waivers.yaml.",
    ]
    raise GateFailure("\n".join(lines))
