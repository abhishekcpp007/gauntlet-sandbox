"""Trajectories: an append-only log of every action a sandbox ran and what it saw.

On disk it's JSONL. The first line is a header, every line after it is one step.
Steps get flushed as they happen, so a crash mid-run still leaves a usable file.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

FORMAT = "gauntlet.trajectory/v1"


class TrajectoryHeader(BaseModel):
    format: str = FORMAT
    sandbox_id: str
    image: str
    created_at: float
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    index: int
    timestamp: float
    action: dict[str, Any]
    observation: dict[str, Any]


class Trajectory:
    def __init__(self, header: TrajectoryHeader, path: str | Path | None = None) -> None:
        self.header = header
        self.steps: list[Step] = []
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write({"type": "header", **header.model_dump()}, mode="w")

    def record(self, action: dict[str, Any], observation: dict[str, Any]) -> Step:
        with self._lock:
            step = Step(
                index=len(self.steps),
                timestamp=time.time(),
                action=action,
                observation=observation,
            )
            self.steps.append(step)
            if self.path:
                self._write({"type": "step", **step.model_dump()})
        return step

    def _write(self, obj: dict[str, Any], mode: str = "a") -> None:
        assert self.path is not None
        with self.path.open(mode, encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.model_dump(),
            "steps": [s.model_dump() for s in self.steps],
        }

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        path = Path(path)
        header: TrajectoryHeader | None = None
        steps: list[Step] = []
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                kind = obj.pop("type", None)
                if kind == "header":
                    header = TrajectoryHeader(**obj)
                elif kind == "step":
                    steps.append(Step(**obj))
                else:
                    raise ValueError(f"{path}:{lineno}: unknown record type {kind!r}")
        if header is None:
            raise ValueError(f"{path}: no header line, is this an gauntlet trajectory?")
        traj = cls(header)  # no path, so loading never overwrites the file
        traj.steps = steps
        return traj

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)
