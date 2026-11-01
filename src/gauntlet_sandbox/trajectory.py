"""Trajectories: an append-only log of every action a sandbox ran and what it saw.

On disk it's JSONL. The first line is a header, every line after it is one step.
Steps get flushed as they happen, so a crash mid-run still leaves a usable file.
Big binary payloads (tar archives) go next to it in `<name>.blobs/<sha256>`, so the
JSONL stays readable and the run can still be replayed byte for byte.
"""

from __future__ import annotations

import hashlib
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
        self.source: Path | None = self.path  # where it was written or loaded from
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

    @property
    def blob_dir(self) -> Path | None:
        if self.source is None:
            return None
        return self.source.with_name(self.source.stem + ".blobs")

    def store_blob(self, data: bytes) -> str:
        """Keep a payload next to the trajectory file. Returns its sha256 either way."""
        digest = hashlib.sha256(data).hexdigest()
        if self.path is not None and self.blob_dir is not None:
            self.blob_dir.mkdir(parents=True, exist_ok=True)
            target = self.blob_dir / digest
            if not target.exists():
                target.write_bytes(data)
        return digest

    def load_blob(self, digest: str) -> bytes:
        target = self.blob_dir / digest if self.blob_dir else None
        if target is None or not target.exists():
            raise FileNotFoundError(f"blob {digest[:12]} isn't stored next to this trajectory")
        return target.read_bytes()

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
        traj.source = path
        return traj

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)
