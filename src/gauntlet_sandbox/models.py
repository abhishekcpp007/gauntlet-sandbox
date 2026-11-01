from __future__ import annotations

from pydantic import BaseModel, Field


class SandboxConfig(BaseModel):
    """How a sandbox boots. Defaults are locked down: no network, capped CPU, memory and pids."""

    image: str = "python:3.12-slim"
    memory: str = Field("512m", description="Docker memory limit, like 512m or 2g")
    cpus: float = Field(1.0, gt=0, le=64)
    pids_limit: int = Field(256, ge=16)
    network: bool = False
    workdir: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    user: str | None = None
    drop_capabilities: bool = True
    default_timeout: float = Field(30.0, gt=0, le=3600)
    max_output_bytes: int = Field(64_000, ge=1024)


class ExecResult(BaseModel):
    """What came back from one command."""

    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out
