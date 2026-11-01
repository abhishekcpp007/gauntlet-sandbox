"""HTTP API so agents in any language can drive sandboxes.

Run it with `gauntlet-sandbox serve`. It binds to localhost by default, and you should
set GAUNTLET_API_KEY before exposing it anywhere else, since it runs arbitrary code.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .models import ExecResult, SandboxConfig
from .sandbox import Sandbox, SandboxError

log = logging.getLogger(__name__)


class ServerSettings(BaseModel):
    trajectory_dir: Path = Path("trajectories")
    max_sandboxes: int = Field(16, ge=1)
    idle_ttl: float = Field(900.0, gt=0, description="Seconds before an idle sandbox is reaped")
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> ServerSettings:
        env = os.environ
        return cls(
            trajectory_dir=Path(env.get("GAUNTLET_TRAJECTORY_DIR", "trajectories")),
            max_sandboxes=int(env.get("GAUNTLET_MAX_SANDBOXES", 16)),
            idle_ttl=float(env.get("GAUNTLET_IDLE_TTL", 900)),
            api_key=env.get("GAUNTLET_API_KEY") or None,
        )


# ---------------------------------------------------------------------- schemas


class CreateSandboxRequest(SandboxConfig):
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecRequest(BaseModel):
    command: str = Field(min_length=1)
    timeout: float | None = Field(None, gt=0, le=3600)
    workdir: str | None = None
    env: dict[str, str] | None = None


class FilePayload(BaseModel):
    path: str = Field(min_length=1)
    content: str
    encoding: Literal["utf-8", "base64"] = "utf-8"


class SnapshotRequest(BaseModel):
    label: str | None = None


class SandboxInfo(BaseModel):
    id: str
    image: str
    created_at: float
    last_used: float
    steps: int
    trajectory_path: str | None


# ---------------------------------------------------------------------- registry


@dataclass
class _Entry:
    sandbox: Sandbox
    trajectory_path: Path | None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    busy: int = 0

    def info(self) -> SandboxInfo:
        sb = self.sandbox
        return SandboxInfo(
            id=sb.id,
            image=sb.config.image,
            created_at=self.created_at,
            last_used=self.last_used,
            steps=len(sb.trajectory) if sb.trajectory else 0,
            trajectory_path=str(self.trajectory_path) if self.trajectory_path else None,
        )


class Registry:
    """Tracks live sandboxes, enforces the cap and reaps idle ones."""

    def __init__(self, settings: ServerSettings, client: Any = None) -> None:
        self.settings = settings
        self.client = client
        self._items: dict[str, _Entry] = {}
        self._pending = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None

    def create(self, config: SandboxConfig, metadata: dict[str, Any]) -> _Entry:
        with self._lock:
            if len(self._items) + self._pending >= self.settings.max_sandboxes:
                raise HTTPException(429, f"Sandbox limit reached ({self.settings.max_sandboxes})")
            self._pending += 1
        try:
            sb = Sandbox(config, metadata=metadata, client=self.client)
            path = self.settings.trajectory_dir / f"{sb.id}.jsonl"
            sb._trajectory_path = path
            sb.start()
            entry = _Entry(sb, path)
            with self._lock:
                self._items[sb.id] = entry
            return entry
        finally:
            with self._lock:
                self._pending -= 1

    def list(self) -> list[_Entry]:
        with self._lock:
            return list(self._items.values())

    def get(self, sid: str) -> _Entry:
        with self._lock:
            entry = self._items.get(sid)
        if entry is None:
            raise HTTPException(404, f"No sandbox {sid}")
        return entry

    @contextmanager
    def use(self, sid: str) -> Iterator[Sandbox]:
        """Borrow a sandbox. The reaper won't touch it while it's borrowed."""
        with self._lock:
            entry = self._items.get(sid)
            if entry is None:
                raise HTTPException(404, f"No sandbox {sid}")
            entry.busy += 1
        try:
            yield entry.sandbox
        finally:
            with self._lock:
                entry.busy -= 1
                entry.last_used = time.time()

    def remove(self, sid: str) -> None:
        with self._lock:
            entry = self._items.pop(sid, None)
        if entry is None:
            raise HTTPException(404, f"No sandbox {sid}")
        entry.sandbox.close()

    def reap(self) -> list[str]:
        cutoff = time.time() - self.settings.idle_ttl
        with self._lock:
            stale = [sid for sid, e in self._items.items() if e.busy == 0 and e.last_used < cutoff]
            entries = [self._items.pop(sid) for sid in stale]
        for e in entries:
            log.info("reaping idle sandbox %s", e.sandbox.id)
            e.sandbox.close()
        return stale

    def start_reaper(self) -> None:
        interval = max(1.0, min(30.0, self.settings.idle_ttl / 2))

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.reap()
                except Exception:  # keep the reaper alive no matter what
                    log.exception("reaper hiccup")

        self._reaper = threading.Thread(target=loop, name="gauntlet-reaper", daemon=True)
        self._reaper.start()

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            entries = list(self._items.values())
            self._items.clear()
        for e in entries:
            e.sandbox.close()


# ---------------------------------------------------------------------- app


def create_app(settings: ServerSettings | None = None, *, client: Any = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    registry = Registry(settings, client=client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry.start_reaper()
        yield
        registry.shutdown()

    app = FastAPI(
        title="gauntlet-sandbox",
        version=__version__,
        description="Locked-down Docker sandboxes for AI agents, with recorded trajectories.",
        lifespan=lifespan,
    )
    app.state.registry = registry

    def check_auth(authorization: str | None = Header(None)) -> None:
        if settings.api_key is None:
            return
        expected = f"Bearer {settings.api_key}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(401, "Missing or wrong API key", {"WWW-Authenticate": "Bearer"})

    @app.exception_handler(SandboxError)
    async def sandbox_error(_: Request, exc: SandboxError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "sandboxes": len(registry.list())}

    api = APIRouter(prefix="/sandboxes", dependencies=[Depends(check_auth)])

    @api.post("", status_code=201)
    def create_sandbox(req: CreateSandboxRequest) -> SandboxInfo:
        config = SandboxConfig(**req.model_dump(exclude={"metadata"}))
        return registry.create(config, req.metadata).info()

    @api.get("")
    def list_sandboxes() -> list[SandboxInfo]:
        return [e.info() for e in registry.list()]

    @api.get("/{sid}")
    def get_sandbox(sid: str) -> SandboxInfo:
        return registry.get(sid).info()

    @api.delete("/{sid}", status_code=204)
    def delete_sandbox(sid: str) -> None:
        registry.remove(sid)

    @api.post("/{sid}/exec")
    def exec_command(sid: str, req: ExecRequest) -> ExecResult:
        with registry.use(sid) as sb:
            return sb.exec(req.command, timeout=req.timeout, workdir=req.workdir, env=req.env)

    @api.put("/{sid}/files")
    def write_file(sid: str, req: FilePayload) -> dict[str, Any]:
        if req.encoding == "base64":
            try:
                data: bytes = base64.b64decode(req.content, validate=True)
            except binascii.Error as e:
                raise HTTPException(422, "content isn't valid base64") from e
        else:
            data = req.content.encode("utf-8")
        with registry.use(sid) as sb:
            path = sb.write_file(req.path, data)
        return {"path": path, "size": len(data)}

    @api.get("/{sid}/files")
    def read_file(sid: str, path: str = Query(min_length=1)) -> FilePayload:
        with registry.use(sid) as sb:
            try:
                data = sb.read_file(path)
            except FileNotFoundError as e:
                raise HTTPException(404, f"No such file: {e}") from e
            except IsADirectoryError as e:
                raise HTTPException(400, f"That's a directory: {e}") from e
        try:
            return FilePayload(path=path, content=data.decode("utf-8"), encoding="utf-8")
        except UnicodeDecodeError:
            return FilePayload(path=path, content=base64.b64encode(data).decode(), encoding="base64")

    @api.post("/{sid}/snapshots", status_code=201)
    def snapshot(sid: str, req: SnapshotRequest | None = None) -> dict[str, str]:
        with registry.use(sid) as sb:
            return {"image": sb.snapshot(label=req.label if req else None)}

    @api.get("/{sid}/trajectory")
    def trajectory(sid: str) -> dict[str, Any]:
        sb = registry.get(sid).sandbox
        if sb.trajectory is None:
            raise HTTPException(404, "This sandbox isn't recording")
        return sb.trajectory.to_dict()

    app.include_router(api)
    return app
