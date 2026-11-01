"""The Sandbox: one throwaway Docker container an agent can poke at safely."""

from __future__ import annotations

import base64
import io
import logging
import math
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

from .models import ExecResult, SandboxConfig
from .trajectory import Trajectory, TrajectoryHeader

log = logging.getLogger(__name__)

LABEL = "gauntlet.sandbox"
SNAPSHOT_REPO = "gauntlet-snapshot"
# GNU timeout exits 124 on TERM, and 137 (128 + SIGKILL) when it has to use KILL.
TIMEOUT_EXIT_CODES = {124, 137}


class SandboxError(RuntimeError):
    """Docker said no, or the sandbox isn't in a state to do what you asked."""


def _encode(data: bytes) -> dict[str, str]:
    try:
        return {"content": data.decode("utf-8"), "encoding": "utf-8"}
    except UnicodeDecodeError:
        return {"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"}


class Sandbox:
    """A locked-down container plus a recorder that logs every step.

    >>> with Sandbox() as sb:
    ...     sb.write_file("hello.py", "print('namaste')")
    ...     sb.exec("python hello.py").stdout
    'namaste\\n'
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        *,
        trajectory_path: str | Path | None = None,
        record: bool = True,
        metadata: dict[str, Any] | None = None,
        client: docker.DockerClient | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self.id = uuid.uuid4().hex[:12]
        self.metadata = metadata or {}
        self.trajectory: Trajectory | None = None
        self._trajectory_path = trajectory_path
        self._record = record
        self._client = client
        self._container: Any = None
        self._snapshot_count = 0

    # ------------------------------------------------------------------ lifecycle

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as e:
                raise SandboxError("Can't reach Docker. Is the daemon running?") from e
        return self._client

    @property
    def running(self) -> bool:
        return self._container is not None

    def start(self) -> Sandbox:
        if self._container is not None:
            return self
        cfg = self.config
        self._ensure_image(cfg.image)

        kwargs: dict[str, Any] = {
            "image": cfg.image,
            "command": ["tail", "-f", "/dev/null"],
            "name": f"gauntlet-{self.id}",
            "detach": True,
            "init": True,
            "working_dir": cfg.workdir,
            # Python caches bytecode keyed on (mtime seconds, size). An agent that rewrites a file
            # within the same second at the same size would silently run the old code.
            "environment": {"PYTHONDONTWRITEBYTECODE": "1", **cfg.env},
            "mem_limit": cfg.memory,
            "memswap_limit": cfg.memory,
            "nano_cpus": int(cfg.cpus * 1e9),
            "pids_limit": cfg.pids_limit,
            "network_disabled": not cfg.network,
            "security_opt": ["no-new-privileges"],
            "labels": {LABEL: "true", f"{LABEL}.id": self.id},
        }
        if cfg.user:
            kwargs["user"] = cfg.user
        if cfg.drop_capabilities:
            kwargs["cap_drop"] = ["ALL"]

        try:
            self._container = self.client.containers.run(**kwargs)
        except APIError as e:
            raise SandboxError(f"Couldn't start a container from {cfg.image}: {e.explanation or e}") from e

        try:
            self._preflight()
        except Exception:
            self.close()
            raise

        if self._record:
            header = TrajectoryHeader(
                sandbox_id=self.id,
                image=cfg.image,
                created_at=time.time(),
                config=cfg.model_dump(),
                metadata=self.metadata,
            )
            self.trajectory = Trajectory(header, path=self._trajectory_path)
        log.info("sandbox %s up on %s", self.id, cfg.image)
        return self

    def close(self) -> None:
        if self._container is None:
            return
        try:
            self._container.remove(force=True)
        except NotFound:
            pass
        except APIError as e:
            log.warning("couldn't remove sandbox %s: %s", self.id, e)
        finally:
            self._container = None

    def __enter__(self) -> Sandbox:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _ensure_image(self, image: str) -> None:
        try:
            self.client.images.get(image)
        except ImageNotFound:
            log.info("pulling %s, first run can take a minute", image)
            try:
                self.client.images.pull(image)
            except (APIError, NotFound) as e:
                raise SandboxError(f"Couldn't pull image {image}: {e}") from e

    def _preflight(self) -> None:
        code, _, _, _ = self._raw_exec(["sh", "-c", "command -v timeout >/dev/null"], workdir="/", limit=1024)
        if code != 0:
            raise SandboxError(
                f"Image {self.config.image} needs `sh` and `timeout`. "
                "Debian, Ubuntu and Alpine based images all work."
            )

    def _require(self) -> Any:
        if self._container is None:
            raise SandboxError("Sandbox isn't running. Call start() or use it as a context manager.")
        return self._container

    def _log(self, action: dict[str, Any], observation: dict[str, Any]) -> None:
        if self.trajectory is not None:
            self.trajectory.record(action, observation)

    # ------------------------------------------------------------------ exec

    def _raw_exec(
        self,
        cmd: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        limit: int,
    ) -> tuple[int, str, str, bool]:
        container = self._require()
        api = self.client.api
        exec_id = api.exec_create(
            container.id,
            cmd,
            stdout=True,
            stderr=True,
            workdir=workdir,
            environment=env,
            user=self.config.user or "",
        )["Id"]

        out, err = bytearray(), bytearray()
        truncated = False
        for chunk_out, chunk_err in api.exec_start(exec_id, stream=True, demux=True):
            for buf, chunk in ((out, chunk_out), (err, chunk_err)):
                if not chunk:
                    continue
                room = max(0, limit - len(buf))
                buf.extend(chunk[:room])
                if len(chunk) > room:
                    truncated = True

        exit_code = api.exec_inspect(exec_id).get("ExitCode")
        return (
            exit_code if exit_code is not None else -1,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            truncated,
        )

    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command. Output is capped and the whole process group dies at the timeout."""
        cfg = self.config
        secs = max(1, math.ceil(timeout or cfg.default_timeout))
        wrapped = ["timeout", "-s", "KILL", str(secs), "sh", "-c", command]

        started = time.monotonic()
        code, out, err, truncated = self._raw_exec(
            wrapped, workdir=workdir or cfg.workdir, env=env, limit=cfg.max_output_bytes
        )
        duration = time.monotonic() - started

        result = ExecResult(
            exit_code=code,
            stdout=out,
            stderr=err,
            duration=round(duration, 4),
            timed_out=code in TIMEOUT_EXIT_CODES and duration >= secs - 0.5,
            truncated=truncated,
        )
        self._log(
            {"type": "exec", "command": command, "timeout": secs, "workdir": workdir, "env": env},
            result.model_dump(),
        )
        return result

    # ------------------------------------------------------------------ files

    def _resolve(self, path: str) -> PurePosixPath:
        p = PurePosixPath(path)
        return p if p.is_absolute() else PurePosixPath(self.config.workdir) / p

    def write_file(self, path: str, content: str | bytes, *, mode: int = 0o644) -> str:
        container = self._require()
        data = content.encode("utf-8") if isinstance(content, str) else content
        target = self._resolve(path)

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=target.name)
            info.size = len(data)
            info.mode = mode
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))

        code, _, err, _ = self._raw_exec(["mkdir", "-p", str(target.parent)], limit=4096)
        if code != 0:
            raise SandboxError(f"Couldn't create {target.parent}: {err.strip()}")
        try:
            container.put_archive(str(target.parent), buf.getvalue())
        except APIError as e:
            raise SandboxError(f"Couldn't write {target}: {e.explanation or e}") from e

        self._log(
            {"type": "write_file", "path": str(target), "mode": mode, **_encode(data)},
            {"ok": True, "size": len(data)},
        )
        return str(target)

    def read_file(self, path: str) -> bytes:
        container = self._require()
        target = self._resolve(path)
        try:
            stream, _ = container.get_archive(str(target))
        except NotFound as e:
            self._log({"type": "read_file", "path": str(target)}, {"ok": False, "error": "not found"})
            raise FileNotFoundError(str(target)) from e

        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as tar:
            member = tar.next()
            fileobj = tar.extractfile(member) if member else None
            if fileobj is None:
                raise IsADirectoryError(str(target))
            data = fileobj.read()

        shown = data[: self.config.max_output_bytes]
        self._log(
            {"type": "read_file", "path": str(target)},
            {"ok": True, "size": len(data), "truncated": len(data) > len(shown), **_encode(shown)},
        )
        return data

    def read_text(self, path: str) -> str:
        return self.read_file(path).decode("utf-8")

    def put_archive(self, dest: str, data: bytes) -> str:
        """Unpack a tar archive (raw bytes) into `dest`, creating it if needed.

        Handy with `git archive`, which already speaks tar. When the trajectory is saved
        to disk, the archive is kept beside it as a blob so the step can be replayed.
        """
        container = self._require()
        target = self._resolve(dest)
        code, _, err, _ = self._raw_exec(["mkdir", "-p", str(target)], limit=4096)
        if code != 0:
            raise SandboxError(f"Couldn't create {target}: {err.strip()}")
        try:
            container.put_archive(str(target), data)
        except APIError as e:
            raise SandboxError(f"Couldn't unpack into {target}: {e.explanation or e}") from e
        digest = self.trajectory.store_blob(data) if self.trajectory is not None else None
        self._log(
            {"type": "put_archive", "dest": str(target), "size": len(data), "sha256": digest},
            {"ok": True},
        )
        return str(target)

    def copy_in(self, local_path: str | Path, dest: str) -> str:
        """Copy a local file, or the contents of a local directory, into `dest`."""
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(str(src))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            if src.is_dir():
                for child in sorted(src.iterdir()):
                    tar.add(child, arcname=child.name)
            else:
                tar.add(src, arcname=src.name)
        return self.put_archive(dest, buf.getvalue())

    # ------------------------------------------------------------------ snapshots

    def snapshot(self, label: str | None = None) -> str:
        """Freeze the filesystem into an image. Returns an image ref you can boot new sandboxes from."""
        container = self._require()
        self._snapshot_count += 1
        tag = f"{self.id}-{self._snapshot_count}"
        try:
            image = container.commit(
                repository=SNAPSHOT_REPO,
                tag=tag,
                conf={"Labels": {f"{LABEL}.snapshot": "true", f"{LABEL}.parent": self.id}},
            )
        except APIError as e:
            raise SandboxError(f"Snapshot failed: {e.explanation or e}") from e
        ref = f"{SNAPSHOT_REPO}:{tag}"
        self._log({"type": "snapshot", "label": label}, {"ok": True, "image": ref, "image_id": image.id})
        return ref

    def fork(self, *, trajectory_path: str | Path | None = None, **overrides: Any) -> Sandbox:
        """Snapshot this sandbox and boot a twin from it. The twin starts a fresh trajectory."""
        ref = self.snapshot(label="fork")
        config = self.config.model_copy(update={"image": ref, **overrides})
        step = len(self.trajectory) - 1 if self.trajectory else None
        child = Sandbox(
            config,
            trajectory_path=trajectory_path,
            record=self._record,
            metadata={**self.metadata, "forked_from": self.id, "forked_at_step": step},
            client=self.client,
        )
        return child.start()

    # ------------------------------------------------------------------ housekeeping

    @staticmethod
    def cleanup_all(client: docker.DockerClient | None = None, *, snapshots: bool = False) -> int:
        """Remove every gauntlet container (and optionally snapshot images). Returns how many went."""
        client = client or docker.from_env()
        removed = 0
        for c in client.containers.list(all=True, filters={"label": LABEL}):
            try:
                c.remove(force=True)
                removed += 1
            except NotFound:
                pass
        if snapshots:
            # only images still tagged as raw snapshots. Images built on top of one (like
            # repo2env's env images) inherit the label but carry their own name, so they stay.
            for img in client.images.list(name=SNAPSHOT_REPO):
                for tag in img.tags:
                    if not tag.startswith(f"{SNAPSHOT_REPO}:"):
                        continue
                    try:
                        client.images.remove(tag)
                        removed += 1
                    except (NotFound, APIError):
                        pass
        return removed

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return f"<Sandbox {self.id} {self.config.image} {state}>"
