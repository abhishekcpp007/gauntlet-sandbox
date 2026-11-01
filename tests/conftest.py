import docker
import pytest

from gauntlet_sandbox import Sandbox, SandboxConfig

IMAGE = "python:3.12-slim"


def _docker_ok() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


DOCKER_OK = _docker_ok()


def pytest_collection_modifyitems(config, items):
    if DOCKER_OK:
        return
    skip = pytest.mark.skip(reason="Docker daemon not reachable")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sandbox(tmp_path):
    with Sandbox(SandboxConfig(image=IMAGE), trajectory_path=tmp_path / "traj.jsonl") as sb:
        yield sb


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    if DOCKER_OK:
        Sandbox.cleanup_all(snapshots=True)
