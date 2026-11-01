import base64

import pytest
from fastapi.testclient import TestClient

from gauntlet_sandbox.server import ServerSettings, create_app

from .conftest import IMAGE


@pytest.fixture
def client(tmp_path):
    settings = ServerSettings(trajectory_dir=tmp_path, max_sandboxes=2)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["ok"] is True


def test_unknown_sandbox(client):
    assert client.post("/sandboxes/nope/exec", json={"command": "ls"}).status_code == 404


def test_auth(tmp_path):
    settings = ServerSettings(trajectory_dir=tmp_path, api_key="s3cret")
    with TestClient(create_app(settings)) as c:
        assert c.get("/sandboxes").status_code == 401
        assert c.get("/sandboxes", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/sandboxes", headers={"Authorization": "Bearer s3cret"}).status_code == 200


@pytest.mark.docker
def test_full_flow(client, tmp_path):
    r = client.post("/sandboxes", json={"image": IMAGE, "metadata": {"task": "demo"}})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    r = client.put(f"/sandboxes/{sid}/files", json={"path": "x.py", "content": "print(6 * 7)"})
    assert r.json()["path"] == "/workspace/x.py"

    r = client.post(f"/sandboxes/{sid}/exec", json={"command": "python x.py"})
    assert r.json()["stdout"] == "42\n"

    raw = b"\x00\xffbin"
    client.put(
        f"/sandboxes/{sid}/files",
        json={"path": "b.bin", "content": base64.b64encode(raw).decode(), "encoding": "base64"},
    )
    r = client.get(f"/sandboxes/{sid}/files", params={"path": "b.bin"})
    assert r.json()["encoding"] == "base64"
    assert base64.b64decode(r.json()["content"]) == raw

    assert client.get(f"/sandboxes/{sid}/files", params={"path": "missing"}).status_code == 404

    traj = client.get(f"/sandboxes/{sid}/trajectory").json()
    assert traj["header"]["metadata"] == {"task": "demo"}
    assert len(traj["steps"]) == 5
    assert (tmp_path / f"{sid}.jsonl").exists()

    assert client.delete(f"/sandboxes/{sid}").status_code == 204
    assert client.get(f"/sandboxes/{sid}").status_code == 404


@pytest.mark.docker
def test_sandbox_limit(client):
    ids = [client.post("/sandboxes", json={"image": IMAGE}).json()["id"] for _ in range(2)]
    assert client.post("/sandboxes", json={"image": IMAGE}).status_code == 429
    for sid in ids:
        client.delete(f"/sandboxes/{sid}")


@pytest.mark.docker
def test_reaper(client):
    sid = client.post("/sandboxes", json={"image": IMAGE}).json()["id"]
    registry = client.app.state.registry
    registry.settings.idle_ttl = 0.001
    assert registry.reap() == [sid]
    assert client.get(f"/sandboxes/{sid}").status_code == 404
