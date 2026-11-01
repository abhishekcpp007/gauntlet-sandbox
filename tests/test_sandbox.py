import pytest
from docker.errors import NotFound

from gauntlet_sandbox import Sandbox, SandboxConfig, SandboxError, Trajectory

from .conftest import IMAGE

pytestmark = pytest.mark.docker


def test_exec_basic(sandbox):
    res = sandbox.exec("echo hello && pwd")
    assert res.ok
    assert res.stdout == "hello\n/workspace\n"


def test_stderr_and_exit_code(sandbox):
    res = sandbox.exec("echo oops >&2; exit 3")
    assert res.exit_code == 3
    assert res.stderr == "oops\n"
    assert res.stdout == ""
    assert not res.ok


def test_timeout_kills_process(sandbox):
    res = sandbox.exec("sleep 30", timeout=1)
    assert res.timed_out
    assert res.duration < 10
    # nothing left running from that command
    assert "sleep" not in sandbox.exec("ps -eo comm 2>/dev/null || ls /proc").stdout


def test_network_off_by_default(sandbox):
    res = sandbox.exec(
        "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\"", timeout=10
    )
    assert res.exit_code != 0


def test_output_is_capped():
    cfg = SandboxConfig(image=IMAGE, max_output_bytes=1024)
    with Sandbox(cfg) as sb:
        res = sb.exec("python -c \"print('x' * 200000)\"")
    assert res.truncated
    assert len(res.stdout) == 1024


def test_files_roundtrip(sandbox):
    path = sandbox.write_file("pkg/hello.py", "print('namaste')\n")
    assert path == "/workspace/pkg/hello.py"
    assert sandbox.read_text("pkg/hello.py") == "print('namaste')\n"
    assert sandbox.exec("python pkg/hello.py").stdout == "namaste\n"

    blob = bytes(range(256))
    sandbox.write_file("/tmp/blob.bin", blob)
    assert sandbox.read_file("/tmp/blob.bin") == blob


def test_quick_rewrite_runs_new_code(sandbox):
    # same size, same second: stale __pycache__ used to win here
    sandbox.write_file("test_mod.py", "from mod import f\nprint(f())\n")
    sandbox.write_file("mod.py", "def f():\n    return 'AAA'\n")
    assert sandbox.exec("python test_mod.py").stdout == "AAA\n"
    sandbox.write_file("mod.py", "def f():\n    return 'BBB'\n")
    assert sandbox.exec("python test_mod.py").stdout == "BBB\n"


def test_read_missing_file(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.read_file("nope.txt")


def test_trajectory_is_recorded(sandbox, tmp_path):
    sandbox.write_file("a.txt", "hi")
    sandbox.exec("cat a.txt")
    sandbox.read_file("a.txt")

    loaded = Trajectory.load(tmp_path / "traj.jsonl")
    assert [s.action["type"] for s in loaded] == ["write_file", "exec", "read_file"]
    assert loaded.steps[0].action["content"] == "hi"
    assert loaded.steps[1].observation["stdout"] == "hi"


def test_fork_is_isolated(sandbox):
    sandbox.write_file("state.txt", "v1")
    child = sandbox.fork()
    try:
        assert child.read_text("state.txt") == "v1"
        child.write_file("state.txt", "child")
        sandbox.write_file("state.txt", "parent")
        assert child.read_text("state.txt") == "child"
        assert sandbox.read_text("state.txt") == "parent"
        assert child.metadata["forked_from"] == sandbox.id
    finally:
        child.close()


def test_close_is_idempotent():
    sb = Sandbox(SandboxConfig(image=IMAGE)).start()
    cid = sb._container.id
    sb.close()
    sb.close()
    assert not sb.running
    with pytest.raises(SandboxError):
        sb.exec("true")
    with pytest.raises(NotFound):
        sb.client.containers.get(cid)


def test_bad_image():
    with pytest.raises(SandboxError):
        Sandbox(SandboxConfig(image="gauntlet/does-not-exist:never")).start()
