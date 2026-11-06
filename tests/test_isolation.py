"""The README promises no network, capped CPU, memory and pids, and every capability dropped.

Network had a test. The rest were configuration that nobody checked actually reached the
kernel, which is the kind of claim that quietly stops being true after a Docker upgrade or a
change to how the container is started. These assert the state inside a running sandbox and
that the two limits worth fearing really do bite.
"""

import pytest

from gauntlet_sandbox import Sandbox, SandboxConfig

from .conftest import IMAGE

pytestmark = pytest.mark.docker

FORK_BOMB = """
import os
n = 0
try:
    while n < 500:
        if os.fork() == 0:
            os._exit(0)
        n += 1
except BlockingIOError:
    print("stopped after", n)
else:
    print("unbounded", n)
"""

MEMORY_HOG = """
b = bytearray()
while True:
    b.extend(b"x" * (10 * 1024 * 1024))
"""


def test_every_capability_is_dropped(sandbox):
    """Empty permitted and effective sets, so nothing can be raised later either."""
    res = sandbox.exec("grep -E '^CapPrm|^CapEff' /proc/self/status")
    caps = dict(line.split(":\t") for line in res.stdout.strip().splitlines())
    assert set(caps) == {"CapPrm", "CapEff"}
    assert all(int(v, 16) == 0 for v in caps.values()), caps


def test_a_dropped_capability_actually_stops_the_syscall(sandbox):
    """The bitmask is the claim; this is the behaviour behind it."""
    res = sandbox.exec("mknod /tmp/node c 1 3")
    assert res.exit_code != 0
    assert "not permitted" in (res.stderr + res.stdout).lower()


def test_privileges_cannot_be_raised(sandbox):
    res = sandbox.exec("grep NoNewPrivs /proc/self/status")
    assert res.stdout.split()[-1] == "1"


def test_the_limits_reach_the_kernel():
    cfg = SandboxConfig(image=IMAGE, memory="256m", pids_limit=32)
    with Sandbox(cfg) as sb:
        pids = sb.exec("cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max")
        mem = sb.exec(
            "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"
        )
        assert pids.stdout.strip() == "32"
        assert int(mem.stdout.strip()) == 256 * 1024 * 1024


def test_a_fork_bomb_is_contained():
    cfg = SandboxConfig(image=IMAGE, memory="256m", pids_limit=32)
    with Sandbox(cfg) as sb:
        sb.write_file("fork.py", FORK_BOMB)
        res = sb.exec("python /workspace/fork.py", timeout=30)
        assert "stopped after" in res.stdout, res
        assert sb.exec("echo alive").stdout.strip() == "alive"  # and the sandbox survived it


def test_a_memory_hog_is_killed_and_the_sandbox_lives():
    cfg = SandboxConfig(image=IMAGE, memory="256m", pids_limit=32)
    with Sandbox(cfg) as sb:
        sb.write_file("hog.py", MEMORY_HOG)
        res = sb.exec("python /workspace/hog.py", timeout=60)
        assert res.exit_code != 0  # 137 when the OOM killer takes it
        assert sb.exec("echo alive").stdout.strip() == "alive"
