import json

import pytest

from gauntlet_sandbox import Trajectory, TrajectoryHeader


def _header():
    return TrajectoryHeader(sandbox_id="abc", image="python:3.12-slim", created_at=1.0)


def test_record_and_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    traj = Trajectory(_header(), path=path)
    traj.record({"type": "exec", "command": "ls"}, {"exit_code": 0, "stdout": "a\n"})
    traj.record({"type": "exec", "command": "pwd"}, {"exit_code": 0, "stdout": "/\n"})

    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["type"] == "header"
    assert len(lines) == 3

    loaded = Trajectory.load(path)
    assert loaded.header.sandbox_id == "abc"
    assert [s.action["command"] for s in loaded] == ["ls", "pwd"]
    assert [s.index for s in loaded] == [0, 1]


def test_load_does_not_overwrite(tmp_path):
    path = tmp_path / "t.jsonl"
    Trajectory(_header(), path=path).record({"type": "exec"}, {})
    before = path.read_text()
    Trajectory.load(path)
    assert path.read_text() == before


def test_in_memory_only():
    traj = Trajectory(_header())
    traj.record({"type": "exec"}, {"exit_code": 1})
    assert traj.to_dict()["steps"][0]["observation"]["exit_code"] == 1


def test_bad_files(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type": "step", "index": 0, "timestamp": 1, "action": {}, "observation": {}}\n')
    with pytest.raises(ValueError, match="no header"):
        Trajectory.load(p)

    p.write_text('{"type": "weird"}\n')
    with pytest.raises(ValueError, match="unknown record type"):
        Trajectory.load(p)
