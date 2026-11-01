# gauntlet-sandbox

Run AI agent actions inside locked-down Docker containers, and get a replayable record of every step.

If you're building an agent, an RL environment or an eval harness, you need somewhere safe for the model to run `rm -rf` and `pip install` without touching your laptop. You also want to know exactly what it did afterwards. This gives you both in one small library.

```python
from gauntlet_sandbox import Sandbox

with Sandbox(trajectory_path="run.jsonl") as sb:
    sb.write_file("solution.py", "def add(a, b):\n    return a + b\n")
    sb.write_file("test_solution.py", "from solution import add\nassert add(2, 3) == 5\nprint('pass')\n")
    result = sb.exec("python test_solution.py")
    print(result.ok, result.stdout)  # True pass
```

Every action lands in `run.jsonl` as it happens:

```
$ gauntlet-sandbox show run.jsonl
trajectory 3f9c1a2b7d4e on python:3.12-slim, 3 steps

#0 write /workspace/solution.py (35 bytes)

#1 write /workspace/test_solution.py (63 bytes)

#2 $ python test_solution.py   -> exit 0 in 0.08s
     pass
```

## What you get

- **Safe defaults.** No network, 512 MB of memory, 1 CPU, a pid cap, every Linux capability dropped and `no-new-privileges` on. You opt into more, never out of less.
- **Real timeouts.** Commands that hang get killed at the deadline, along with every child process they spawned. `result.timed_out` tells you it happened.
- **Capped output.** An agent that prints 2 GB won't take your process down with it. You get the first `max_output_bytes` and a `truncated` flag.
- **Trajectories.** One JSONL file per sandbox, written step by step so a crash doesn't lose anything. Load it back with `Trajectory.load()`.
- **Bulk copies.** `sb.copy_in("./my-repo", "repo")` drops a whole directory in, and `sb.put_archive()` takes raw tar bytes straight from `git archive`.
- **Snapshots and forks.** `sb.snapshot()` freezes the filesystem into an image. `sb.fork()` boots an isolated twin from it, which is how you branch an agent run at step N and try something else.
- **No stale bytecode.** `PYTHONDONTWRITEBYTECODE=1` is on by default. Without it, an agent that rewrites a file twice in one second at the same size runs the old code, and you'll spend an hour wondering why its correct fix still fails.
- **An HTTP API.** So agents written in any language, or running on another box, can use it too.

## Install

```bash
pip install gauntlet-sandbox            # library + CLI
pip install "gauntlet-sandbox[server]"  # adds the HTTP API
```

You'll need Docker running. The default image is `python:3.12-slim`, and any Debian, Ubuntu or Alpine based image works.

## CLI

```bash
# one command in a fresh box
gauntlet-sandbox run -- python -c "print(6 * 7)"

# poke around interactively, and keep the trajectory
gauntlet-sandbox repl --image node:20-slim --record session.jsonl

# read a trajectory back
gauntlet-sandbox show session.jsonl

# clean up anything left behind
gauntlet-sandbox gc --snapshots
```

## HTTP API

```bash
GAUNTLET_API_KEY=pick-something gauntlet-sandbox serve --port 8000
```

| Method | Path | What it does |
|---|---|---|
| `POST` | `/sandboxes` | Start one. Body is any `SandboxConfig` field plus `metadata` |
| `GET` | `/sandboxes` | List live ones |
| `POST` | `/sandboxes/{id}/exec` | `{"command": "...", "timeout": 10}` |
| `PUT` | `/sandboxes/{id}/files` | `{"path": "...", "content": "...", "encoding": "utf-8" or "base64"}` |
| `GET` | `/sandboxes/{id}/files?path=...` | Read a file back |
| `POST` | `/sandboxes/{id}/snapshots` | Freeze it into an image |
| `GET` | `/sandboxes/{id}/trajectory` | Everything it's done so far |
| `DELETE` | `/sandboxes/{id}` | Tear it down |

Interactive docs live at `/docs` once it's running. A few env vars tune the server: `GAUNTLET_MAX_SANDBOXES` (default 16), `GAUNTLET_IDLE_TTL` in seconds (default 900, idle boxes get reaped after that) and `GAUNTLET_TRAJECTORY_DIR`.

The server binds to `127.0.0.1` by default. It runs whatever code it's sent, so set `GAUNTLET_API_KEY` before you put it anywhere else.

## Config

```python
from gauntlet_sandbox import Sandbox, SandboxConfig

cfg = SandboxConfig(
    image="python:3.12-slim",
    memory="1g",
    cpus=2,
    network=True,  # off by default
    env={"PYTHONUNBUFFERED": "1"},
    default_timeout=60,
    max_output_bytes=128_000,
)
```

## Trajectory format

Plain JSONL. Line one is a header, then one line per step:

```json
{"type": "header", "format": "gauntlet.trajectory/v1", "sandbox_id": "...", "image": "...", "config": {}, "metadata": {}}
{"type": "step", "index": 0, "timestamp": 1730563200.1, "action": {"type": "exec", "command": "ls"}, "observation": {"exit_code": 0, "stdout": "...", "stderr": "", "duration": 0.04, "timed_out": false, "truncated": false}}
```

Action types are `exec`, `write_file`, `read_file`, `put_archive` and `snapshot`. File writes keep their full content, so a run can be replayed from the log alone.

## A note on isolation

Containers aren't VMs. These defaults stop the everyday stuff an agent might do by accident, but if you're running truly hostile code, put this on a throwaway VM or use a runtime like gVisor (`--runtime=runsc`) underneath Docker.

## Part of Gauntlet

This is the first piece of **Gauntlet**, an open source arena where AI agents get tasks, run them in sandboxes, get their runs replayed and get checked for cheating. Next up is Repo2Env, which turns any GitHub repo into a task this sandbox can run.

## Develop

```bash
uv sync --extra server
uv run pytest           # docker tests skip themselves if Docker isn't running
uv run ruff check .
```

MIT licensed.
