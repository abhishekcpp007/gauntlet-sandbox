"""Command line: `gauntlet-sandbox run|repl|serve|show|gc`."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from . import __version__
from .models import SandboxConfig
from .sandbox import Sandbox, SandboxError
from .trajectory import Trajectory


def _config(args: argparse.Namespace) -> SandboxConfig:
    return SandboxConfig(
        image=args.image,
        memory=args.memory,
        cpus=args.cpus,
        network=args.network,
        default_timeout=args.timeout,
    )


def _add_box_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--image", default="python:3.12-slim", help="Docker image (default: %(default)s)")
    p.add_argument("--memory", default="512m", help="memory cap (default: %(default)s)")
    p.add_argument("--cpus", type=float, default=1.0, help="CPU cap (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=30, help="seconds per command (default: 30)")
    p.add_argument("--network", action="store_true", help="allow network access (off by default)")
    p.add_argument("--record", metavar="FILE", help="save the trajectory to this JSONL file")


def cmd_run(args: argparse.Namespace) -> int:
    if not args.command:
        print("give me a command, like: gauntlet-sandbox run -- python -c 'print(42)'", file=sys.stderr)
        return 2
    command = args.command[0] if len(args.command) == 1 else shlex.join(args.command)
    with Sandbox(_config(args), trajectory_path=args.record) as sb:
        res = sb.exec(command)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.timed_out:
        print(f"\n[gauntlet] killed after {args.timeout:g}s timeout", file=sys.stderr)
        return 124
    return res.exit_code


def cmd_repl(args: argparse.Namespace) -> int:
    with Sandbox(_config(args), trajectory_path=args.record) as sb:
        print(f"gauntlet sandbox {sb.id} on {args.image}. Ctrl-D to leave.")
        while True:
            try:
                line = input("gauntlet$ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            res = sb.exec(line)
            sys.stdout.write(res.stdout)
            sys.stderr.write(res.stderr)
            if res.timed_out:
                print("[timed out]", file=sys.stderr)
            elif res.exit_code:
                print(f"[exit {res.exit_code}]", file=sys.stderr)
    if args.record:
        print(f"trajectory saved to {args.record}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .server import ServerSettings, create_app
    except ImportError:
        print("server extras missing: pip install 'gauntlet-sandbox[server]'", file=sys.stderr)
        return 1
    settings = ServerSettings.from_env()
    if args.trajectory_dir:
        settings.trajectory_dir = Path(args.trajectory_dir)
    if args.host not in ("127.0.0.1", "localhost") and not settings.api_key:
        print("warning: binding beyond localhost without GAUNTLET_API_KEY set", file=sys.stderr)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


def _short(text: str, n: int = 6) -> str:
    lines = text.rstrip("\n").splitlines()
    shown = lines[:n] + ([f"... ({len(lines) - n} more lines)"] if len(lines) > n else [])
    return "\n".join("     " + ln for ln in shown)


def cmd_show(args: argparse.Namespace) -> int:
    traj = Trajectory.load(args.file)
    h = traj.header
    print(f"trajectory {h.sandbox_id} on {h.image}, {len(traj)} steps")
    if h.metadata:
        print(f"metadata {h.metadata}")
    for s in traj:
        a, o = s.action, s.observation
        kind = a.get("type")
        if kind == "exec":
            flag = " TIMEOUT" if o.get("timed_out") else ""
            print(f"\n#{s.index} $ {a['command']}   -> exit {o['exit_code']} in {o['duration']:.2f}s{flag}")
            if o.get("stdout"):
                print(_short(o["stdout"]))
            if o.get("stderr"):
                print(_short(o["stderr"]))
        elif kind == "write_file":
            print(f"\n#{s.index} write {a['path']} ({o.get('size', '?')} bytes)")
        elif kind == "read_file":
            status = f"{o.get('size')} bytes" if o.get("ok") else o.get("error")
            print(f"\n#{s.index} read {a['path']} ({status})")
        elif kind == "snapshot":
            print(f"\n#{s.index} snapshot -> {o.get('image')}")
        else:
            print(f"\n#{s.index} {kind}")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    n = Sandbox.cleanup_all(snapshots=args.snapshots)
    print(f"removed {n} item(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gauntlet-sandbox", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run one command in a fresh sandbox")
    _add_box_args(p)
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("repl", help="interactive shell inside a sandbox")
    _add_box_args(p)
    p.set_defaults(fn=cmd_repl)

    p = sub.add_parser("serve", help="start the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--trajectory-dir")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("show", help="pretty-print a trajectory file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("gc", help="remove leftover sandbox containers")
    p.add_argument("--snapshots", action="store_true", help="also delete snapshot images")
    p.set_defaults(fn=cmd_gc)

    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[0] == "--":
        args.command = args.command[1:]
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        return args.fn(args)
    except SandboxError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
