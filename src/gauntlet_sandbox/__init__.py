"""gauntlet-sandbox: safe Docker sandboxes for AI agents, with every step recorded."""

from .models import ExecResult, SandboxConfig
from .sandbox import Sandbox, SandboxError
from .trajectory import Step, Trajectory, TrajectoryHeader

__all__ = [
    "ExecResult",
    "Sandbox",
    "SandboxConfig",
    "SandboxError",
    "Step",
    "Trajectory",
    "TrajectoryHeader",
]
__version__ = "0.1.0"
