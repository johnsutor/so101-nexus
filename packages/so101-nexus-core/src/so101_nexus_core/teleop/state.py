"""Typed dataclasses replacing the previous dict-threaded teleop session state.

``InitConfig`` is the validated, normalized snapshot of UI inputs for one init
attempt. ``InitState`` tracks the background init worker's progress.
``TeleopSession`` holds the live runtime objects produced after a successful init.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from so101_nexus_core.teleop.dataset import FieldSelection
    from so101_nexus_core.teleop.leader import LeaderProtocol
    from so101_nexus_core.teleop.recorder import RecordingState


@dataclass(frozen=True)
class InitConfig:
    """Normalized, validated snapshot of UI inputs for one init attempt."""

    env_id: str
    robot_type: str
    leader_id: str
    fps: int
    wrist_wh: tuple[int, int]
    overhead_wh: tuple[int, int]
    repo_id: str
    num_episodes: int
    action_space: str
    max_steps: int
    countdown: int
    wrist_roll_offset_deg: float
    field_selection: FieldSelection


@dataclass
class InitState:
    """Mutable state for the background init worker."""

    running: bool = False
    done: bool = False
    processed: bool = False
    error: str | None = None
    warning: str | None = None
    log_lines: list[str] = field(default_factory=list)
    last_config: InitConfig | None = None

    @property
    def log_text(self) -> str:
        """Newline-joined view of log_lines for the UI textbox."""
        return "\n".join(self.log_lines)

    def append_log(self, message: str) -> None:
        """Append one log line."""
        self.log_lines.append(message)

    def reset_for_new_attempt(self, *, warning: str | None, last_config: InitConfig) -> None:
        """Clear state for a fresh init attempt while remembering last_config."""
        self.running = True
        self.done = False
        self.processed = False
        self.error = None
        self.warning = warning
        self.log_lines = []
        self.last_config = last_config


@dataclass
class TeleopSession:
    """Live runtime objects populated by the init worker, consumed by the recording loop."""

    leader: LeaderProtocol | None = None
    dataset: Any = None
    state: RecordingState | None = None
    joint_names: tuple[str, ...] = ()
    fps: int = 0
    action_space: str = ""
    max_steps: int = 0
    countdown: int = 0
    wrist_wh: tuple[int, int] = (0, 0)
    overhead_wh: tuple[int, int] = (0, 0)
    env_id: str = ""
    robot_type: str = ""
    wrist_roll_offset_deg: float = 0.0
    field_selection: FieldSelection | None = None
    action_pipeline: Any = None
