"""Guide physical reference capture before optional dynamics identification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from so101_nexus.calibration.session import CalibrationSession


@dataclass
class CalibrationGuide:
    """Advance a shared hardware session through explicit profile creation steps.

    Parameters
    ----------
    session : CalibrationSession
        Hardware owner used for read-only monitoring and explicit reference capture.
    """

    session: CalibrationSession
    stage: int = 0
    joint_index: int = 0
    reference_deg: list[float] = field(default_factory=lambda: [0, 0, 0, 0, 0, 45])
    view: str = "front"
    focused: bool = False

    def _require_stage(self, stage: int) -> None:
        if self.stage != stage:
            raise ValueError(f"Complete the current guide stage first; requested stage {stage}.")

    def _select_joint(self, index: int) -> None:
        self.joint_index = index
        self.view = "front"
        self.focused = index >= 3

    def _show_save(self) -> None:
        self.stage = 3
        self.view = "front"
        self.focused = False

    def connect(self, port: str, calibration_dir: str) -> None:
        """Open read-only monitoring and begin the first body-joint reference."""
        with self.session.lock:
            self.stage = 0
            self.session.connect(port, calibration_dir)
            self.reference_deg = [0, 0, 0, 0, 0, 45]
            self._select_joint(0)
            self.stage = 1

    def capture_next(self, model_deg: float) -> None:
        """Capture the explicitly confirmed physical reference and advance one joint."""
        with self.session.lock:
            self._require_stage(1)
            if (
                isinstance(model_deg, bool)
                or not isinstance(model_deg, (int, float))
                or not math.isfinite(model_deg)
                or model_deg != self.reference_deg[self.joint_index]
            ):
                raise ValueError(
                    "Reference view is updating. Wait for the ghost to update and check "
                    "the physical joint before capturing."
                )
            self.session.capture_joint(self.joint_index, model_deg, confirmed=True)
            self.reference_deg[self.joint_index] = float(model_deg)
            if self.joint_index == 4:
                self.stage = 2
            else:
                self._select_joint(self.joint_index + 1)

    def previous_joint(self) -> None:
        """Return to the previous body joint without changing its calibration."""
        with self.session.lock:
            self.session._idle()
            if self.stage == 0:
                raise ValueError("Connect before returning to a body-joint stage.")
            index = max(0, self.joint_index - 1) if self.stage == 1 else 4
            self.stage = 1
            self._select_joint(index)

    def set_reference(self, model_deg: float) -> None:
        """Change a known model reference and revoke its previous confirmation."""
        with self.session.lock:
            self._require_stage(1)
            self.session._idle()
            if (
                isinstance(model_deg, bool)
                or not isinstance(model_deg, (int, float))
                or not math.isfinite(model_deg)
            ):
                raise ValueError("The reference angle must be finite model degrees.")
            self.reference_deg[self.joint_index] = float(model_deg)
            self.session.confirmed.discard(self.joint_index)
            self.session.verified = False

    def preview_rotation(self, delta_deg: float) -> None:
        """Adjust the provisional measured overlay while preserving the known target."""
        with self.session.lock:
            self._require_stage(1)
            self.session.preview_joint_offset(self.joint_index, delta_deg)

    def confirm_gripper(self, lower_deg: float, upper_deg: float, *, confirmed: bool) -> None:
        """Advance after separate confirmation of the physical gripper mapping."""
        with self.session.lock:
            self._require_stage(2)
            self.session.set_gripper(lower_deg, upper_deg, confirmed=confirmed)
            self._show_save()

    def load_profile(self, path: str | Path) -> None:
        """Restore a trusted profile without treating the current pose as a reference."""
        with self.session.lock:
            self.session.load_profile(path)
            self._show_save()

    def save_profile(self, path: str | Path, period_s: float) -> Path:
        """Save the confirmed profile using the operator's command period."""
        with self.session.lock:
            self._require_stage(3)
            self.session._aligned()
            self.session.set_period(period_s)
            return self.session.save_profile(path)
