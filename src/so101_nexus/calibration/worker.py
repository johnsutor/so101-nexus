"""Isolate timed motor collection from the browser process and its interpreter lock."""

from __future__ import annotations

import multiprocessing
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.calibration.session import CalibrationSession

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from pathlib import Path

    from so101_nexus.excitation import ExcitationConfig
    from so101_nexus.profiles import RobotProfile


class _ChildLease:
    def release(self) -> None:
        pass


def _snapshot(session: CalibrationSession, *, final: bool = False) -> dict[str, Any]:
    measured = session.measured
    return {
        "final": final,
        "phase": session.phase,
        "status": session.status,
        "measured": None if measured is None else measured.tolist(),
        "progress": session.progress,
        "collection_progress": session.collection_progress,
        "collection_complete": session.collection_complete,
        "collection_error": session.collection_error,
        "artifacts": session.artifacts,
    }


def _apply_control(session: CalibrationSession, control: tuple[float, bool, bool]) -> None:
    last_seen, cancelled, motion = control
    session.last_seen = last_seen
    if cancelled:
        session.cancel.set()
    if motion:
        session.motion.set()


def _initial_control(connection: Connection, session: CalibrationSession) -> None:
    try:
        if connection.poll(5):
            _apply_control(session, connection.recv())
            while connection.poll():
                _apply_control(session, connection.recv())
        else:
            session.cancel.set()
    except (EOFError, OSError):
        session.cancel.set()


def _receive_controls(connection: Connection, session: CalibrationSession) -> None:
    try:
        while True:
            _apply_control(session, connection.recv())
    except (EOFError, OSError):
        session.cancel.set()


def _publish_state(
    connection: Connection, session: CalibrationSession, finished: threading.Event
) -> None:
    try:
        while not finished.wait(0.2):
            connection.send(_snapshot(session))
        connection.send(_snapshot(session, final=True))
    except (BrokenPipeError, EOFError, OSError):
        session.cancel.set()


def _run_collection(
    connection: Connection,
    profile: RobotProfile,
    config: ExcitationConfig,
    destination: Path,
    port: str,
    profile_path: Path | None,
    session_type: type[CalibrationSession],
) -> None:
    session = session_type()
    session.port = port
    session.profile = profile
    session.profile_path = profile_path
    session.config = config
    session.phase = "arming"
    session.torque_possible = True
    session.artifacts = [str(profile_path)] if profile_path else []
    # Stop may arrive during interpreter startup, before any child thread can receive it.
    _initial_control(connection, session)
    finished = threading.Event()
    threading.Thread(target=_receive_controls, args=(connection, session), daemon=True).start()
    publisher = threading.Thread(
        target=_publish_state, args=(connection, session, finished), daemon=True
    )
    publisher.start()
    try:
        if session.should_stop():
            session.phase = "stopped"
            session.status = (
                "Preparation cancelled before motor connection. Torque was not changed."
            )
            session.collection_error = session.status
        else:
            session._collect(profile, config, destination, _ChildLease())
    finally:
        finished.set()
        publisher.join(timeout=2)
        connection.close()


def _stop_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join(timeout=1)
    if process.is_alive():
        raise RuntimeError("Motor process is still stopping. Port remains owned; retry disconnect.")


class _ProcessCleanup:
    def __init__(self, process: Any) -> None:
        self.process = process

    def disconnect(self, *, disable_torque: bool) -> None:
        _stop_process(self.process)
        self.process.close()


class IsolatedCalibrationSession(CalibrationSession):
    """Keep browser work outside the process that owns timed motor communication."""

    _collector_type: type[CalibrationSession] = CalibrationSession

    def _apply_worker_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self.phase = state["phase"]
            self.status = state["status"]
            measured = state["measured"]
            self.measured = None if measured is None else np.asarray(measured)
            self.progress = state["progress"]
            self.collection_progress = state["collection_progress"]
            if state["final"]:
                self.collection_complete = state["collection_complete"]
                self.collection_error = state["collection_error"]
                self.artifacts = state["artifacts"]
                self.verified = False
                if self.collection_error:
                    self.failure_sequence += 1

    def _collect(
        self, profile: RobotProfile, config: ExcitationConfig, destination: Path, lease: Any
    ) -> None:
        context = multiprocessing.get_context("spawn")
        connection, child_connection = context.Pipe()
        process = context.Process(
            target=_run_collection,
            args=(
                child_connection,
                profile,
                config,
                destination,
                self.port,
                self.profile_path,
                self._collector_type,
            ),
            name="calibration-motors",
        )
        final = None
        try:
            process.start()
            child_connection.close()
            final = self._supervise(connection, process)
            process.join(timeout=2)
            if process.exitcode != 0 or final is None:
                raise RuntimeError(f"Motor process exited unexpectedly (code {process.exitcode})")
        except Exception as error:
            message = str(error)
            self.cancel.set()
            if process.pid is not None:
                try:
                    _stop_process(process)
                except RuntimeError as cleanup_error:
                    self._pending_bus, self._pending_lease = _ProcessCleanup(process), lease
                    message = str(cleanup_error)
            artifacts, snapshot_error = self._recording_artifacts(profile, destination)
            final = _snapshot(self, final=True)
            final.update(
                phase="stopped",
                measured=None,
                collection_complete=False,
                collection_error=(
                    f"Collection stopped: {message}.{snapshot_error} "
                    "Torque may remain enabled at the last target. Support before release."
                ),
                artifacts=([str(self.profile_path)] if self.profile_path else []) + artifacts,
            )
            final["status"] = final["collection_error"]
        finally:
            connection.close()
            child_connection.close()
            if process.pid is None or not process.is_alive():
                lease.release()
                if process.pid is not None:
                    process.close()
        self._apply_worker_state(final)

    def _supervise(self, connection: Connection, process: Any) -> dict[str, Any] | None:
        final = None
        next_control = 0.0
        stop_deadline = None
        while process.is_alive() or connection.poll():
            stopping = self.should_stop()
            if stopping and stop_deadline is None:
                stop_deadline = time.monotonic() + 2
            if stop_deadline is not None and time.monotonic() > stop_deadline:
                _stop_process(process)
                break
            if time.monotonic() >= next_control and process.is_alive():
                try:
                    connection.send((self.last_seen, stopping, self.motion.is_set()))
                except (BrokenPipeError, OSError):
                    break
                next_control = time.monotonic() + 0.05
            if connection.poll(0.05):
                try:
                    state = connection.recv()
                except (EOFError, OSError):
                    break
                if state["final"]:
                    final = state
                    stop_deadline = time.monotonic() + 2
                else:
                    self._apply_worker_state(state)
        return final
