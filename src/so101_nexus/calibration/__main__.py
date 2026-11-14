"""Launch the local guided calibration app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LaunchOptions:
    """Local server settings; calibration takes place in the browser."""

    port: int = 7861
    open_browser: bool = True


def main() -> None:
    """Launch Gradio on loopback without connecting to hardware."""
    import os

    import tyro

    from so101_nexus.calibration.app import build_app
    from so101_nexus.calibration.storage import calibration_dir, profiles_dir

    options = tyro.cli(LaunchOptions)
    build_app().queue().launch(
        server_name="127.0.0.1",
        server_port=options.port,
        inbrowser=options.open_browser,
        allowed_paths=[
            str(profiles_dir()),
            str(calibration_dir()),
            *[
                path.strip()
                for path in os.environ.get("GRADIO_ALLOWED_PATHS", "").split(",")
                if path.strip()
            ],
        ],
    )


if __name__ == "__main__":
    main()
