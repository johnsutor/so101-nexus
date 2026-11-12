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
    import tyro

    from so101_nexus.calibration.app import build_app

    options = tyro.cli(LaunchOptions)
    build_app().queue().launch(
        server_name="127.0.0.1", server_port=options.port, inbrowser=options.open_browser
    )


if __name__ == "__main__":
    main()
