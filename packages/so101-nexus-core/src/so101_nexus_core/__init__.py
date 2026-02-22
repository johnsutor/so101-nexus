from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SO_ARM100_DIR = ASSETS_DIR / "SO-ARM100"
SO101_DIR = ASSETS_DIR / "SO101"


def get_so101_simulation_dir() -> Path:
    return SO101_DIR


def get_so100_simulation_dir() -> Path:
    return SO_ARM100_DIR / "Simulation" / "SO100"
