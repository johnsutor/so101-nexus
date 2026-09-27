"""Source distributions keep source files and exclude generated documentation."""

import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_excludes_generated_docs(tmp_path):
    for name in ("pyproject.toml", "README.md", ".gitignore"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    source_files = (
        "src/so101_nexus/__init__.py",
        "docs/content/docs/index.mdx",
        "docs/lib/source.ts",
        "patches/mujoco-warp-3.13.0/box_box_cpu_sat_port_double.patch",
    )
    generated_files = (
        "docs/.next/cache/build.bin",
        "docs/out/index.html",
        "docs/.source/index.ts",
        "docs/node_modules/dependency/index.js",
    )
    for name in (*source_files, *generated_files):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    shutil.copyfile(ROOT / "docs/.gitignore", tmp_path / "docs/.gitignore")
    subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", "dist"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (archive_path,) = (tmp_path / "dist").glob("*.tar.gz")
    with tarfile.open(archive_path) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames()}
    assert set(source_files) <= names
    assert not set(generated_files) & names
