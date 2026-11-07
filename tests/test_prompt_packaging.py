"""Release archives and wheels must contain every catalog resource."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest

from caiso.prompts import CATALOGS

ROOT = Path(__file__).resolve().parents[1]


def test_sdist_wheel_and_zip_import_include_all_prompt_files(tmp_path):
    pytest.importorskip("setuptools.build_meta")
    source = tmp_path / "source"
    source.mkdir()
    for name in ("caiso", "scripts"):
        shutil.copytree(ROOT / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("pyproject.toml", "MANIFEST.in", "LICENSE", "README.md"):
        shutil.copy2(ROOT / name, source / name)
    expected = {
        "caiso/prompts/" + name + ".txt"
        for catalog in CATALOGS.values()
        for reference in catalog.values()
        for name in (reference if isinstance(reference, tuple) else [reference])
    }
    output = tmp_path / "dist"
    output.mkdir()
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")

    def run(code, directory):
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=directory, env=environment,
            text=True, capture_output=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    run(f"from setuptools.build_meta import build_sdist; build_sdist({str(output)!r})", source)
    with tarfile.open(next(output.glob("*.tar.gz"))) as archive:
        prefix = archive.getmembers()[0].name.split("/")[0]
        assert expected <= {name.removeprefix(prefix + "/") for name in archive.getnames()}
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = tmp_path / "unpacked" / prefix
    run(f"from setuptools.build_meta import build_wheel; build_wheel({str(output)!r})", unpacked)
    wheel = next(output.glob("*.whl"))
    with ZipFile(wheel) as archive:
        assert expected == {name for name in archive.namelist() if name.startswith("caiso/prompts/") and name.endswith(".txt")}
        for name in expected:
            assert archive.read(name) == (ROOT / name).read_bytes()
    # Import directly from the wheel in a fresh process, outside the checkout.
    result = run(
        f"import sys; sys.path.insert(0, {str(wheel)!r}); "
        "import caiso.prompts as p; "
        f"assert p.__file__.startswith({str(wheel)!r}), p.__file__; "
        "[p.catalog_entries(name) for name in p.CATALOGS]; print(p.validate_resources())",
        tmp_path,
    )
    assert result.strip().endswith(str(len(expected)))
