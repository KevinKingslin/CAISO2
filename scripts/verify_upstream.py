"""Verify that the imported benchmark files still match the pinned release."""

import hashlib
import json
from pathlib import Path


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "UPSTREAM.json").read_text())
    checked = 0
    failures = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"Missing: {record['path']}")
        elif record["expected_sha256"]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["expected_sha256"]:
                failures.append(f"Changed: {record['path']}")
            checked += 1
    if (root / "optimas").exists():
        failures.append("Unexpected /optimas directory; package must be /caiso")
    if failures:
        raise ValueError("\n".join(failures))
    return checked, manifest["commit"]


if __name__ == "__main__":
    count, commit = verify()
    print(f"Verified {count} unchanged/namespace-only files against OPTIMAS {commit}")
