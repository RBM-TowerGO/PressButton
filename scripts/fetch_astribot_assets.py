#!/usr/bin/env python3
"""Download the pinned Astribot asset manifest without installing dependencies."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/astribot_sources.lock.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    destination = ROOT / manifest["destination"]

    def fetch(entry):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe asset path: {relative}")
        target = destination / relative
        if target.exists():
            if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                raise ValueError(f"Conflicting asset (preserved): {target}")
        else:
            if args.check_only:
                raise FileNotFoundError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    temporary = Path(out.name)
                    with urlopen(entry["url"], timeout=120) as response:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                    raise ValueError(f"Downloaded asset checksum mismatch: {relative}")
                # A concurrent writer must never be overwritten.
                target.hardlink_to(temporary)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        print(f"Verified {relative}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(fetch, manifest["files"]))
    print(f"PASS: {len(manifest['files'])} Astribot assets verified", flush=True)


if __name__ == "__main__":
    main()
