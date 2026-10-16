"""Content-addressed immutable shared assets for resident workers."""
import os
from pathlib import Path
import uuid
from .contract import sha256


def share_asset(path, cache):
    path, cache = Path(path), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    digest = sha256(path)
    with path.open('rb') as file:
        os.fsync(file.fileno())
    canonical = cache / (digest + path.suffix)
    try:
        # The producer has closed the complete file before publishing this link.
        os.link(path, canonical)
    except FileExistsError:
        if sha256(canonical) != digest:
            raise ValueError(f'Corrupted shared asset: {canonical}')
    temporary = path.with_name('.' + path.name + '.link-' + uuid.uuid4().hex)
    try:
        os.link(canonical, temporary)
        os.replace(temporary, path)
        for directory in (cache, path.parent):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(sha256=digest, canonical=str(canonical.resolve()), bytes=path.stat().st_size)


def validate_assets(references):
    for name, reference in references.items():
        if sha256(reference['canonical']) != reference['sha256']:
            raise ValueError(f'Corrupt referenced asset: {name}')
