"""Shared label files must stay readable while another scene is rendering."""
import os
from pathlib import Path

import pytest

Usd = pytest.importorskip("pxr.Usd")
from PIL import Image
from pressb import scene


def test_identical_label_does_not_rewrite_live_texture(tmp_path, monkeypatch):
    monkeypatch.setattr(scene, "ROOT", tmp_path)
    for _ in range(2):
        stage = Usd.Stage.CreateInMemory()
        scene._label(stage, "/Label", (0,0,0), .1, .02, "Test", "shared")
        path = tmp_path/"assets/generated/labels/shared.png"
        if _ == 0:
            first = path.stat().st_mtime_ns
    assert path.stat().st_mtime_ns == first


def test_changed_label_replaces_complete_png_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(scene, "ROOT", tmp_path)
    scene._label(Usd.Stage.CreateInMemory(), "/Label", (0,0,0), .1, .02, "Old", "shared")
    path = tmp_path/"assets/generated/labels/shared.png"
    old = path.read_bytes()
    replace = os.replace
    calls = []

    def checked_replace(source, destination):
        assert Path(destination).read_bytes() == old
        with Image.open(source) as image:
            image.verify()
        calls.append(1)
        replace(source,destination)

    monkeypatch.setattr(os,"replace",checked_replace)
    scene._label(Usd.Stage.CreateInMemory(), "/Label", (0,0,0), .1, .02, "New", "shared")
    assert calls == [1]
    assert path.read_bytes() != old
    assert list(path.parent.iterdir()) == [path]
