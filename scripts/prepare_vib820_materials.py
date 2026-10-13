#!/usr/bin/env python3
"""Prepare a clean, fine hairline finish from the CC0 ambientCG Metal010 maps.

Source maps remain unchanged. This is texture processing, not measured material
calibration: remove broad grime, retain microstructure, and reduce normal strength.
Run with the project Python after downloading assets/materials/vib820/source.
"""
from pathlib import Path
import json
import hashlib
import io
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "assets/materials/vib820"


def main():
    src = DIRECTORY / "source"
    def read(channel):
        return Image.open(src / f"Metal010_2K-JPG_{channel}.jpg")
    def fine(im):
        return (np.asarray(im, dtype=np.float32) -
                np.asarray(im.filter(ImageFilter.GaussianBlur(18)), dtype=np.float32)) / 255.
    def save(array, name):
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG")
        data = stream.getvalue()
        path = DIRECTORY / name
        if path.exists() and path.read_bytes() == data:
            return
        # Never expose a truncated PNG to the renderer's asset file watcher.
        temporary = path.with_suffix(".png.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    # Narrow albedo range prevents a clean elevator panel looking like dirty sheet metal.
    color = .76 + fine(read("Color")) * .12
    save(np.uint8(np.clip(color, 0, 1)*255), "steel_clean_color.png")
    rough = .25 + fine(read("Roughness").convert("L")) * .20
    save(np.uint8(np.clip(rough, .18, .34)*255), "steel_clean_roughness.png")
    normal = np.asarray(read("NormalGL"), dtype=np.float32)/127.5 - 1
    normal[:, :, :2] *= .12
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    save(np.uint8(np.clip(normal*.5+.5, 0, 1)*255), "steel_clean_normal_gl.png")
    sources = {
        "license": "CC0-1.0",
        "metal": {"asset": "Metal010", "creator": "ambientCG / Lennart Demes",
                  "url": "https://ambientcg.com/a/Metal010",
                  "download": "https://ambientcg.com/get?file=Metal010_2K-JPG.zip",
                  "license_url": "https://docs.ambientcg.com/license/",
                  "usage": "Derived clean albedo, fine roughness and OpenGL normal; no displacement."},
        "environment": {"asset": "Large Corridor", "creator": "Sergej Majboroda / Poly Haven",
                        "url": "https://polyhaven.com/a/large_corridor",
                        "download": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/large_corridor_2k.hdr",
                        "license_url": "https://polyhaven.com/license",
                        "usage": "Approximate indoor reflection/illumination environment, not a scan of the real lobby."},
        "shaders": {"provider": "NVIDIA Isaac Sim 5.0 installed MDL library",
                    "modules": ["OmniSurface.mdl", "OmniPBR.mdl"],
                    "url": "https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniSurface.html",
                    "usage": "Built-in shader references only; NVIDIA MDL source files are not redistributed."},
        "processing": "scripts/prepare_vib820_materials.py; authoring estimates, not measured BRDF",
        "sha256": {str(p.relative_to(DIRECTORY)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(DIRECTORY.rglob("*")) if p.is_file() and p.suffix in (".jpg", ".png", ".hdr", ".zip")}}
    (DIRECTORY / "manifest.json").write_text(json.dumps(sources, indent=2)+"\n")
    print("Prepared", DIRECTORY)


if __name__ == "__main__":
    main()
