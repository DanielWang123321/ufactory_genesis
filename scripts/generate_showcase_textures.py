#!/usr/bin/env python3
"""Generate tileable textures for the xArm6 packaging showcase scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "scenes" / "packaging_showcase" / "textures"
MESH_DIR = ROOT / "assets" / "scenes" / "packaging_showcase" / "meshes"
SIZE = 512
UV_BOX_OBJ = MESH_DIR / "uv_box.obj"


def _export_uv_box_obj(path: Path) -> None:
  """Unit cube (1 m) with per-face UVs for ImageTexture on gs.morphs.Mesh."""
  import trimesh

  h = 0.5
  verts: list[tuple[float, float, float]] = []
  faces: list[list[int]] = []
  uvs: list[tuple[float, float]] = []
  idx = 0

  def add_face(corners, uv_corners):
    nonlocal idx
    base = idx
    for c in corners:
      verts.append(c)
    for u in uv_corners:
      uvs.append(u)
    faces.append([base, base + 1, base + 2])
    faces.append([base, base + 2, base + 3])
    idx += 4

  # +X, -X, +Y, -Y, +Z, -Z
  add_face(
    [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
  )
  add_face(
    [(-h, -h, h), (-h, h, h), (-h, h, -h), (-h, -h, -h)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
  )
  add_face(
    [(-h, h, -h), (-h, h, h), (h, h, h), (h, h, -h)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
  )
  add_face(
    [(-h, -h, h), (-h, -h, -h), (h, -h, -h), (h, -h, h)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
  )
  add_face(
    [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],
    [(0, 0), (1, 0), (1, 1), (0, 1)],
  )
  add_face(
    [(-h, -h, -h), (-h, h, -h), (h, h, -h), (h, -h, -h)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
  )

  mesh = trimesh.Trimesh(
    vertices=np.asarray(verts, dtype=np.float64),
    faces=np.asarray(faces, dtype=np.int64),
    process=False,
  )
  mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64))
  path.parent.mkdir(parents=True, exist_ok=True)
  # Plain OBJ (no .mtl): avoids Genesis overwriting surface with missing map_Kd → black mesh.
  with path.open("w", encoding="utf-8") as f:
    for x, y, z in mesh.vertices:
      f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
    for u, v in mesh.visual.uv:
      f.write(f"vt {u:.8f} {v:.8f}\n")
    for face in mesh.faces:
      a, b, c = face + 1
      f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
  try:
    from PIL import Image
  except ImportError as exc:
    raise SystemExit("Pillow required: pip install Pillow") from exc
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)).save(path, quality=92)


def _wood_grain(seed: int, base: tuple[float, float, float], contrast: float) -> np.ndarray:
  rng = np.random.default_rng(seed)
  y, x = np.mgrid[0:SIZE, 0:SIZE]
  phase = rng.uniform(0, 2 * np.pi)
  rings = np.sin((x * 0.045 + y * 0.012) + phase)
  grain = np.sin(y * 0.35 + rng.normal(0, 0.4, SIZE)[y] * 0.15)
  noise = rng.normal(0, 0.04, (SIZE, SIZE))
  v = 0.55 + contrast * (0.25 * rings + 0.20 * grain + noise)
  v = np.clip(v, 0, 1)
  rgb = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
  for c, b in enumerate(base):
    rgb[:, :, c] = b * (0.65 + 0.35 * v)
  return rgb


def _cardboard(seed: int) -> np.ndarray:
  """黄褐色瓦楞纸箱纹理（常规快递箱色调）。"""
  rng = np.random.default_rng(seed)
  y, x = np.mgrid[0:SIZE, 0:SIZE]
  flute = 0.5 + 0.5 * np.sin(y * 0.55)
  fiber = rng.normal(0, 0.05, (SIZE, SIZE))
  stain = 0.03 * np.sin(x * 0.08) * np.sin(y * 0.05)
  v = np.clip(0.60 * flute + fiber + stain, 0, 1)
  base = np.array([0.82, 0.66, 0.40], dtype=np.float32)  # tan / 黄褐
  rgb = base * (0.78 + 0.22 * v)[..., None]
  bands = 0.94 + 0.06 * np.sin(x * 0.22)
  rgb *= bands[..., None]
  return np.clip(rgb, 0, 1)


def main() -> None:
  _save_rgb(OUT_DIR / "wood_table.jpg", _wood_grain(7, (0.55, 0.36, 0.20), 1.0))
  _save_rgb(OUT_DIR / "wood_block.jpg", _wood_grain(19, (0.62, 0.42, 0.22), 0.85))
  _save_rgb(OUT_DIR / "cardboard.jpg", _cardboard(42))
  _export_uv_box_obj(UV_BOX_OBJ)
  print(f"Wrote textures to {OUT_DIR}")
  print(f"Wrote mesh to {UV_BOX_OBJ}")


if __name__ == "__main__":
  main()
