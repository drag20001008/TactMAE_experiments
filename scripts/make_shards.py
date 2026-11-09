"""
make_shards.py
Converts ShapeNetCoreV2Rendered + tactile.npz into OctMAE webdataset tar shards.

Each shard contains samples with these keys per entry:
  {key}.rgb.jpg          RGB image (JPEG)
  {key}.depth.png        uint16 depth, depth_scale=0.1 → mm
  {key}.camera.json      cam_K, cam_R_w2c, cam_t_w2c (mm), depth_scale
  {key}.gt.json          object pose (unused in training, kept for compat)
  {key}.gt_info.json     visib_fract, bbox
  {key}.mask_visib.json  COCO RLE mask
  {key}.spc.npz          GT surface: (N,7) float16 [x,y,z,sdf,nx,ny,nz] in cam/mm
  {key}.tactile.npz      tactile points (optional, from precompute_tactile.py)

Usage:
  python scripts/make_shards.py \\
      --rendered_root /nas1/.../ShapeNetCoreV2Rendered \\
      --shapenet_root /nas1/.../ShapeNetCore \\
      --output_dir    /nas1/.../OctMAE_ShapeNet_shards \\
      --categories 02876657 02880940 02942699 03467517 03593526 03797390
"""

import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm


# ── constants ─────────────────────────────────────────────────────────────────
DEPTH_SCALE   = 0.1       # uint16 × 0.1 = mm  (OctMAE convention)
N_SPC_SURF    = 40000     # on-surface samples per object (SDF = 0)
N_SPC_NEAR    = 10000     # near-surface samples per object (|SDF| < 1mm)
SHARD_SIZE    = 500       # samples per tar shard
VAL_RATIO     = 0.1       # fraction of objects held out for validation
TEST_RATIO    = 0.1       # fraction of objects held out for test


# ── depth helpers ─────────────────────────────────────────────────────────────

def load_depth_meters(path: Path) -> np.ndarray:
    """Load rendered uint16 depth (max 10 m) → float32 meters."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    return raw / 65535.0 * 10.0


def encode_depth_png(depth_m: np.ndarray) -> bytes:
    """Encode depth (meters) → uint16 PNG with depth_scale = 0.1 (0.1 mm/unit)."""
    d16 = (depth_m * 10000.0).clip(0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(d16).save(buf, format='PNG')
    return buf.getvalue()


# ── per-file encoders ─────────────────────────────────────────────────────────

def encode_rgb(path: Path) -> bytes:
    buf = io.BytesIO()
    Image.open(path).convert('RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def encode_camera_json(intr_path: Path, c2w_path: Path) -> bytes:
    K   = torch.load(str(intr_path), weights_only=True).numpy()
    c2w = torch.load(str(c2w_path),  weights_only=True).numpy().astype(np.float64)
    w2c = np.linalg.inv(c2w)
    d = {
        "cam_K":     K.flatten().tolist(),
        "cam_R_w2c": w2c[:3, :3].flatten().tolist(),
        "cam_t_w2c": (w2c[:3, 3] * 1000.0).tolist(),   # m → mm
        "depth_scale": DEPTH_SCALE,
    }
    return json.dumps(d).encode()


def encode_mask_jsons(mask_path: Path):
    """Returns (mask_visib_bytes, gt_info_bytes)."""
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    binary = (raw > 0).astype(np.uint8)

    rle = coco_mask.encode(np.asfortranarray(binary))
    rle_dict = {"size": list(rle["size"]), "counts": rle["counts"].decode()}
    mask_visib = {"0": rle_dict}

    ys, xs = np.where(binary)
    if len(ys) > 0:
        x1, y1 = int(xs.min()), int(ys.min())
        bbox = [x1, y1, int(xs.max() - x1), int(ys.max() - y1)]
        px   = int(binary.sum())
        frac = 1.0
    else:
        bbox = [-1, -1, -1, -1]
        px   = 0
        frac = 0.0

    gt_info = [{
        "bbox_obj": bbox, "bbox_visib": bbox,
        "px_count_all": px, "px_count_valid": px, "px_count_visib": px,
        "visib_fract": frac,
    }]
    return json.dumps(mask_visib).encode(), json.dumps(gt_info).encode(), frac


def encode_gt_json(c2w_path: Path) -> bytes:
    c2w = torch.load(str(c2w_path), weights_only=True).numpy().astype(np.float64)
    w2c = np.linalg.inv(c2w)
    gt  = [{"obj_id": 0,
            "cam_R_m2c": w2c[:3, :3].flatten().tolist(),
            "cam_t_m2c": (w2c[:3, 3] * 1000.0).tolist()}]
    return json.dumps(gt).encode()


# ── SPC generation ─────────────────────────────────────────────────────────────

def build_spc_world(mesh_path: Path):
    """
    Sample surface + near-surface points in Blender world frame (meters).
    렌더링 코드와 동일한 방식으로 center/scale 계산:
      bb = obj.get_bound_box()
      center = (bb.max + bb.min) / 2
      scale  = 0.2 / (bb.max - bb.min).max()
    Blender imports OBJ with Y-up → Z-up conversion: (x,y,z) → (x,z,-y).
    """
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    # Blender OBJ import rotates Y-up OBJ to Z-up world: (x,y,z) -> (x,z,-y)
    R_yup = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    verts_blender = (R_yup @ mesh.vertices.T).T
    bb_min = verts_blender.min(axis=0)
    bb_max = verts_blender.max(axis=0)
    ctr  = (bb_min + bb_max) / 2.0
    scl  = 0.2 / (bb_max - bb_min).max()
    verts_norm = (verts_blender - ctr) * scl
    mesh_norm  = trimesh.Trimesh(vertices=verts_norm, faces=mesh.faces, process=False)

    # surface samples — SDF = 0
    pts_surf, fi_surf = trimesh.sample.sample_surface(mesh_norm, N_SPC_SURF)
    nrm_surf = mesh_norm.face_normals[fi_surf]
    sdf_surf = np.zeros(N_SPC_SURF, dtype=np.float32)

    # near-surface samples — 법선 방향으로 ±1.25mm 이내 offset, SDF = offset
    pts_on, fi_near = trimesh.sample.sample_surface(mesh_norm, N_SPC_NEAR)
    nrm_near = mesh_norm.face_normals[fi_near]
    offsets  = np.random.uniform(-1.25e-3, 1.25e-3, size=(N_SPC_NEAR, 1))
    pts_near = pts_on + nrm_near * offsets
    sdf_near = (offsets.squeeze() * 1000.0).astype(np.float32)  # m → mm, 양수=외부

    pts     = np.concatenate([pts_surf, pts_near], axis=0).astype(np.float32)
    normals = np.concatenate([nrm_surf, nrm_near],  axis=0).astype(np.float32)
    sdf_mm  = np.concatenate([sdf_surf, sdf_near],  axis=0).astype(np.float32)
    return pts, normals, sdf_mm


def encode_spc_npz(pts_world, normals_world, sdf_mm, c2w_path: Path) -> bytes:
    """Transform world-frame SPC to camera frame (mm) and encode as npz bytes."""
    c2w = torch.load(str(c2w_path), weights_only=True).numpy().astype(np.float64)
    w2c = np.linalg.inv(c2w)
    R   = w2c[:3, :3].astype(np.float32)
    t   = (w2c[:3, 3] * 1000.0).astype(np.float32)   # m → mm

    pts_cam = (R @ pts_world.T).T * 1000.0 + t        # world(m) → cam(mm)
    nrm_cam = (R @ normals_world.T).T
    nrm_cam /= np.linalg.norm(nrm_cam, axis=1, keepdims=True) + 1e-8

    spc     = np.column_stack([pts_cam, sdf_mm, nrm_cam]).astype(np.float16)
    obj_ids = np.zeros((len(spc), 1), dtype=np.uint16)

    buf = io.BytesIO()
    np.savez(buf, spc=spc, obj_ids=obj_ids)
    return buf.getvalue()


# ── tar helper ────────────────────────────────────────────────────────────────

def tar_add(tf: tarfile.TarFile, key: str, ext: str, data: bytes):
    info      = tarfile.TarInfo(name=f"{key}.{ext}")
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


# ── shard manager ─────────────────────────────────────────────────────────────

class ShardWriter:
    def __init__(self, root: Path, split: str, shard_size: int):
        self.root       = root / split
        self.split      = split
        self.shard_size = shard_size
        self.idx        = 0
        self.count      = 0
        self.total      = 0
        self._tf        = None
        self._open()

    def _open(self):
        if self._tf:
            self._tf.close()
        path    = self.root / f"shapenet_{self.idx:06d}.tar"
        self._tf = tarfile.open(str(path), 'w')

    def add(self, key: str, ext: str, data: bytes):
        tar_add(self._tf, key, ext, data)

    def commit(self):
        self.count += 1
        self.total += 1
        if self.count >= self.shard_size:
            self.idx  += 1
            self.count = 0
            self._open()

    def close(self):
        if self._tf:
            self._tf.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendered_root', required=True)
    parser.add_argument('--shapenet_root',  required=True)
    parser.add_argument('--output_dir',     required=True)
    parser.add_argument('--categories', nargs='*', default=None)
    parser.add_argument('--shard_size', type=int,   default=SHARD_SIZE)
    parser.add_argument('--val_ratio',  type=float, default=VAL_RATIO)
    parser.add_argument('--test_ratio', type=float, default=TEST_RATIO)
    parser.add_argument('--seed',       type=int,   default=42)
    args = parser.parse_args()

    rendered_root = Path(args.rendered_root)
    shapenet_root = Path(args.shapenet_root)
    output_dir    = Path(args.output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val').mkdir(parents=True,   exist_ok=True)
    (output_dir / 'test').mkdir(parents=True,  exist_ok=True)

    random.seed(args.seed)

    categories = args.categories or sorted(
        p.name for p in rendered_root.iterdir() if p.is_dir()
    )

    train_w = ShardWriter(output_dir, 'train', args.shard_size)
    val_w   = ShardWriter(output_dir, 'val',   args.shard_size)
    test_w  = ShardWriter(output_dir, 'test',  args.shard_size)

    for cat_id in categories:
        cat_dir = rendered_root / cat_id
        if not cat_dir.is_dir():
            print(f"[SKIP] category {cat_id} not found in rendered_root")
            continue

        obj_ids  = sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
        n_val    = max(1, int(len(obj_ids) * args.val_ratio))
        n_test   = max(1, int(len(obj_ids) * args.test_ratio))
        held_out = set(random.sample(obj_ids, n_val + n_test))
        val_set  = set(random.sample(sorted(held_out), n_val))
        test_set = held_out - val_set

        for obj_id in tqdm(obj_ids, desc=cat_id):
            mesh_path   = shapenet_root / cat_id / obj_id / "models" / "model_normalized.obj"
            cameras_dir = rendered_root / cat_id / obj_id / "cameras"

            if not mesh_path.exists():
                continue
            if not cameras_dir.exists():
                continue

            try:
                pts_w, nrm_w, sdf_mm = build_spc_world(mesh_path)
            except Exception as e:
                print(f"[SPC FAIL] {cat_id}/{obj_id}: {e}")
                continue

            if obj_id in val_set:
                w = val_w
            elif obj_id in test_set:
                w = test_w
            else:
                w = train_w
            key_pfx  = f"{cat_id}_{obj_id}"

            for view_dir in sorted(cameras_dir.iterdir()):
                try:
                    view_idx = int(view_dir.name)
                except ValueError:
                    continue

                key          = f"{key_pfx}_{view_idx:05d}"
                rgb_p        = view_dir / "rgb.png"
                depth_p      = view_dir / "depth.png"
                mask_p       = view_dir / "mask.png"
                intr_p       = view_dir / "intrinsics.pt"
                c2w_p        = view_dir / "cam2world.pt"
                tactile_p    = view_dir / "tactile.npz"

                if not all(p.exists() for p in [rgb_p, depth_p, mask_p, intr_p, c2w_p]):
                    continue

                try:
                    rgb_b        = encode_rgb(rgb_p)
                    depth_b      = encode_depth_png(load_depth_meters(depth_p))
                    cam_b        = encode_camera_json(intr_p, c2w_p)
                    mask_b, gtinfo_b, frac = encode_mask_jsons(mask_p)
                    if frac == 0.0:
                        continue
                    gt_b         = encode_gt_json(c2w_p)
                    spc_b        = encode_spc_npz(pts_w, nrm_w, sdf_mm, c2w_p)
                except Exception as e:
                    print(f"[SKIP] {key}: {e}")
                    continue

                w.add(key, 'rgb.jpg',        rgb_b)
                w.add(key, 'depth.png',       depth_b)
                w.add(key, 'camera.json',     cam_b)
                w.add(key, 'gt.json',         gt_b)
                w.add(key, 'gt_info.json',    gtinfo_b)
                w.add(key, 'mask_visib.json', mask_b)
                w.add(key, 'spc.npz',         spc_b)
                if tactile_p.exists() and tactile_p.stat().st_size > 0:
                    w.add(key, 'tactile.npz', tactile_p.read_bytes())

                w.commit()

    train_w.close()
    val_w.close()
    test_w.close()

    print(f"\nDone.")
    print(f"  train: {train_w.total} samples in {train_w.idx + 1} shards")
    print(f"  val:   {val_w.total} samples in {val_w.idx + 1} shards")
    print(f"  test:  {test_w.total} samples in {test_w.idx + 1} shards")
    print(f"\nConfig values to use:")
    print(f"  train_dataset_size: {train_w.total}")
    print(f"  val_dataset_size:   {val_w.total}")
    print(f"  test_dataset_size:  {test_w.total}")
    print(f"  train_dataset_url:  {output_dir}/train/shapenet_{{000000..{train_w.idx:06d}}}.tar")
    print(f"  val_dataset_url:    {output_dir}/val/shapenet_{{000000..{val_w.idx:06d}}}.tar")
    print(f"  test_dataset_url:   {output_dir}/test/shapenet_{{000000..{test_w.idx:06d}}}.tar")


if __name__ == '__main__':
    main()
