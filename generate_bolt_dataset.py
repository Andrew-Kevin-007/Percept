import os
import random
import math
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import yaml

import trimesh
import pyrender
import pyglet


# ----------------------------
# Configuration structures
# ----------------------------

@dataclass
class ViewSpec:
    name: str
    eye: np.ndarray  # camera position
    up: np.ndarray   # camera up vector


@dataclass
class DefectSpec:
    kind: str  # scratch | spot | none
    bbox_xyxy: Optional[Tuple[int, int, int, int]]


# ----------------------------
# Utility helpers
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def yolo_bbox_from_xyxy(xmin: int, ymin: int, xmax: int, ymax: int, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # Clamp to image bounds
    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymax = max(0, min(ymax, img_h - 1))
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    cx = xmin + w / 2.0
    cy = ymin + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def random_direction_on_sphere(rng: random.Random) -> np.ndarray:
    theta = rng.uniform(0, 2 * math.pi)
    u = rng.uniform(-1, 1)
    s = math.sqrt(1 - u * u)
    return np.array([s * math.cos(theta), s * math.sin(theta), u], dtype=np.float32)


def compose_on_white(color_img: np.ndarray, depth: np.ndarray) -> np.ndarray:
    # Replace background (inf depth) with white
    h, w, _ = color_img.shape
    out = color_img.copy()
    bg_mask = ~np.isfinite(depth)
    out[bg_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def compute_object_bbox_from_depth(depth: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    mask = np.isfinite(depth)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    return xmin, ymin, xmax, ymax


def add_defects_pil(img: Image.Image, rng: random.Random, max_defects: int = 2) -> List[DefectSpec]:
    draw = ImageDraw.Draw(img)
    defects: List[DefectSpec] = []
    w, h = img.size
    num_defects = rng.choice([0, 0, 1, 1, 2])  # more often none or one
    for _ in range(min(max_defects, num_defects)):
        kind = rng.choice(["scratch", "spot"])
        if kind == "scratch":
            # Random thin line with slight blur to mimic scratch
            x0 = rng.randint(int(0.1 * w), int(0.9 * w))
            y0 = rng.randint(int(0.1 * h), int(0.9 * h))
            length = rng.randint(int(0.15 * w), int(0.35 * w))
            angle = rng.uniform(0, math.pi)
            x1 = int(x0 + length * math.cos(angle))
            y1 = int(y0 + length * math.sin(angle))
            thickness = rng.randint(1, 3)
            color = (rng.randint(80, 140),) * 3  # darker gray
            draw.line([(x0, y0), (x1, y1)], fill=color, width=thickness)
            # Create bbox around the line with some padding
            xmin = max(0, min(x0, x1) - 2)
            xmax = min(w - 1, max(x0, x1) + 2)
            ymin = max(0, min(y0, y1) - 2)
            ymax = min(h - 1, max(y0, y1) + 2)
            defects.append(DefectSpec(kind="scratch", bbox_xyxy=(xmin, ymin, xmax, ymax)))
        else:
            # Small spot imperfection
            cx = rng.randint(int(0.2 * w), int(0.8 * w))
            cy = rng.randint(int(0.2 * h), int(0.8 * h))
            radius = rng.randint(2, 6)
            color = (rng.randint(60, 120),) * 3
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.ellipse(bbox, fill=color)
            defects.append(DefectSpec(kind="spot", bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3])))

    # Slight overall perturbation to mimic sensor
    if rng.random() < 0.3:
        img.putalpha(255)
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        img.paste(img_blur)
        img = img.convert("RGB")
    return defects


# ----------------------------
# Rendering pipeline
# ----------------------------

def load_and_normalize_mesh(gltf_path: str) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(gltf_path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene_or_mesh.dump())
    else:
        mesh = scene_or_mesh

    # Center at origin
    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    # Uniform scale to fit within unit cube (approx) for stable ortho sizing
    extents = mesh.extents
    max_extent = float(max(extents)) if isinstance(extents, (list, tuple, np.ndarray)) else float(extents)
    if max_extent <= 0:
        max_extent = 1.0
    scale = 1.0 / max_extent
    mesh.apply_scale(scale)
    return mesh


def get_view_specs() -> List[ViewSpec]:
    # Ortho views from canonical axes; slight radius so object fills view via ortho scale
    radius = 2.0
    return [
        ViewSpec("top", eye=np.array([0.0, 0.0, radius], dtype=np.float32), up=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ViewSpec("front", eye=np.array([0.0, radius, 0.0], dtype=np.float32), up=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ViewSpec("side", eye=np.array([radius, 0.0, 0.0], dtype=np.float32), up=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ViewSpec("rear", eye=np.array([0.0, -radius, 0.0], dtype=np.float32), up=np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = (target - eye).astype(np.float32)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.eye(4, dtype=np.float32)
    R[0, :3] = right
    R[1, :3] = true_up
    R[2, :3] = -forward
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return R @ T


def render_sample(
    scene_mesh: trimesh.Trimesh,
    view: ViewSpec,
    img_size: int,
    rng: random.Random,
    angle_jitter_deg: float = 5.0,
) -> Tuple[Image.Image, np.ndarray]:
    ensure_gl_context()
    # Create pyrender scene
    scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.15, 0.15, 0.15])

    # Slight model rotation jitter to simulate small misalignments
    rx = math.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
    ry = math.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
    rz = math.radians(rng.uniform(-angle_jitter_deg, angle_jitter_deg))
    Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
    Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
    Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    transform = Rz @ Ry @ Rx

    # Optionally slight uniform scale to mimic dimension deviation
    if rng.random() < 0.2:
        s = rng.uniform(0.95, 1.05)
        S = np.eye(4)
        S[:3, :3] *= s
        transform = S @ transform

    tri = scene_mesh.copy()
    tri.apply_transform(transform)

    # Assign a simple gray material to keep appearance consistent
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.4, roughnessFactor=0.6, baseColorFactor=[0.75, 0.75, 0.75, 1.0]
    )
    mesh = pyrender.Mesh.from_trimesh(tri, material=material, smooth=True)
    scene.add(mesh)

    # Orthographic camera: set xmag/ymag so model fits nicely with margins
    # Since we normalized to unit cube, use ymagnification ~ 0.8 to frame tightly
    ymag = 0.8
    xmag = 0.8
    cam = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    cam_pose = look_at(view.eye, np.array([0.0, 0.0, 0.0], dtype=np.float32), view.up)
    scene.add(cam, pose=cam_pose)

    # Lighting variations: 1-2 directional lights from random directions
    num_lights = rng.choice([1, 2])
    for _ in range(num_lights):
        direction = random_direction_on_sphere(rng)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=rng.uniform(2.0, 4.0))
        # Place the light a bit far in the direction
        light_pose = np.eye(4)
        light_pose[:3, 3] = -direction * 5.0
        scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(viewport_width=img_size, viewport_height=img_size)
    # Use default render flags to reduce OpenGL feature requirements
    color, depth = r.render(scene)
    r.delete()

    color = compose_on_white(color, depth)
    pil_img = Image.fromarray(color)
    return pil_img, depth


def save_yolo_labels(label_path: str, labels: List[Tuple[int, float, float, float, float]]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def generate_dataset(
    gltf_path: str,
    out_dir: str = "dataset",
    images_per_view: int = 50,
    img_size: int = 640,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> None:
    ensure_gl_context()
    rng = random.Random(seed)

    # Prepare output folders
    images_train = os.path.join(out_dir, "images", "train")
    images_val = os.path.join(out_dir, "images", "val")
    labels_train = os.path.join(out_dir, "labels", "train")
    labels_val = os.path.join(out_dir, "labels", "val")
    for p in [images_train, images_val, labels_train, labels_val]:
        ensure_dir(p)

    # Load and normalize mesh once
    mesh = load_and_normalize_mesh(gltf_path)
    views = get_view_specs()

    # Accumulate file lists for yaml
    train_image_paths: List[str] = []
    val_image_paths: List[str] = []

    # Class mapping: 0=bolt, 1=defect
    for view in views:
        for i in range(images_per_view):
            pil_img, depth = render_sample(mesh, view, img_size, rng)

            # Compute bolt bbox from depth mask
            bbox_xyxy = compute_object_bbox_from_depth(depth)
            labels: List[Tuple[int, float, float, float, float]] = []
            if bbox_xyxy is not None:
                cx, cy, w, h = yolo_bbox_from_xyxy(bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3], img_size, img_size)
                labels.append((0, cx, cy, w, h))

            # Add 2D defects (some images only)
            defects: List[DefectSpec] = []
            if rng.random() < 0.6:  # probability to add any defects
                defects = add_defects_pil(pil_img, rng)

            # Convert defect bboxes to YOLO
            for d in defects:
                if d.bbox_xyxy is None:
                    continue
                xmin, ymin, xmax, ymax = d.bbox_xyxy
                cx, cy, w, h = yolo_bbox_from_xyxy(xmin, ymin, xmax, ymax, img_size, img_size)
                labels.append((1, cx, cy, w, h))

            # Split into train/val
            is_val = rng.random() < val_fraction
            subset_images = images_val if is_val else images_train
            subset_labels = labels_val if is_val else labels_train

            base_name = f"bolt_{view.name}_{i:04d}"
            img_path = os.path.join(subset_images, base_name + ".png")
            lbl_path = os.path.join(subset_labels, base_name + ".txt")
            pil_img.save(img_path, format="PNG")
            save_yolo_labels(lbl_path, labels)

            # Track for yaml as relative paths
            rel_path = os.path.relpath(img_path, out_dir)
            if is_val:
                val_image_paths.append(rel_path)
            else:
                train_image_paths.append(rel_path)

    # Write dataset.yaml
    dataset_yaml = {
        "path": out_dir,
        "train": os.path.join("images", "train"),
        "val": os.path.join("images", "val"),
        "names": {0: "bolt", 1: "defect"},
    }
    with open(os.path.join(out_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)


# Maintain a hidden pyglet window so that a valid OpenGL context exists for offscreen rendering.
_GL_WINDOW: Optional[pyglet.window.Window] = None


def ensure_gl_context() -> None:
    global _GL_WINDOW
    if _GL_WINDOW is None:
        # Create a tiny hidden window; this creates and makes current an OpenGL context
        _GL_WINDOW = pyglet.window.Window(width=1, height=1, visible=False)
        _GL_WINDOW.switch_to()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic bolt dataset from GLTF.")
    parser.add_argument("--input", type=str, default="create-a-12mm-bolt.gltf", help="Path to GLTF file")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output dataset directory")
    parser.add_argument("--images_per_view", type=int, default=50, help="Number of images per canonical view")
    parser.add_argument("--img_size", type=int, default=640, help="Square image size in pixels")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input GLTF not found: {args.input}")

    # Clean output if exists to avoid stale files
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    generate_dataset(
        gltf_path=args.input,
        out_dir=args.out_dir,
        images_per_view=args.images_per_view,
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


