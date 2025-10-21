## Synthetic Bolt Dataset Generator

Generates orthographic images (top, front, side, rear) of a 12mm bolt from `create-a-12mm-bolt.gltf` with slight rotations and lighting variations. Adds small simulated 2D defects on some images and exports YOLOv8 labels for bolt and defects.

### Setup

1. Python 3.10+ recommended.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, ensure a working OpenGL context is available (GPU drivers up to date). If running headless, you may need `osmesa`/`egl`; this script targets desktop OpenGL via `pyrender`.

### Usage

```bash
python generate_bolt_dataset.py --input create-a-12mm-bolt.gltf --out_dir dataset --images_per_view 50 --img_size 640 --val_fraction 0.2
``)

Outputs structure:

```
dataset/
  images/
    train/*.png
    val/*.png
  labels/
    train/*.txt
    val/*.txt
  dataset.yaml
```

YOLO classes: `0 = bolt`, `1 = defect`.


