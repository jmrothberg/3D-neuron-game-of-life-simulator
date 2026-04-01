#!/usr/bin/env python3
"""
Download MNIST or Fashion-MNIST (IDX, gzip) from the public web and write
JSON suitable for the browser simulator (press M → J and pick this file).

Output shape matches neurosim/training.py: one-hot target is at cells[9:19, 14]
(first index x = 9..18, second y = 14), same as neurosim_web.js prediction.

Uses only the Python standard library (no TensorFlow).
"""
from __future__ import annotations

import argparse
import gzip
import json
import ssl
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Grid matches neurosim.config WIDTH × HEIGHT
GRID = 28
EPS = 1e-8

# Default 12-gene “stamp” compatible with neurosim Cell (non-autonomous template).
# weight count 9 → reach 1; only charge matters for input/output training layers.
def _default_genes():
    return [
        15, 2, 3, 10,  # breeding + MR
        9, 0.01, 5, 0.001, 1e-6,  # WG=9, BR, AW, CD, WD
        0.01, 1e-7, 0.1,  # LR, GT, AS
    ]


def _training_cell(layer: int, x: int, y: int, charge: float) -> dict:
    genes = _default_genes()
    n_weights = int(genes[4])
    wm = int(n_weights ** 0.5)
    reach = (wm - 1) // 2
    return {
        "x": x,
        "y": y,
        "layer": layer,
        "genes": genes,
        "weights": [0.0] * n_weights,
        "bias": 0.0,
        "charge": float(charge),
        "error": EPS,
        "gradient": 0.0,
        "reach": reach,
        "forward_charges": [],
        "reverse_charges": [],
        "max_charge_diff_forward": 0,
        "max_charge_diff_reverse": 0,
        "significant_charge_change_forward": False,
        "significant_charge_change_reverse": False,
        "gradient_history": [],
        "avg_gradient_magnitude": 0,
        "significant_gradient_change": False,
    }


URLS = {
    "mnist": {
        "images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    },
    "fashion": {
        "images": "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "labels": "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    },
}


def download(url: str, dest: Path, timeout: int = 120, verify_ssl: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} → {dest}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "neurosim-mnist-json/1.0"})
    ctx = None if verify_ssl else ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        dest.write_bytes(resp.read())


def parse_images_gz(path: Path) -> tuple[int, int, int, list]:
    """Return rows, cols, count, and list of count × rows × cols integers 0–255."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic}")
        buf = f.read()
    expected = n * rows * cols
    if len(buf) != expected:
        raise ValueError(f"image buffer length {len(buf)} != {expected}")
    images = []
    stride = rows * cols
    for i in range(n):
        chunk = buf[i * stride : (i + 1) * stride]
        img = []
        for r in range(rows):
            row = list(chunk[r * cols : (r + 1) * cols])
            img.append(row)
        images.append(img)
    return rows, cols, n, images


def parse_labels_gz(path: Path) -> list[int]:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic}")
        buf = f.read()
        if len(buf) != n:
            raise ValueError("label length mismatch")
    return list(buf)


def build_sample(image: list[list[int]], label: int, layer_last: int) -> dict:
    """
    image: 28×28 raw ints 0–255, row-major (image[x][y] = pixel at x,y).
    """
    layer0 = []
    layer_l = []
    for x in range(GRID):
        row0 = []
        rowl = []
        for y in range(GRID):
            ch = image[x][y] / 255.0
            row0.append(_training_cell(0, x, y, ch))
            rowl.append(None)
        layer0.append(row0)
        layer_l.append(rowl)

    one_hot = [1.0 if d == label else 0.0 for d in range(10)]
    for xi, val in zip(range(9, 19), one_hot):
        layer_l[xi][14] = _training_cell(layer_last, xi, 14, val)

    return {"layer0": layer0, "layerLast": layer_l}


def main() -> int:
    ap = argparse.ArgumentParser(description="MNIST/Fashion-MNIST → neurosim_web training JSON")
    ap.add_argument(
        "--dataset",
        choices=("mnist", "fashion"),
        default="mnist",
        help="Which dataset to download (default: mnist)",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".mnist_cache",
        help="Where to store downloaded gzip files",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "mnist_training_web.json",
        help="Output JSON path",
    )
    ap.add_argument("--count", type=int, default=500, help="Number of samples (max train size)")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N training images")
    ap.add_argument(
        "--layer-last",
        type=int,
        default=15,
        help="Layer index stored in layerLast (default 15, like old pickles; web uses num_layers-1 at runtime)",
    )
    ap.add_argument("--indent", type=int, default=None, help="JSON indent (default: compact)")
    ap.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable TLS certificate verification (use if downloads fail with CERTIFICATE_VERIFY_FAILED)",
    )
    args = ap.parse_args()

    if args.count < 1 or args.offset < 0:
        print("count must be >= 1 and offset >= 0", file=sys.stderr)
        return 2

    urls = URLS[args.dataset]
    img_path = args.cache_dir / f"{args.dataset}_train_images.gz"
    lbl_path = args.cache_dir / f"{args.dataset}_train_labels.gz"

    verify = not args.no_verify_ssl
    if not verify:
        print("Warning: SSL verification disabled for downloads.", file=sys.stderr)
    try:
        download(urls["images"], img_path, verify_ssl=verify)
        download(urls["labels"], lbl_path, verify_ssl=verify)
    except urllib.error.URLError as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1

    rows, cols, n_total, images = parse_images_gz(img_path)
    labels = parse_labels_gz(lbl_path)
    if len(labels) != n_total:
        raise SystemExit("image/label count mismatch")
    if rows != GRID or cols != GRID:
        raise SystemExit(f"expected {GRID}×{GRID} images, got {rows}×{cols}")

    end = min(args.offset + args.count, n_total)
    if args.offset >= n_total:
        print("offset past end of dataset", file=sys.stderr)
        return 2

    samples = []
    for i in range(args.offset, end):
        samples.append(build_sample(images[i], labels[i], args.layer_last))

    payload = {
        "format": "neurosim_web_training_v1",
        "dataset": args.dataset,
        "width": GRID,
        "height": GRID,
        "offset": args.offset,
        "count": len(samples),
        "layer_last_index_note": args.layer_last,
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=args.indent)
    print(f"Wrote {len(samples)} samples to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
