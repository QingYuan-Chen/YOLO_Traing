from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a YOLO26 PyTorch checkpoint to a static TensorRT FP16 engine."
    )
    parser.add_argument("weights", type=Path)
    parser.add_argument("--task", choices=["segment", "depth"], required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace in GiB")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.output and args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to replace it: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for TensorRT export")

    model = YOLO(str(args.weights), task=args.task)
    exported = Path(
        model.export(
            format="engine",
            imgsz=args.imgsz,
            batch=1,
            dynamic=False,
            quantize=16,
            simplify=True,
            workspace=args.workspace,
            device=0,
        )
    )
    output = exported
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, args.output)
        output = args.output

    print(f"engine={output.resolve()}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={sha256(output)}")
    print(f"gpu={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
