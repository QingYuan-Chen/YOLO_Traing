from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.mean(ordered),
        "fps_from_median": 1_000.0 / statistics.median(ordered),
    }


def depth_array(result: object) -> np.ndarray:
    depth = getattr(result, "depth", None)
    if depth is None:
        raise AttributeError("Result has no depth output")
    tensor = depth.data if hasattr(depth, "data") else depth
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor.detach().float().cpu().numpy()


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    difference = candidate64 - reference64
    ratio = np.maximum(reference64 / candidate64, candidate64 / reference64)
    return {
        "mae_m": float(np.mean(np.abs(difference))),
        "rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_abs_relative": float(np.mean(np.abs(difference) / reference64)),
        "delta1_agreement": float(np.mean(ratio < 1.25)),
        "max_abs_m": float(np.max(np.abs(difference))),
    }


def benchmark(
    model_path: Path,
    image: np.ndarray,
    imgsz: int,
    warmup: int,
    iterations: int,
) -> tuple[dict[str, object], np.ndarray]:
    model = YOLO(str(model_path), task="depth")

    def predict() -> object:
        return model.predict(
            source=image,
            imgsz=imgsz,
            rect=False,
            batch=1,
            device=0,
            verbose=False,
        )[0]

    result = predict()
    for _ in range(warmup):
        result = predict()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        result = predict()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1_000)
    depth = depth_array(result)
    report = {
        "model": str(model_path),
        "latency_ms": latency_summary(samples),
        "ultralytics_last_result_speed_ms": dict(result.speed),
        "depth_m": {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "median": float(np.median(depth)),
            "finite_fraction": float(np.isfinite(depth).mean()),
        },
    }
    return report, depth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO26 depth PT and TensorRT with identical square preprocessing."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    pt_report, pt_depth = benchmark(
        args.weights, image, args.imgsz, args.warmup, args.iterations
    )
    engine_report, engine_depth = benchmark(
        args.engine, image, args.imgsz, args.warmup, args.iterations
    )
    report = {
        "test_contract": {
            "image": str(args.image),
            "imgsz": args.imgsz,
            "batch": 1,
            "rect": False,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "gpu": torch.cuda.get_device_name(0),
        },
        "pytorch": pt_report,
        "tensorrt": engine_report,
        "tensorrt_versus_pytorch": compare(pt_depth, engine_depth),
        "median_end_to_end_speedup": (
            pt_report["latency_ms"]["median"] / engine_report["latency_ms"]["median"]
        ),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
