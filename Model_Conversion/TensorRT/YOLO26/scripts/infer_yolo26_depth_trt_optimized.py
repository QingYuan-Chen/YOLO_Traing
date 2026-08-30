from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmark_yolo26_trt_native import (
    NativeTensorRTRunner,
    capture_cuda_graph,
    elapsed_gpu_ms,
    gpu_letterbox_bgr_to_rgb,
)


def latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.mean(ordered),
    }


class OptimizedDepthPipeline:
    """Fixed-shape YOLO26 depth TensorRT pipeline with persistent buffers and CUDA Graph."""

    def __init__(
        self,
        engine_path: Path,
        source_shape: tuple[int, int, int],
        preprocess: str,
    ) -> None:
        if len(source_shape) != 3 or source_shape[2] != 3:
            raise ValueError(f"Expected HWC BGR source, got {source_shape}")
        self.runner = NativeTensorRTRunner.load(engine_path)
        self.stream = torch.cuda.Stream()
        self.preprocess = preprocess
        self.source_shape = source_shape
        self.target_shape = tuple(self.runner.input.shape[2:])
        if len(self.runner.output_names) != 1:
            raise ValueError(f"Expected one depth output, found {self.runner.output_names}")
        output = self.runner.tensors[self.runner.output_names[0]]
        if output.shape[:2] != (1, 1):
            raise ValueError(f"Expected NCHW batch-1 depth output, got {tuple(output.shape)}")

        source_height, source_width = source_shape[:2]
        target_height, target_width = self.target_shape
        scale = min(target_height / source_height, target_width / source_width)
        self.resized_height = round(source_height * scale)
        self.resized_width = round(source_width * scale)
        self.top = round((target_height - self.resized_height) / 2 - 0.1)
        self.left = round((target_width - self.resized_width) / 2 - 0.1)

        if preprocess == "fast-gpu":
            self.source_cpu = torch.empty(source_shape, dtype=torch.uint8, pin_memory=True)
        elif preprocess == "exact-cpu":
            self.source_cpu = torch.empty(
                (target_height, target_width, 3), dtype=torch.uint8, pin_memory=True
            )
        else:
            raise ValueError(f"Unsupported preprocess mode: {preprocess}")
        self.source_gpu = torch.empty_like(self.source_cpu, device="cuda")
        self.depth_cpu = torch.empty(
            (source_height, source_width), dtype=torch.float32, pin_memory=True
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.depth_gpu: torch.Tensor | None = None

    def prepare_frame(self, image: np.ndarray) -> None:
        if image.shape != self.source_shape:
            raise ValueError(
                f"CUDA Graph requires fixed source shape {self.source_shape}, got {image.shape}"
            )
        if self.preprocess == "fast-gpu":
            self.source_cpu.copy_(torch.from_numpy(np.ascontiguousarray(image)))
            return

        target = self.source_cpu.numpy()
        target.fill(114)
        resized = cv2.resize(
            image,
            (self.resized_width, self.resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        target[
            self.top : self.top + self.resized_height,
            self.left : self.left + self.resized_width,
        ] = resized

    def _exact_gpu_upload_and_normalize(self) -> None:
        self.source_gpu.copy_(self.source_cpu, non_blocking=True)
        destination = self.runner.input
        destination[0, 0].copy_(self.source_gpu[:, :, 2])
        destination[0, 1].copy_(self.source_gpu[:, :, 1])
        destination[0, 2].copy_(self.source_gpu[:, :, 0])
        destination.mul_(1.0 / 255.0)

    def _operation(self) -> torch.Tensor:
        if self.preprocess == "fast-gpu":
            gpu_letterbox_bgr_to_rgb(self.source_cpu, self.source_gpu, self.runner.input)
        else:
            self._exact_gpu_upload_and_normalize()
        self.runner.execute(self.stream)
        depth = self.runner.tensors[self.runner.output_names[0]]
        depth = depth[
            ...,
            self.top : self.top + self.resized_height,
            self.left : self.left + self.resized_width,
        ]
        return F.interpolate(
            depth.float(),
            size=self.source_shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    def capture(self, warmup: int) -> None:
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.depth_gpu = self._operation()
        self.stream.synchronize()
        self.graph, self.depth_gpu = capture_cuda_graph(self._operation, self.stream)

    def infer_gpu(self, image: np.ndarray, synchronize: bool = True) -> torch.Tensor:
        if self.graph is None or self.depth_gpu is None:
            raise RuntimeError("capture() must be called before inference")
        self.prepare_frame(image)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        if synchronize:
            self.stream.synchronize()
        return self.depth_gpu

    def infer_cpu(self, image: np.ndarray) -> torch.Tensor:
        if self.graph is None or self.depth_gpu is None:
            raise RuntimeError("capture() must be called before inference")
        self.prepare_frame(image)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
            self.depth_cpu.copy_(self.depth_gpu, non_blocking=True)
        self.stream.synchronize()
        return self.depth_cpu

    def benchmark_graph_gpu_ms(self, warmup: int, iterations: int) -> float:
        if self.graph is None:
            raise RuntimeError("capture() must be called before benchmarking")
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.graph.replay()
            self.stream.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            for _ in range(iterations):
                self.graph.replay()
            end.record(self.stream)
            end.synchronize()
        return start.elapsed_time(end) / iterations


def save_depth_preview(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Depth output has no finite positive values")
    near_m, far_m = np.percentile(depth[valid], [2.0, 98.0])
    clipped = np.clip(depth, near_m, far_m)
    inverse = (far_m - clipped) / max(far_m - near_m, 1e-6)
    color = cv2.applyColorMap(np.round(inverse * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.putText(
        color,
        f"near={near_m:.2f}m far={far_m:.2f}m",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), color):
        raise RuntimeError(f"Failed to save {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimized YOLO26 depth TensorRT inference with CUDA Graph."
    )
    parser.add_argument("engine", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--preprocess", choices=["exact-cpu", "fast-gpu"], default="exact-cpu"
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-copy-iterations", type=int, default=30)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--save-preview", type=Path)
    parser.add_argument("--save-depth-npy", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read {args.image}")
    pipeline = OptimizedDepthPipeline(args.engine, image.shape, args.preprocess)
    pipeline.prepare_frame(image)
    pipeline.capture(args.warmup)

    engine_only_ms = elapsed_gpu_ms(
        lambda: pipeline.runner.execute(pipeline.stream),
        pipeline.stream,
        args.warmup,
        args.iterations,
    )
    graph_gpu_ms = pipeline.benchmark_graph_gpu_ms(args.warmup, args.iterations)

    synchronized_samples: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        pipeline.infer_gpu(image, synchronize=True)
        synchronized_samples.append((time.perf_counter() - started) * 1_000)

    cpu_samples: list[float] = []
    depth_cpu: torch.Tensor | None = None
    for _ in range(args.cpu_copy_iterations):
        started = time.perf_counter()
        depth_cpu = pipeline.infer_cpu(image)
        _ = depth_cpu[0, 0].item()
        cpu_samples.append((time.perf_counter() - started) * 1_000)
    if depth_cpu is None:
        raise RuntimeError("No CPU output was produced")
    depth = depth_cpu.numpy().copy()
    valid = np.isfinite(depth) & (depth > 0)

    results = {
        "engine": str(args.engine),
        "image": str(args.image),
        "gpu": torch.cuda.get_device_name(0),
        "preprocess": args.preprocess,
        "source_shape_hwc": list(image.shape),
        "engine_input_shape_nchw": list(pipeline.runner.input.shape),
        "engine_only_gpu_ms": engine_only_ms,
        "engine_only_gpu_fps": 1_000.0 / engine_only_ms,
        "cuda_graph_preprocess_engine_resize_gpu_ms": graph_gpu_ms,
        "cuda_graph_preprocess_engine_resize_gpu_fps": 1_000.0 / graph_gpu_ms,
        "host_frame_to_gpu_depth_latency_ms": latency_summary(synchronized_samples),
        "host_frame_to_cpu_depth_latency_ms": latency_summary(cpu_samples),
        "depth_m": {
            "min": float(depth[valid].min()),
            "max": float(depth[valid].max()),
            "mean": float(depth[valid].mean()),
            "median": float(np.median(depth[valid])),
            "finite_positive_fraction": float(valid.mean()),
        },
    }
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.save_preview:
        save_depth_preview(args.save_preview, depth)
    if args.save_depth_npy:
        args.save_depth_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_depth_npy, depth)


if __name__ == "__main__":
    main()
