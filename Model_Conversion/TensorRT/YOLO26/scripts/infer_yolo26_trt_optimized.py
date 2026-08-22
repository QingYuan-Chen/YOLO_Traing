from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from benchmark_yolo26_trt_native import (
    NativeTensorRTRunner,
    capture_cuda_graph,
    gpu_letterbox_bgr_to_rgb,
    segment_postprocess_static,
)


class OptimizedSegmentationPipeline:
    def __init__(
        self,
        engine_path: Path,
        source_shape: tuple[int, int, int],
        preprocess: str,
        confidence: float,
        max_masks: int,
    ) -> None:
        if len(source_shape) != 3 or source_shape[2] != 3:
            raise ValueError(f"Expected HWC BGR source, got {source_shape}")
        self.runner = NativeTensorRTRunner.load(engine_path)
        self.stream = torch.cuda.Stream()
        self.preprocess = preprocess
        self.confidence = confidence
        self.max_masks = max_masks
        self.target_shape = tuple(self.runner.input.shape[2:])
        self.source_shape = source_shape
        available_detections = self.runner.tensors["output0"].shape[1]
        if not 1 <= max_masks <= available_detections:
            raise ValueError(
                f"max_masks must be between 1 and {available_detections}, got {max_masks}"
            )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        output_dtype = self.runner.tensors["output0"].dtype
        target_height, target_width = self.target_shape
        self.detections_cpu = torch.empty(
            (max_masks, 6), dtype=output_dtype, pin_memory=True
        )
        self.valid_cpu = torch.empty((max_masks,), dtype=torch.bool, pin_memory=True)
        self.masks_cpu = torch.empty(
            (max_masks, target_height, target_width),
            dtype=torch.uint8,
            pin_memory=True,
        )

        if preprocess == "fast-gpu":
            self.source_cpu = torch.empty(
                source_shape, dtype=torch.uint8, pin_memory=True
            )
            self.source_gpu = torch.empty_like(self.source_cpu, device="cuda")
        elif preprocess == "exact-cpu":
            target_height, target_width = self.target_shape
            target_hwc = (target_height, target_width, 3)
            self.source_cpu = torch.empty(
                target_hwc, dtype=torch.uint8, pin_memory=True
            )
            self.source_gpu = torch.empty_like(self.source_cpu, device="cuda")
        else:
            raise ValueError(f"Unsupported preprocess mode: {preprocess}")

    def prepare_frame(self, image: np.ndarray) -> None:
        if image.shape != self.source_shape:
            raise ValueError(
                f"CUDA Graph requires fixed source shape {self.source_shape}, got {image.shape}"
            )
        if self.preprocess == "fast-gpu":
            contiguous = np.ascontiguousarray(image)
            self.source_cpu.copy_(torch.from_numpy(contiguous))
            return

        target = self.source_cpu.numpy()
        target.fill(114)
        source_height, source_width = image.shape[:2]
        target_height, target_width = self.target_shape
        scale = min(target_height / source_height, target_width / source_width)
        resized_height = round(source_height * scale)
        resized_width = round(source_width * scale)
        top = round((target_height - resized_height) / 2 - 0.1)
        left = round((target_width - resized_width) / 2 - 0.1)
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        target[
            top : top + resized_height,
            left : left + resized_width,
        ] = resized

    def _exact_gpu_upload_and_normalize(self) -> None:
        self.source_gpu.copy_(self.source_cpu, non_blocking=True)
        destination = self.runner.input
        destination[0, 0].copy_(self.source_gpu[:, :, 2])
        destination[0, 1].copy_(self.source_gpu[:, :, 1])
        destination[0, 2].copy_(self.source_gpu[:, :, 0])
        destination.mul_(1.0 / 255.0)

    def _operation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.preprocess == "fast-gpu":
            gpu_letterbox_bgr_to_rgb(
                self.source_cpu,
                self.source_gpu,
                self.runner.input,
            )
        else:
            self._exact_gpu_upload_and_normalize()
        self.runner.execute(self.stream)
        return segment_postprocess_static(
            self.runner.tensors["output0"],
            self.runner.tensors["output1"],
            self.confidence,
            self.max_masks,
            self.target_shape,
            upsample=True,
        )

    def capture(self, warmup: int) -> None:
        with torch.cuda.stream(self.stream):
            for _ in range(warmup):
                self.static_outputs = self._operation()
        self.stream.synchronize()
        self.graph, captured_outputs = capture_cuda_graph(self._operation, self.stream)
        self.static_outputs = captured_outputs

    def infer_gpu(
        self,
        image: np.ndarray,
        synchronize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph is None or self.static_outputs is None:
            raise RuntimeError("capture() must be called before infer_gpu()")
        self.prepare_frame(image)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        if synchronize:
            self.stream.synchronize()
        return self.static_outputs

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

    def infer_cpu(
        self,
        image: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.graph is None or self.static_outputs is None:
            raise RuntimeError("capture() must be called before infer_cpu()")
        self.prepare_frame(image)
        detections, valid, masks = self.static_outputs
        with torch.cuda.stream(self.stream):
            self.graph.replay()
            self.detections_cpu.copy_(detections, non_blocking=True)
            self.valid_cpu.copy_(valid, non_blocking=True)
            self.masks_cpu.copy_(masks, non_blocking=True)
        self.stream.synchronize()
        return self.detections_cpu, self.valid_cpu, self.masks_cpu


def latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "mean": statistics.mean(ordered),
    }


def cpu_result_copy_ms(
    pipeline: OptimizedSegmentationPipeline,
    image: np.ndarray,
    iterations: int,
) -> tuple[dict[str, float], int]:
    samples: list[float] = []
    detection_count = 0
    for _ in range(iterations):
        started = time.perf_counter()
        detections, valid, masks = pipeline.infer_cpu(image)
        detection_count = int(valid.sum().item())
        _ = detections[0, 0].item()
        _ = masks[0, 0, 0].item()
        samples.append((time.perf_counter() - started) * 1_000)
    return latency_summary(samples), detection_count


def create_letterboxed_background(
    image: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    target_height, target_width = target_shape
    scale = min(target_height / source_height, target_width / source_width)
    resized_height = round(source_height * scale)
    resized_width = round(source_width * scale)
    top = round((target_height - resized_height) / 2 - 0.1)
    left = round((target_width - resized_width) / 2 - 0.1)
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return canvas


def save_overlay(
    path: Path,
    image: np.ndarray,
    target_shape: tuple[int, int],
    detections: torch.Tensor,
    valid: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    count = int(valid.sum().item())
    boxes = detections[:count].cpu().numpy()
    mask_array = masks[:count].cpu().numpy().astype(bool)
    canvas = create_letterboxed_background(image, target_shape)
    colors = [(0, 255, 0), (255, 80, 80), (80, 160, 255), (255, 0, 255)]
    for index, (box, mask) in enumerate(zip(boxes, mask_array)):
        color = np.array(colors[index % len(colors)], dtype=np.float32)
        canvas[mask] = (canvas[mask] * 0.55 + color * 0.45).astype(np.uint8)
        x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colors[index % len(colors)], 2)
        cv2.putText(
            canvas,
            f"cls={int(box[5])} conf={box[4]:.3f}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[index % len(colors)],
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to save {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run optimized YOLO26 segmentation with TensorRT and CUDA Graph."
    )
    parser.add_argument("engine", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--preprocess",
        choices=["exact-cpu", "fast-gpu"],
        default="exact-cpu",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-masks", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-copy-iterations", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--save-overlay", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read {args.image}")
    pipeline = OptimizedSegmentationPipeline(
        args.engine,
        image.shape,
        args.preprocess,
        args.confidence,
        args.max_masks,
    )
    pipeline.prepare_frame(image)
    pipeline.capture(args.warmup)

    graph_gpu_ms = pipeline.benchmark_graph_gpu_ms(args.warmup, args.iterations)
    synchronized_samples: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        outputs = pipeline.infer_gpu(image, synchronize=True)
        synchronized_samples.append((time.perf_counter() - started) * 1_000)
    cpu_copy_latency, detection_count = cpu_result_copy_ms(
        pipeline,
        image,
        args.cpu_copy_iterations,
    )
    detections, valid, masks = outputs

    results = {
        "engine": str(args.engine),
        "image": str(args.image),
        "gpu": torch.cuda.get_device_name(0),
        "preprocess": args.preprocess,
        "confidence": args.confidence,
        "max_masks": args.max_masks,
        "detections": detection_count,
        "cuda_graph_gpu_ms": graph_gpu_ms,
        "cuda_graph_gpu_fps": 1_000.0 / graph_gpu_ms,
        "host_frame_to_gpu_results_latency_ms": latency_summary(
            synchronized_samples
        ),
        "host_frame_to_cpu_fixed_boxes_and_masks_latency_ms": cpu_copy_latency,
    }
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.save_overlay:
        save_overlay(
            args.save_overlay,
            image,
            pipeline.target_shape,
            detections,
            valid,
            masks,
        )


if __name__ == "__main__":
    main()
