from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENGINE_DIR = PACKAGE_ROOT / "models" / "tensorrt"
DEFAULT_IMAGE = PACKAGE_ROOT / "sample_data" / "bus.jpg"


TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.uint8: torch.uint8,
    trt.bool: torch.bool,
}


@dataclass
class NativeTensorRTRunner:
    engine_path: Path
    logger: trt.Logger
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    context: trt.IExecutionContext
    tensors: dict[str, torch.Tensor]
    input_name: str
    output_names: list[str]
    metadata: dict[str, object]

    @classmethod
    def load(cls, engine_path: Path) -> "NativeTensorRTRunner":
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        plan, metadata = read_ultralytics_engine(engine_path)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize {engine_path}")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"TensorRT failed to create a context for {engine_path}")

        tensors: dict[str, torch.Tensor] = {}
        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
                shape = tuple(engine.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise ValueError(f"Dynamic input is not supported by this benchmark: {shape}")
                context.set_input_shape(name, shape)

        if len(input_names) != 1:
            raise ValueError(f"Expected one input, found {input_names}")

        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            shape = tuple(context.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"Unresolved shape for {name}: {shape}")
            dtype = TRT_TO_TORCH.get(engine.get_tensor_dtype(name))
            if dtype is None:
                raise TypeError(f"Unsupported TensorRT dtype for {name}")
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            tensors[name] = tensor
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Failed to bind TensorRT tensor {name}")

        return cls(
            engine_path=engine_path,
            logger=logger,
            runtime=runtime,
            engine=engine,
            context=context,
            tensors=tensors,
            input_name=input_names[0],
            output_names=output_names,
            metadata=metadata,
        )

    @property
    def input(self) -> torch.Tensor:
        return self.tensors[self.input_name]

    def execute(self, stream: torch.cuda.Stream) -> None:
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path.name}")


def read_ultralytics_engine(path: Path) -> tuple[bytes, dict[str, object]]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"Engine is too small: {path}")

    metadata_length = int.from_bytes(payload[:4], byteorder="little", signed=False)
    if not 0 < metadata_length < min(len(payload) - 4, 1_000_000):
        return payload, {}
    try:
        metadata = json.loads(payload[4 : 4 + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return payload, {}
    return payload[4 + metadata_length :], metadata


def load_pinned_bgr(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = np.ascontiguousarray(image)
    pinned = torch.empty(image.shape, dtype=torch.uint8, pin_memory=True)
    pinned.copy_(torch.from_numpy(image))
    return pinned


def gpu_letterbox_bgr_to_rgb(
    source_cpu: torch.Tensor,
    source_gpu: torch.Tensor,
    destination: torch.Tensor,
) -> None:
    if destination.ndim != 4 or destination.shape[0] != 1 or destination.shape[1] != 3:
        raise ValueError(f"Expected NCHW batch-1 destination, got {destination.shape}")

    source_gpu.copy_(source_cpu, non_blocking=True)
    source_height, source_width = source_gpu.shape[:2]
    target_height, target_width = destination.shape[2:]
    scale = min(target_height / source_height, target_width / source_width)
    resized_height = round(source_height * scale)
    resized_width = round(source_width * scale)
    top = round((target_height - resized_height) / 2 - 0.1)
    left = round((target_width - resized_width) / 2 - 0.1)

    chw_bgr = source_gpu.permute(2, 0, 1).unsqueeze(0).to(destination.dtype)
    resized_bgr = F.interpolate(
        chw_bgr,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    destination.fill_(114.0 / 255.0)
    view = destination[:, :, top : top + resized_height, left : left + resized_width]
    scale_value = 1.0 / 255.0
    view[:, 0].copy_(resized_bgr[:, 2].mul(scale_value))
    view[:, 1].copy_(resized_bgr[:, 1].mul(scale_value))
    view[:, 2].copy_(resized_bgr[:, 0].mul(scale_value))


def elapsed_gpu_ms(
    operation: Callable[[], object],
    stream: torch.cuda.Stream,
    warmup: int,
    iterations: int,
) -> float:
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            operation()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iterations):
            operation()
        end.record(stream)
        end.synchronize()
    return start.elapsed_time(end) / iterations


def synchronized_latency_ms(
    operation: Callable[[], object],
    stream: torch.cuda.Stream,
    iterations: int,
) -> tuple[float, float, float]:
    samples: list[float] = []
    with torch.cuda.stream(stream):
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            stream.synchronize()
            samples.append((time.perf_counter() - started) * 1_000)
    samples.sort()
    p95_index = min(len(samples) - 1, int(len(samples) * 0.95))
    return statistics.median(samples), samples[p95_index], statistics.mean(samples)


def capture_cuda_graph(
    operation: Callable[[], object],
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, object]:
    with torch.cuda.stream(stream):
        operation()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = operation()
    stream.synchronize()
    return graph, static_outputs


def segment_postprocess_static(
    predictions: torch.Tensor,
    prototypes: torch.Tensor,
    confidence: float,
    max_masks: int,
    target_shape: tuple[int, int],
    upsample: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    detections = predictions[0, :max_masks]
    valid = detections[:, 4] > confidence
    coefficients = detections[:, 6:].float()
    proto = prototypes[0].float()
    channels, mask_height, mask_width = proto.shape
    masks = (coefficients @ proto.view(channels, -1)).view(
        -1, mask_height, mask_width
    )

    target_height, target_width = target_shape
    boxes = detections[:, :4].float()
    x1 = boxes[:, 0, None, None] * (mask_width / target_width)
    y1 = boxes[:, 1, None, None] * (mask_height / target_height)
    x2 = boxes[:, 2, None, None] * (mask_width / target_width)
    y2 = boxes[:, 3, None, None] * (mask_height / target_height)
    columns = torch.arange(mask_width, device=masks.device, dtype=boxes.dtype)[
        None, None, :
    ]
    rows = torch.arange(mask_height, device=masks.device, dtype=boxes.dtype)[
        None, :, None
    ]
    masks = masks * (
        (columns >= x1) * (columns < x2) * (rows >= y1) * (rows < y2)
    )
    masks = masks * valid[:, None, None]
    if upsample:
        masks = F.interpolate(
            masks[None], target_shape, mode="bilinear", align_corners=False
        )[0]
    return detections[:, :6], valid, masks.gt_(0.0).to(torch.uint8)


def summarize_outputs(runner: NativeTensorRTRunner) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in runner.output_names:
        tensor = runner.tensors[name]
        item: dict[str, object] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
        if tensor.ndim == 3 and tensor.shape[-1] == 38:
            scores = tensor[0, :, 4]
            item["scores_ge_0.25"] = int((scores >= 0.25).sum().item())
            item["top_scores"] = [
                round(float(value), 6) for value in scores.topk(k=5).values.tolist()
            ]
        summary[name] = item
    return summary


def benchmark_engine(
    engine_path: Path,
    image_path: Path,
    warmup: int,
    iterations: int,
    latency_iterations: int,
    confidence: float,
    max_masks: int,
) -> dict[str, object]:
    runner = NativeTensorRTRunner.load(engine_path)
    stream = torch.cuda.Stream()
    source_cpu = load_pinned_bgr(image_path)
    source_gpu = torch.empty_like(source_cpu, device="cuda")

    with torch.cuda.stream(stream):
        gpu_letterbox_bgr_to_rgb(source_cpu, source_gpu, runner.input)
        runner.execute(stream)
    stream.synchronize()

    inference_ms = elapsed_gpu_ms(
        lambda: runner.execute(stream), stream, warmup, iterations
    )

    def preprocess_and_infer() -> None:
        gpu_letterbox_bgr_to_rgb(source_cpu, source_gpu, runner.input)
        runner.execute(stream)

    pipeline_ms = elapsed_gpu_ms(preprocess_and_infer, stream, warmup, iterations)
    median_ms, p95_ms, mean_ms = synchronized_latency_ms(
        preprocess_and_infer, stream, latency_iterations
    )

    output0 = runner.tensors[runner.output_names[0]]
    output1 = runner.tensors[runner.output_names[1]]
    target_shape = tuple(runner.input.shape[2:])

    def postprocess_low_resolution() -> object:
        return segment_postprocess_static(
            output0,
            output1,
            confidence,
            max_masks,
            target_shape,
            upsample=False,
        )

    def postprocess_full_resolution() -> object:
        return segment_postprocess_static(
            output0,
            output1,
            confidence,
            max_masks,
            target_shape,
            upsample=True,
        )

    low_resolution_postprocess_ms = elapsed_gpu_ms(
        postprocess_low_resolution, stream, warmup, iterations
    )
    full_resolution_postprocess_ms = elapsed_gpu_ms(
        postprocess_full_resolution, stream, warmup, iterations
    )

    def full_pipeline() -> object:
        preprocess_and_infer()
        return postprocess_full_resolution()

    full_pipeline_ms = elapsed_gpu_ms(full_pipeline, stream, warmup, iterations)
    full_median_ms, full_p95_ms, full_mean_ms = synchronized_latency_ms(
        full_pipeline, stream, latency_iterations
    )

    cuda_graph_metrics: dict[str, object] = {}
    try:
        inference_graph, inference_graph_outputs = capture_cuda_graph(
            lambda: runner.execute(stream), stream
        )
        inference_graph_ms = elapsed_gpu_ms(
            inference_graph.replay, stream, warmup, iterations
        )
        pipeline_graph, pipeline_graph_outputs = capture_cuda_graph(
            preprocess_and_infer, stream
        )
        pipeline_graph_ms = elapsed_gpu_ms(
            pipeline_graph.replay, stream, warmup, iterations
        )
        full_graph, full_graph_outputs = capture_cuda_graph(full_pipeline, stream)
        full_graph_ms = elapsed_gpu_ms(full_graph.replay, stream, warmup, iterations)
        graph_median_ms, graph_p95_ms, graph_mean_ms = synchronized_latency_ms(
            full_graph.replay, stream, latency_iterations
        )
        cuda_graph_metrics = {
            "inference_gpu_ms": inference_graph_ms,
            "preprocess_plus_inference_gpu_ms": pipeline_graph_ms,
            "full_segmentation_pipeline_gpu_ms": full_graph_ms,
            "synchronized_full_pipeline_latency_ms": {
                "median": graph_median_ms,
                "p95": graph_p95_ms,
                "mean": graph_mean_ms,
            },
        }
        _ = (
            inference_graph_outputs,
            pipeline_graph_outputs,
            full_graph_outputs,
        )
    except RuntimeError as error:
        cuda_graph_metrics = {"error": str(error)}

    with torch.cuda.stream(stream):
        preprocess_and_infer()
    stream.synchronize()

    return {
        "engine": str(engine_path),
        "metadata": runner.metadata,
        "input": {
            "name": runner.input_name,
            "shape": list(runner.input.shape),
            "dtype": str(runner.input.dtype),
        },
        "native_inference_gpu_ms": inference_ms,
        "native_inference_fps": 1_000.0 / inference_ms,
        "gpu_preprocess_plus_inference_ms": pipeline_ms,
        "gpu_preprocess_plus_inference_fps": 1_000.0 / pipeline_ms,
        "synchronized_pipeline_latency_ms": {
            "median": median_ms,
            "p95": p95_ms,
            "mean": mean_ms,
        },
        "segmentation_postprocess": {
            "confidence": confidence,
            "max_masks": max_masks,
            "prototype_resolution_gpu_ms": low_resolution_postprocess_ms,
            "full_640_resolution_gpu_ms": full_resolution_postprocess_ms,
        },
        "full_segmentation_pipeline_gpu_ms": full_pipeline_ms,
        "full_segmentation_pipeline_fps": 1_000.0 / full_pipeline_ms,
        "synchronized_full_pipeline_latency_ms": {
            "median": full_median_ms,
            "p95": full_p95_ms,
            "mean": full_mean_ms,
        },
        "cuda_graph": cuda_graph_metrics,
        "outputs": summarize_outputs(runner),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO26 TensorRT engines with a native persistent context."
    )
    parser.add_argument("engines", nargs="*", type=Path)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-masks", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engines = args.engines or sorted(DEFAULT_ENGINE_DIR.glob("*.engine"))
    if not engines:
        raise FileNotFoundError(f"No TensorRT engines found under {DEFAULT_ENGINE_DIR}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.backends.cudnn.benchmark = True
    results = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "tensorrt": trt.__version__,
            "image": str(args.image),
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "results": [
            benchmark_engine(
                engine_path,
                args.image,
                args.warmup,
                args.iterations,
                args.latency_iterations,
                args.confidence,
                args.max_masks,
            )
            for engine_path in engines
        ],
    }
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
