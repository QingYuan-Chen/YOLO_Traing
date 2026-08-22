from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt
import torch

from benchmark_yolo26_trt_native import NativeTensorRTRunner, TRT_TO_TORCH


@dataclass
class ExecutionSlot:
    context: trt.IExecutionContext
    tensors: dict[str, torch.Tensor]
    stream: torch.cuda.Stream

    def execute(self) -> None:
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")


def create_slot(engine: trt.ICudaEngine) -> ExecutionSlot:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    tensors: dict[str, torch.Tensor] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = tuple(engine.get_tensor_shape(name))
            context.set_input_shape(name, shape)

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = TRT_TO_TORCH[engine.get_tensor_dtype(name)]
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
        tensors[name] = tensor
        if not context.set_tensor_address(name, tensor.data_ptr()):
            raise RuntimeError(f"Failed to bind {name}")
    return ExecutionSlot(context=context, tensors=tensors, stream=torch.cuda.Stream())


def capture_slot(slot: ExecutionSlot, warmup: int) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(slot.stream):
        for _ in range(warmup):
            slot.execute()
    slot.stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=slot.stream):
        slot.execute()
    slot.stream.synchronize()
    return graph


def measure_concurrent(
    slots: list[ExecutionSlot],
    operations: list[object],
    iterations: int,
) -> tuple[float, float]:
    control_stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    finished = [torch.cuda.Event() for _ in slots]

    with torch.cuda.stream(control_stream):
        start.record(control_stream)
    for slot, operation, done in zip(slots, operations, finished):
        with torch.cuda.stream(slot.stream):
            slot.stream.wait_event(start)
            for _ in range(iterations):
                operation()
            done.record(slot.stream)
    with torch.cuda.stream(control_stream):
        for done in finished:
            control_stream.wait_event(done)
        stop.record(control_stream)
    stop.synchronize()

    elapsed_ms = start.elapsed_time(stop)
    images = len(slots) * iterations
    return elapsed_ms / images, images * 1_000.0 / elapsed_ms


def benchmark_level(
    engine: trt.ICudaEngine,
    concurrency: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    memory_before = torch.cuda.mem_get_info()
    slots = [create_slot(engine) for _ in range(concurrency)]
    for slot in slots:
        slot.tensors[engine.get_tensor_name(0)].normal_()

    for slot in slots:
        with torch.cuda.stream(slot.stream):
            for _ in range(warmup):
                slot.execute()
    for slot in slots:
        slot.stream.synchronize()
    native_ms, native_fps = measure_concurrent(
        slots, [slot.execute for slot in slots], iterations
    )

    graphs = [capture_slot(slot, warmup=5) for slot in slots]
    graph_ms, graph_fps = measure_concurrent(
        slots, [graph.replay for graph in graphs], iterations
    )
    memory_after = torch.cuda.mem_get_info()
    result = {
        "concurrency": concurrency,
        "native_effective_ms_per_image": native_ms,
        "native_total_fps": native_fps,
        "cuda_graph_effective_ms_per_image": graph_ms,
        "cuda_graph_total_fps": graph_fps,
        "free_vram_before_bytes": memory_before[0],
        "free_vram_after_bytes": memory_after[0],
        "context_vram_delta_bytes": memory_before[0] - memory_after[0],
    }

    del graphs
    del slots
    gc.collect()
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure YOLO26 TensorRT multi-context concurrency throughput."
    )
    parser.add_argument("engine", type=Path)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    runner = NativeTensorRTRunner.load(args.engine)
    results = {
        "engine": str(args.engine),
        "gpu": torch.cuda.get_device_name(0),
        "results": [
            benchmark_level(runner.engine, level, args.warmup, args.iterations)
            for level in args.levels
        ],
    }
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
