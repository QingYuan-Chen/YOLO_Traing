from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from benchmark_yolo26_trt_native import NativeTensorRTRunner


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = reference.astype(np.float64, copy=False)
    candidate = candidate.astype(np.float64, copy=False)
    difference = candidate - reference
    ratio = np.maximum(reference / candidate, candidate / reference)
    return {
        "mae_m": float(np.mean(np.abs(difference))),
        "rmse_m": float(np.sqrt(np.mean(np.square(difference)))),
        "mean_abs_relative": float(np.mean(np.abs(difference) / reference)),
        "delta1_agreement": float(np.mean(ratio < 1.25)),
        "max_abs_m": float(np.max(np.abs(difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare YOLO26 depth PT, ONNX, and TensorRT raw outputs.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    letterboxed = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)(image=bgr)
    rgb = np.ascontiguousarray(letterboxed[..., ::-1].transpose(2, 0, 1))[None]
    input_array = rgb.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(input_array).cuda()

    model = YOLO(str(args.weights)).model.cuda().eval()
    head = model.model[-1]
    with torch.inference_mode():
        pt_output = model(input_tensor)
    if isinstance(pt_output, (tuple, list)):
        pt_output = pt_output[0]
    if pt_output.shape[-2:] != (args.imgsz, args.imgsz):
        pt_output = F.interpolate(pt_output, size=(args.imgsz, args.imgsz), mode="bilinear", align_corners=False)
    pt_array = pt_output.detach().float().cpu().numpy()

    ort_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_array = ort_session.run(None, {ort_session.get_inputs()[0].name: input_array})[0]

    runner = NativeTensorRTRunner.load(args.engine)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner.input.copy_(input_tensor)
        runner.execute(stream)
    stream.synchronize()
    engine_array = runner.tensors[runner.output_names[0]].detach().float().cpu().numpy()

    report = {
        "input_shape": list(input_array.shape),
        "output_shape": list(pt_array.shape),
        "cal_a": float(head.cal_a.detach().cpu()),
        "cal_b": float(head.cal_b.detach().cpu()),
        "pt": {
            "min_m": float(pt_array.min()),
            "max_m": float(pt_array.max()),
            "mean_m": float(pt_array.mean()),
        },
        "onnx": {
            "min_m": float(onnx_array.min()),
            "max_m": float(onnx_array.max()),
            "mean_m": float(onnx_array.mean()),
            "versus_pt": metrics(pt_array, onnx_array),
        },
        "engine": {
            "min_m": float(engine_array.min()),
            "max_m": float(engine_array.max()),
            "mean_m": float(engine_array.mean()),
            "versus_pt": metrics(pt_array, engine_array),
            "versus_onnx": metrics(onnx_array, engine_array),
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
