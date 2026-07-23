import argparse
import os
import shutil
import tempfile
from pathlib import Path

workspace_root = Path(
    os.environ.get(
        "YOLO_WORKSPACE_ROOT",
        str(Path.home() / "Desktop" / "YOLOTraining"),
    )
).expanduser()
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(workspace_root / "Config" / "ultralytics"),
)

import onnx
import onnxruntime as ort
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an Ultralytics YOLO .pt model to a validated ONNX file."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve(strict=True)
    output_path = args.output.resolve()

    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Model must be a .pt file: {model_path}")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Output must end with .onnx: {output_path}")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="yolo-onnx-export-") as stage_root:
        stage_model = Path(stage_root) / model_path.name
        shutil.copy2(model_path, stage_model)

        exported = YOLO(stage_model).export(
            format="onnx",
            imgsz=args.imgsz,
            batch=1,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=False,
            device="cpu",
        )
        stage_onnx = Path(exported)

        model = onnx.load(stage_onnx)
        onnx.checker.check_model(model)
        session = ort.InferenceSession(
            str(stage_onnx), providers=["CPUExecutionProvider"]
        )
        print("IR", model.ir_version)
        print("OPSET", [(item.domain, item.version) for item in model.opset_import])
        print(
            "INPUTS",
            [(item.name, item.shape, item.type) for item in session.get_inputs()],
        )
        print(
            "OUTPUTS",
            [(item.name, item.shape, item.type) for item in session.get_outputs()],
        )

        shutil.copy2(stage_onnx, output_path)

    print(f"ONNX written to: {output_path}")


if __name__ == "__main__":
    main()
