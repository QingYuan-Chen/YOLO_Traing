# 导出 ONNX 模型
from project_paths import YOLOV8N_RUN_ROOT

from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO(YOLOV8N_RUN_ROOT / "weights" / "best.pt")
    model.export(format="onnx", simplify=True, imgsz=640, opset=12)
