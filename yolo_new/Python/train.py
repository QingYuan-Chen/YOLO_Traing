# 定制自己的新版 YOLO 模型，例如 YOLOv8、YOLO11、YOLO12
from project_paths import BOTTLE_DATA_YAML, BOTTLE_RUNS_ROOT, YOLOV8N_PRETRAINED

from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO(YOLOV8N_PRETRAINED)
    model.train(
        data=BOTTLE_DATA_YAML,
        epochs=300,
        batch=32,
        workers=0,
        imgsz=640,
        device=0,
        project=BOTTLE_RUNS_ROOT,
        name="yolov8n_640_python",
    )

