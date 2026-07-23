# 评估训练好的模型
from project_paths import BOTTLE_DATA_YAML, YOLOV8N_RUN_ROOT

from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO(YOLOV8N_RUN_ROOT / "weights" / "best.pt")
    model.val(data=BOTTLE_DATA_YAML, split="test", imgsz=640, device=0)
