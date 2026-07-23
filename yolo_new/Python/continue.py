# 定制自己的模型，推荐使用5,8,11,12
from project_paths import YOLOV8N_RUN_ROOT

from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO(YOLOV8N_RUN_ROOT / "weights" / "last.pt")
    model.train(resume=True)
