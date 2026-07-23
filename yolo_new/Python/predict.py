# 评估训练好的模型
from project_paths import BOTTLE_DATA_ROOT, BOTTLE_RUNS_ROOT, YOLOV8N_RUN_ROOT

from ultralytics import YOLO  # type: ignore

if __name__ == "__main__":
    model = YOLO(YOLOV8N_RUN_ROOT / "weights" / "best.pt")
    model.predict(
        BOTTLE_DATA_ROOT / "test" / "images",
        save=True,
        imgsz=640,
        device=0,
        project=BOTTLE_RUNS_ROOT / "predictions",
        name="yolov8n_640",
    )


