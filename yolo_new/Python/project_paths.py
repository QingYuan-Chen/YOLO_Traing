"""Shared paths for the Ultralytics training examples.

Set YOLO_WORKSPACE_ROOT to keep datasets and generated artifacts somewhere
other than the default Desktop/YOLOTraining directory.
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(
    os.environ.get(
        "YOLO_WORKSPACE_ROOT",
        str(Path.home() / "Desktop" / "YOLOTraining"),
    )
).expanduser()

CONFIG_ROOT = WORKSPACE_ROOT / "Config" / "ultralytics"
BOTTLE_DATA_ROOT = (
    WORKSPACE_ROOT
    / "Datasets"
    / "bottle"
    / "Plastic Bottle 2.0.v39i.yolov8"
)
BOTTLE_DATA_YAML = WORKSPACE_ROOT / "Datasets" / "bottle" / "bottle_plastic.yaml"
BOTTLE_RUNS_ROOT = WORKSPACE_ROOT / "Training_runs" / "bottle"
YOLOV8N_PRETRAINED = (
    REPO_ROOT / "Model_Traning" / "weights" / "yolov8" / "yolov8n.pt"
)
YOLOV8N_RUN_ROOT = BOTTLE_RUNS_ROOT / "yolov8n_640"

os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_ROOT))
