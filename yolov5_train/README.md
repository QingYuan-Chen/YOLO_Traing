# 经典 YOLOv5 训练

本目录用于经典 `ultralytics/yolov5` 路线。它和通过 `ultralytics` Python 包加载的新版 YOLO 是两套不同入口。

## 1. 上游源码策略

完整 YOLOv5 源码位于本地 `yolov5_train/yolov5/`，并由 `.gitignore` 排除，不重复提交到本仓库。

初始化：

```powershell
powershell -ExecutionPolicy Bypass -File .\yolov5_train\setup_yolov5_env.ps1
```

初始化脚本为新克隆锁定经过当前流程验证的提交：

```text
70b964b6d5067fff621f724c85d0e39e6b4c8e4e
```

如果本地已经存在另一个提交，脚本只警告，不会强制切换或覆盖本地修改。

## 2. 训练

```powershell
powershell -ExecutionPolicy Bypass -File .\yolov5_train\train_yolov5.ps1
```

默认塑料瓶示例：

```text
imgsz   = 320
batch   = 16
epochs  = 200
workers = 0
device  = 0
name    = yolov5n_classic_320
```

默认输出：

```text
%YOLO_WORKSPACE_ROOT%\Training_runs\bottle\yolov5n_classic_320
```

脚本会先切换到 YOLOv5 源码目录再执行 `train.py`，避免 Windows 跨盘符相对路径错误。

可以覆盖参数：

```powershell
.\yolov5_train\train_yolov5.ps1 `
  -DataYaml 'D:\datasets\data.yaml' `
  -Weights 'D:\weights\yolov5n.pt' `
  -ProjectDir 'D:\runs' `
  -RunName 'experiment_001'
```

如果同名输出目录已经存在，脚本会停止。继续已有实验应使用对应 `last.pt` 和明确的续训流程。

## 3. 导出 ONNX

```powershell
.\yolov5_train\export_yolov5_onnx.ps1
```

该入口调用：

```text
Model_Conversion/export_yolov5_pt_to_onnx.ps1
```

K230 经典 YOLOv5 示例默认使用固定 batch、320 输入、opset 12 和传统 ONNX 导出器。转换前后都应检查实际输入输出张量。

## 4. 常见坑

- 不要从另一个盘符直接启动 `train.py`。
- 不要自动更新到最新 `master` 后继续声称环境可复现。
- YOLOv5 的输出和后处理不能根据 `.pt` 文件名判断。
- `workers=0` 是 Windows 上更稳妥的起点。
- 正式比较必须使用相同验证集和评估参数。

完整说明见：

- [训练指南](../docs/训练指南.md)
- [常见问题](../docs/常见问题.md)
- [模型转换](../Model_Conversion/README.md)
