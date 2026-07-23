# Ultralytics 新版 YOLO 训练示例

本目录演示 YOLOv8、YOLO11、YOLO12、YOLO26 等新版 Ultralytics 模型的训练、验证、预测、断点续训和导出。

## 目录

```text
yolo_new/
├─ Powershell/        # 命令行封装
├─ Python/            # Python API 示例
└─ data.yaml          # 塑料瓶示例数据集配置
```

## 工作目录

脚本默认使用：

```text
%USERPROFILE%\Desktop\YOLOTraining
```

可以在当前 PowerShell 会话覆盖：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

## 两类训练入口

创建或补齐已锁定版本的示例环境：

```powershell
powershell -ExecutionPolicy Bypass -File .\yolo_new\Powershell\setup_yolo_new_env.ps1
```

该脚本默认使用 Python 3.12、CUDA 12.8 PyTorch、Ultralytics 8.4.60，并在结束时执行导入检查。

### 已验证入口

仓库根目录的 `start_yolov8n_640.ps1` 保存已完成塑料瓶 YOLOv8n 640 训练所使用的参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_yolov8n_640.ps1
```

默认输出名为 `yolov8n_640`。脚本发现同名输出目录时会停止，避免混入或覆盖已有训练证据。

### 教学实验入口

```powershell
powershell -ExecutionPolicy Bypass -File .\yolo_new\Powershell\train_yolo_new.ps1
```

默认使用单独的 `yolov8n_640_manual` run name，便于修改 batch、epochs、模型或数据集而不污染已验证结果。

Python API 示例：

```powershell
& 'E:\Anaconda_envs\envs\yolo\python.exe' .\yolo_new\Python\train.py
```

Python 示例通过 `Python/project_paths.py` 解析仓库路径和外部工作目录。

## 验证

```powershell
powershell -ExecutionPolicy Bypass -File .\yolo_new\Powershell\val_yolo_new.ps1
```

覆盖默认权重或数据集：

```powershell
.\yolo_new\Powershell\val_yolo_new.ps1 `
  -Weights 'D:\runs\weights\best.pt' `
  -DataYaml 'D:\datasets\data.yaml' `
  -ImageSize 640 `
  -Split test
```

## 预测

```powershell
.\yolo_new\Powershell\predict_yolo_new.ps1 `
  -Weights 'D:\runs\weights\best.pt' `
  -Source 'D:\images'
```

## 断点续训

```powershell
.\yolo_new\Powershell\resume_yolo_new.ps1 `
  -LastWeights 'D:\runs\weights\last.pt'
```

只对同一次实验的 `last.pt` 使用断点续训。

## 导出 ONNX

推荐使用统一转换入口：

```powershell
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath 'D:\runs\weights\best.pt' `
  -OutputPath 'D:\models\model.onnx' `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify
```

兼容入口：

```powershell
.\yolo_new\Powershell\export_yolo_new_onnx.ps1
```

转换脚本会在临时目录导出，并使用 ONNX checker 和 ONNX Runtime 验证后再复制到目标位置。

## 注意

- 新版 Ultralytics 模型不能直接套用经典 YOLOv5 的板端后处理。
- 文件名不能证明模型代际，应检查模型结构和输入输出。
- 正式实验应显式记录模型、数据集、`imgsz`、batch、epochs 和 run name。
- 不要同时启动两个任务争抢同一张 GPU。
