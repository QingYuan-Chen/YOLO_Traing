# 预训练权重与离线资源

`Model_Traning` 是仓库已有目录名，暂时保留该拼写以兼容现有脚本。

本目录保存：

- 预训练权重的本地目录结构。
- 离线依赖版本清单。
- GitHub Release 资产打包和下载脚本。

数据集、训练结果、转换模型和 wheel 默认保存在仓库外的 `YOLOTraining` 工作目录。

## 预训练权重

写模型名称时，Ultralytics 通常会在本地缺失时联网下载：

```python
model = YOLO("yolov8n.pt")
```

写本地路径时，文件必须真实存在：

```python
model = YOLO(r"E:\YOLO\Model_Traning\weights\yolov8\yolov8n.pt")
```

经典 YOLOv5：

```powershell
python train.py --weights "E:\YOLO\Model_Traning\weights\yolov5\yolov5n.pt"
```

预训练权重只是训练起点。删除预训练权重不会删除已经得到的 `best.pt`，但引用该文件的后续训练会失败。

## 外部工作目录

默认位置：

```text
%USERPROFILE%\Desktop\YOLOTraining
```

自定义：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

离线 wheels 位于：

```text
%YOLO_WORKSPACE_ROOT%\Dependencies\wheels
```

## Release 资产方案

打包权重：

```powershell
.\Model_Traning\prepare_release_assets.ps1 weights
```

打包 wheels：

```powershell
.\Model_Traning\prepare_release_assets.ps1 wheels
```

计划使用的 Release tags：

```text
weights-pretrained
wheels-cu128
```

当前 GitHub 仓库尚未发布正式 Release。因此下面的下载命令只有在对应 tag 和资产上传后才会可用：

```powershell
.\Model_Traning\download_assets.ps1 weights
.\Model_Traning\download_assets.ps1 wheels
```

不要把“脚本存在”写成“下载服务已经可用”。发布后还应记录资产文件名、大小和 SHA-256。

## 依赖清单

`requirements.txt` 是已有 Windows/CUDA 环境的离线依赖快照，不代表所有机器都应无条件安装全部包。

模型转换的最小依赖另见：

```text
Model_Conversion/requirements-conversion.txt
```

环境复现时应同时记录：

- Python 版本。
- PyTorch 和 CUDA wheel。
- Ultralytics 或 YOLOv5 提交。
- ONNX 和 ONNX Runtime 版本。
- GPU 型号与支持的计算能力。
