# YOLO 模型转换

本目录只保存转换脚本和说明，不保存 `.pt`、`.onnx`、`.kmodel`、`.hef` 等模型文件。

文档中的绝对路径来自一个已验证的 Windows 示例环境。其他机器可以先设置：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

转换脚本接受显式输入和输出路径；不要依赖用户名或盘符恰好相同。

最小 Python 依赖：

```powershell
pip install -r .\Model_Conversion\requirements-conversion.txt
```

模型产物统一放在：

```text
C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion
```

## 目录

```text
Model_Conversion/
├─ README.md
├─ export_pt_to_onnx.ps1          # 新版 Ultralytics YOLO，PowerShell
├─ export_pt_to_onnx.py           # 新版 Ultralytics YOLO，Python
├─ export_yolov5_pt_to_onnx.ps1   # 经典 YOLOv5，PowerShell
├─ K230/
│  ├─ README.md
│  └─ compile_onnx_to_kmodel.ps1
└─ Hailo/
   ├─ README.md
   ├─ prepare_hailo_workspace.ps1
   └─ compile_yolov8n_hailo8l.ps1
```

## 一、PT 转 ONNX

### 1. 新版 Ultralytics YOLO：PowerShell 命令

适用于 YOLOv8、YOLO11、YOLO12、YOLO26 等由 `ultralytics` 包加载的模型。

```powershell
$env:YOLO_CONFIG_DIR = 'C:\Users\MECHREU\Desktop\YOLOTraining\Config\ultralytics'

& 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe' export `
  model='C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov8n_640\weights\best.pt' `
  format=onnx `
  imgsz=640 `
  batch=1 `
  opset=17 `
  simplify=True `
  dynamic=False `
  device=cpu
```

也可以运行本目录的安全封装脚本。脚本会在临时目录导出，不会在训练目录旁边留下中间 ONNX：

```powershell
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov8n_640\weights\best.pt' `
  -OutputPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov8n.onnx' `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify
```

### 2. 新版 Ultralytics YOLO：Python 命令

```python
import shutil
from pathlib import Path

from ultralytics import YOLO

model = YOLO(
    r"C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov8n_640\weights\best.pt"
)
exported = Path(model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    opset=17,
    simplify=True,
    dynamic=False,
    device="cpu",
))
output = Path(
    r"C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov8n.onnx"
)
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    raise FileExistsError(output)
shutil.move(exported, output)
print(output)
```

参数化脚本用法：

```powershell
& 'E:\Anaconda_envs\envs\yolo\python.exe' `
  '.\Model_Conversion\export_pt_to_onnx.py' `
  --model 'C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov8n_640\weights\best.pt' `
  --output 'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov8n.onnx' `
  --imgsz 640 `
  --opset 17 `
  --simplify
```

### 3. 经典 YOLOv5：PowerShell 命令

经典 YOLOv5 应使用仓库自己的 `export.py`，不要直接套新版 Ultralytics 导出入口。

```powershell
conda activate yolov5gpu128
Set-Location 'E:\YOLO\yolov5_train\yolov5'

python export.py `
  --weights 'C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov5n_classic_320\weights\best.pt' `
  --img 320 `
  --batch-size 1 `
  --include onnx `
  --opset 12 `
  --device cpu
```

K230 使用的经典 YOLOv5 ONNX 建议运行封装脚本，它会强制传统导出器 `dynamo=False`：

```powershell
.\Model_Conversion\export_yolov5_pt_to_onnx.ps1 `
  -ModelPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Training_runs\bottle\yolov5n_classic_320\weights\best.pt' `
  -OutputPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov5n_bottle_320_k230.onnx' `
  -ImageSize 320 `
  -Opset 12
```

## 二、目标平台差异

| 目标 | 推荐 ONNX | 量化数据 | 最终格式 |
|---|---|---|---|
| PC / ONNX Runtime | 静态输入，常用 opset 12 或 17 | 不需要 | `.onnx` |
| K230 / nncase 2.9 | 静态 NCHW、batch 1、opset 12；经典 YOLOv5 不简化 | 10 张以上代表性图片 | `.kmodel` |
| Hailo-8L | 静态输入；当前 YOLOv8n 640 使用 opset 17 | 当前成功流程使用 1024 张图片 | `.hef` |

不要只修改 ONNX 的 `ir_version` 来解决兼容问题。K230 的旧 ONNX Runtime 同时受 opset 限制；应从 PT 重新导出 opset 12。

## 三、后续步骤

- ONNX 转 K230 KModel：见 [K230/README.md](K230/README.md)。
- ONNX 转 Hailo-8L HEF：见 [Hailo/README.md](Hailo/README.md)。

### YOLOv8n 转 Hailo-8L 快速流程

先将 `best.pt` 导出为静态 640 ONNX：

```powershell
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath '<训练结果>\weights\best.pt' `
  -OutputPath '<转换目录>\yolov8n.onnx' `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify
```

准备容器工作区和 1024 张校准图片：

```powershell
.\Model_Conversion\Hailo\prepare_hailo_workspace.ps1 `
  -OnnxPath '<转换目录>\yolov8n.onnx' `
  -CalibrationImages '<数据集>\valid\images' `
  -WorkName 'yolov8n-bottle-640' `
  -CalibrationCount 1024
```

先预检，再开始耗时编译：

```powershell
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath '<转换目录>\yolov8n_bottle_640_hailo8l.hef' `
  -Classes 1 `
  -PreflightOnly

.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath '<转换目录>\yolov8n_bottle_640_hailo8l.hef' `
  -Classes 1
```

标准 YOLOv8n 的六个卷积端节点已写入编译脚本。模型结构或导出版本变化时必须重新检查节点，不能直接复用。

## 四、已验证参考产物

```text
K230:
yolov5n_bottle_320_k230.kmodel
SHA-256 36F50EC97A9E1EF2AFD8BB0625CC6E1E8B6C388BD706F74F04A02385EA00920A

Hailo-8L:
yolov8n_bottle_640_hailo8l.hef
SHA-256 339083786AB96041C454C0AE17B3B82C24EB492D03417A59388E1AA13B4D1CD2
```
