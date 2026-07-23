# YOLOv8n 转 Hailo-8L HEF

本文说明如何把 Ultralytics YOLOv8n 检测模型从 PyTorch `best.pt` 转换成 Hailo-8L 可以加载的 `.hef`。

当前仓库保存的已验证示例是：

```text
YOLOv8n
输入尺寸：640 x 640
类别数量：1
目标架构：hailo8l
校准图片：1024 张
最终文件：yolov8n_bottle_640_hailo8l.hef
```

## 转换链路

```text
best.pt
  │ Ultralytics export
  ▼
静态 yolov8n.onnx
  │ Hailo Model Zoo parse + optimize + quantize + compile
  ▼
yolov8n_bottle_640_hailo8l.hef
```

`.hef` 是 Hailo Executable Format。它不是通用 ONNX，而是针对指定 Hailo 硬件架构完成优化、量化和编译后的设备模型。

## 1. 前置条件

需要：

- Windows PowerShell。
- Docker Desktop 正常运行。
- 已有 Docker 容器 `hailo-suite`。
- 容器内存在 Hailo Model Zoo 和虚拟环境：

```text
/local/workspace/hailo_virtualenv
```

- YOLOv8n 训练权重 `best.pt`。
- 有代表性的校准图片。

检查 Docker：

```powershell
docker info
docker ps -a --filter 'name=hailo-suite'
```

如果容器名不同，可通过脚本的 `-Container` 参数指定。

## 2. 设置工作目录

仓库脚本默认使用：

```text
%USERPROFILE%\Desktop\YOLOTraining
```

也可以覆盖：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

下面以当前塑料瓶示例为例：

```powershell
$WorkspaceRoot = if ($env:YOLO_WORKSPACE_ROOT) {
    $env:YOLO_WORKSPACE_ROOT
} else {
    Join-Path $env:USERPROFILE 'Desktop\YOLOTraining'
}

$BestPt = Join-Path $WorkspaceRoot 'Training_runs\bottle\yolov8n_640\weights\best.pt'
$OnnxPath = Join-Path $WorkspaceRoot 'Module_conversion\bottle\yolov8n.onnx'
$CalibrationImages = Join-Path $WorkspaceRoot 'Datasets\bottle\Plastic Bottle 2.0.v39i.yolov8\valid\images'
$HefPath = Join-Path $WorkspaceRoot 'Module_conversion\bottle\yolov8n_bottle_640_hailo8l.hef'
```

## 3. 从 YOLOv8n PT 导出静态 ONNX

从仓库根目录执行：

```powershell
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath $BestPt `
  -OutputPath $OnnxPath `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify `
  -WorkspaceRoot $WorkspaceRoot
```

当前 Hailo-8L 成功流程使用：

```text
batch       = 1
input       = [1, 3, 640, 640]
opset       = 17
dynamic     = False
simplify    = True
device      = CPU
```

导出脚本会：

1. 把 `.pt` 复制到临时目录。
2. 使用 Ultralytics 导出 ONNX。
3. 运行 `onnx.checker.check_model`。
4. 使用 ONNX Runtime 创建 CPU 推理会话。
5. 验证成功后才把 ONNX 复制到目标路径。
6. 如果目标文件已经存在则停止，避免覆盖参考产物。

## 4. 检查 ONNX

```powershell
& 'E:\Anaconda_envs\envs\yolo\python.exe' -c @'
import onnx
import onnxruntime as ort
import sys

path = sys.argv[1]
model = onnx.load(path)
onnx.checker.check_model(model)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

print("IR", model.ir_version)
print("OPSET", [(item.domain, item.version) for item in model.opset_import])
print("INPUTS", [(item.name, item.shape, item.type) for item in session.get_inputs()])
print("OUTPUTS", [(item.name, item.shape, item.type) for item in session.get_outputs()])
'@ $OnnxPath
```

至少确认：

- 输入是静态 `[1, 3, 640, 640]`。
- ONNX checker 通过。
- ONNX Runtime 能创建会话。
- 模型确实是标准 Ultralytics YOLOv8n 检测结构。

文件名叫 `yolov8n.onnx` 不能代替结构检查。

## 5. 准备 Hailo 校准工作区

```powershell
.\Model_Conversion\Hailo\prepare_hailo_workspace.ps1 `
  -OnnxPath $OnnxPath `
  -CalibrationImages $CalibrationImages `
  -WorkName 'yolov8n-bottle-640' `
  -CalibrationCount 1024 `
  -Container 'hailo-suite'
```

准备脚本会：

- 检查 Docker 引擎和目标容器。
- 在图片目录中按顺序等间隔抽取最多 1024 张图片。
- 把图片复制到临时 staging 目录并统一命名。
- 清理容器内同名工作区。
- 在容器内创建：

```text
/home/hailo/model-conversion/yolov8n-bottle-640
├─ yolov8n.onnx
└─ calib_images/
```

- 将目录所有权设置给容器内 `hailo` 用户。
- 核对容器中的校准图片数量。

注意：该脚本会删除容器内同名 `WorkName` 工作区，但不会删除 Windows 上的 ONNX、训练权重或校准图片。

校准图片应覆盖真实部署场景中的：

- 光照变化。
- 目标大小和位置变化。
- 遮挡与背景变化。
- 摄像头实际颜色和曝光特征。

图片数量多不等于校准质量高。当前 1024 张只是已成功流程使用的数量。

## 6. 先执行编译预检

预检只检查容器工作区、ONNX、校准图片和 Docker 工作目录，不进行耗时编译：

```powershell
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath $HefPath `
  -Classes 1 `
  -Container 'hailo-suite' `
  -PreflightOnly
```

看到下面的信息才表示预检通过：

```text
Hailo 编译预检通过，未启动编译。
```

## 7. 编译为 Hailo-8L HEF

去掉 `-PreflightOnly`：

```powershell
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath $HefPath `
  -Classes 1 `
  -Container 'hailo-suite'
```

脚本在容器中执行的核心命令等价于：

```bash
source /local/workspace/hailo_virtualenv/bin/activate

hailomz compile yolov8n \
  --ckpt /home/hailo/model-conversion/yolov8n-bottle-640/yolov8n.onnx \
  --calib-path /home/hailo/model-conversion/yolov8n-bottle-640/calib_images \
  --hw-arch hailo8l \
  --classes 1 \
  --performance \
  --end-node-names \
    /model.22/cv2.0/cv2.0.2/Conv \
    /model.22/cv3.0/cv3.0.2/Conv \
    /model.22/cv2.1/cv2.1.2/Conv \
    /model.22/cv3.1/cv3.1.2/Conv \
    /model.22/cv2.2/cv2.2.2/Conv \
    /model.22/cv3.2/cv3.2.2/Conv
```

编译日志保存在容器工作区：

```text
/home/hailo/model-conversion/yolov8n-bottle-640/yolov8n_compile.log
```

脚本成功后会：

1. 将生成的 HEF 重命名为带 `_hailo8l` 的文件。
2. 从容器复制到 Windows 的 `$HefPath`。
3. 分别计算容器和 Windows 文件的 SHA-256。
4. 两边哈希不一致时直接报错。
5. 目标 HEF 已存在时停止，避免覆盖。

编译可能持续较长时间，应保持 Docker Desktop 和容器运行。

## 8. 为什么必须指定六个端节点

当前标准 YOLOv8n 检测头具有三个尺度，每个尺度包含边框回归和类别分支，因此显式指定六个卷积端节点。

如果让解析器自动选择，可能选到 Sigmoid 或 Concat 等激活/拼接节点，随后出现 NMS 配置错误，例如：

```text
expected conv but found activation layer
```

这六个节点只适用于当前已验证的标准 YOLOv8n 图结构。如果发生以下变化，必须重新检查 ONNX 节点：

- 更换 YOLO 模型尺寸或代际。
- 修改检测头。
- 更换 Ultralytics 导出版本后节点名变化。
- 使用分割、姿态或分类模型。
- ONNX 图被其他工具再次简化或修改。

不要为了让命令运行而盲目复制端节点名称。

## 9. 验证 HEF

Windows：

```powershell
Get-Item -LiteralPath $HefPath |
  Select-Object FullName, Length, LastWriteTime

Get-FileHash -LiteralPath $HefPath -Algorithm SHA256
```

当前已验证参考文件：

```text
yolov8n_bottle_640_hailo8l.hef
SHA-256 339083786AB96041C454C0AE17B3B82C24EB492D03417A59388E1AA13B4D1CD2
```

哈希一致只证明文件身份一致。部署前仍需在 Hailo-8L 上验证：

- HEF 可以加载。
- 输入预处理与训练一致。
- 输出后处理和 NMS 正确。
- 类别数和类别顺序正确。
- 实际摄像头画面能够稳定检测。

## 10. 常见故障

### Docker 不可用

```text
Docker Desktop 未启动或 Docker 引擎不可用
```

先运行：

```powershell
docker info
docker start hailo-suite
```

### 容器工作区没有写权限

不要把日志和产物写入不可写的 `/workspace`。当前脚本使用：

```text
/home/hailo/model-conversion/<WorkName>
```

### 找不到 ONNX

确认：

- Windows ONNX 路径存在。
- `prepare_hailo_workspace.ps1` 已成功执行。
- `-OnnxFileName` 与复制进容器的文件名完全一致。

### 校准目录为空

准备脚本只接受：

```text
.jpg .jpeg .png .bmp
```

确认图片真实存在，且不是只包含子目录。

### NMS 或端节点错误

如果模型不是标准 YOLOv8n，先检查 ONNX 网络图和输出层，再修改端节点。不要删除端节点参数后继续声称转换可靠。

### 输出文件已存在

这是防覆盖保护。使用新文件名，或者在明确确认旧文件已经备份后再处理旧文件。

### Windows、WSL 和容器路径混用

- Windows PowerShell：`C:\...`、`Get-FileHash`。
- WSL：`/mnt/c/...`、`sha256sum`。
- 容器：`/home/hailo/...`。

不要把 `C:\...` 路径直接放进 Linux 容器命令。

## 11. 验证边界

完整流程分为独立关口：

```text
PT 可加载
  → ONNX checker/Runtime 通过
  → 校准工作区正确
  → Hailo 编译成功
  → HEF 哈希核对
  → Hailo-8L 实机推理验证
```

任何一关成功都不能替代下一关。本仓库记录的参考 HEF 已完成文件级核对，但文档更新本身没有重新运行耗时编译。
