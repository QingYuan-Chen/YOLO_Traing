# YOLO 训练、模型转换与边缘部署指南

本仓库是一套面向实践的 YOLO 学习资料和可复用脚本，覆盖从数据集准备到边缘设备部署的完整链路：

```text
认识 YOLO
  → 准备数据集
  → 配置 Python / CUDA 环境
  → 训练与断点续训
  → 验证与预测
  → 导出 ONNX
  → 编译 K230 KModel 或 Hailo HEF
  → 核对板端预处理、输出和后处理
```

仓库同时保留两条训练路线：

- 经典 Ultralytics YOLOv5 仓库路线。
- `ultralytics` Python 包提供的 YOLOv8、YOLO11、YOLO12、YOLO26 等新版路线。

塑料瓶检测是本仓库中已经完成训练和文件级转换验证的示例，不是唯一支持的任务。替换数据集 YAML、初始权重、类别名称和输出路径后，脚本可以用于自己的目标检测项目。

> [!IMPORTANT]
> 仓库保存的是代码、配置模板和操作说明，不直接保存数据集、训练权重、ONNX、KModel 或 HEF。模型文件名也不能证明模型结构，部署前必须核对真实输入输出、检测头、类别顺序和文件哈希。

## 1. 仓库能解决什么问题

| 目标 | 本仓库提供的内容 | 主要入口 |
|---|---|---|
| 学习 YOLO 基础 | YOLO 工作方式、模型代际、尺寸选择、指标解释 | [YOLO 基础](docs/YOLO基础.md) |
| 准备训练数据 | 数据集 YAML 模板、目录约定、标签检查项 | [训练指南](docs/训练指南.md) |
| 训练新版 YOLO | 环境安装、训练、验证、预测、续训脚本 | [yolo_new/README.md](yolo_new/README.md) |
| 训练经典 YOLOv5 | 上游源码初始化、锁定版本、训练和导出 | [yolov5_train/README.md](yolov5_train/README.md) |
| 导出 ONNX | 新版 YOLO 和经典 YOLOv5 的独立导出脚本 | [模型转换总览](Model_Conversion/README.md) |
| 部署到 K230 | opset 12 ONNX、nncase 2.9 编译、板端推理脚本 | [K230 转换](Model_Conversion/K230/README.md) |
| 部署到 Hailo-8L | YOLOv8n 解析、校准、量化和 HEF 编译 | [Hailo-8L 转换](Model_Conversion/Hailo/README.md) |
| 排查失败 | CUDA、Windows 路径、数据集、输出目录和后处理问题 | [常见问题](docs/常见问题.md) |
| 对照实际结果 | 已验证训练指标、转换产物及 SHA-256 | [已验证示例](docs/已验证示例.md) |

如果是第一次接触本仓库，推荐按下面的顺序阅读：

1. 本 README 的“核心概念”和“环境准备”。
2. [YOLO 基础](docs/YOLO基础.md)。
3. [训练指南](docs/训练指南.md)。
4. 根据目标选择新版 YOLO 或经典 YOLOv5 路线。
5. 训练成功后再阅读 [模型转换总览](Model_Conversion/README.md)。
6. 遇到错误时按 [常见问题](docs/常见问题.md)逐项排查。

## 2. 核心概念

### 2.1 YOLO 是什么

YOLO 是一类单阶段视觉模型。目标检测任务通常一次前向计算就产生候选框、类别和置信度，再通过阈值过滤和 NMS 得到最终结果。相较于只输出整张图片类别的分类模型，检测模型还需要回答“目标在哪里”。

一个完整的检测结果至少包含：

```text
类别 ID + 置信度 + 边界框坐标
```

训练、导出和部署必须围绕同一份模型契约：

| 项目 | 需要明确的内容 |
|---|---|
| 任务类型 | detect、segment、pose、classify 或 OBB |
| 模型代际 | 经典 YOLOv5、YOLOv8、YOLO11 等 |
| 输入 | 尺寸、batch、NCHW/NHWC、RGB/BGR、数据类型 |
| 预处理 | resize、Letterbox、填充值、归一化和量化范围 |
| 输出 | 张量数量、形状、顺序、是否已经解码 |
| 后处理 | 置信度计算、NMS、坐标还原和类别映射 |

其中任意一项不匹配，都可能出现“模型能加载但检测结果完全错误”的情况。

### 2.2 经典 YOLOv5 与新版 Ultralytics YOLO

| 项目 | 经典 YOLOv5 | 新版 Ultralytics YOLO |
|---|---|---|
| 代码来源 | 独立的 `ultralytics/yolov5` 仓库 | `ultralytics` Python 包 |
| 常用入口 | `train.py`、`val.py`、`detect.py`、`export.py` | `yolo` CLI 或 `from ultralytics import YOLO` |
| 检测头 | 经典 anchor-based YOLOv5 Detect | 随模型代际变化，通常不能套用 YOLOv5 后处理 |
| 本仓库目录 | `yolov5_train/` | `yolo_new/` |
| 当前部署示例 | YOLOv5n 320 → K230 | YOLOv8n 640 → Hailo-8L |

不要因为文件叫 `YOLOV5.pt` 就认定它是经典 YOLOv5。应结合权重内部模型类型、网络层名称、ONNX 张量和 SHA-256 判断。

### 2.3 n、s、m、l、x 怎么选

同一代模型经常提供不同规模：

```text
n → s → m → l → x
小、快、资源占用低        大、精度潜力高、资源需求高
```

边缘部署建议先从 `n` 或 `s` 开始。模型越大通常意味着：

- GPU 显存占用更高。
- 训练和导出时间更长。
- 板端延迟、内存和带宽压力更大。
- 转换工具更容易遇到算子或资源限制。

### 2.4 输入尺寸不是文件名的一部分

`yolov8n_640.pt` 只是一个命名习惯。实际训练尺寸由训练参数决定，导出尺寸由导出参数决定，最终还要检查 ONNX 或板端模型输入张量。

增大输入尺寸通常有利于小目标，但会增加计算量。公平比较两个模型时，应至少统一：

- 数据集版本和划分。
- 输入尺寸。
- 训练轮数和主要增强策略。
- 验证 split 和验证命令。
- 置信度、IoU 和 NMS 设置。

## 3. 仓库结构

```text
YOLO_Traing/
├─ README.md                     # 项目总入口
├─ configs/
│  └─ bottle.example.yaml        # 数据集配置模板
├─ docs/
│  ├─ YOLO基础.md
│  ├─ 训练指南.md
│  ├─ 常见问题.md
│  ├─ 已验证示例.md
│  └─ archive/                   # 历史实验笔记，只作背景参考
├─ yolo_new/
│  ├─ Powershell/                # 新版 YOLO 命令行封装
│  └─ Python/                    # 新版 YOLO Python API 示例
├─ yolov5_train/
│  ├─ setup_yolov5_env.ps1       # 创建环境并克隆指定 YOLOv5 提交
│  ├─ train_yolov5.ps1
│  └─ export_yolov5_onnx.ps1
├─ Model_Conversion/
│  ├─ export_pt_to_onnx.ps1      # 新版 YOLO：PT → ONNX
│  ├─ export_yolov5_pt_to_onnx.ps1
│  ├─ K230/                      # ONNX → KModel
│  └─ Hailo/                     # ONNX → HEF
├─ K230_Run/
│  ├─ README.md
│  └─ yolov5.py                  # YOLOv5n 320 板端摄像头推理
├─ Model_Traning/                # 预训练权重和离线资源管理
├─ scripts/
│  └─ project_paths.ps1          # 外部工作目录解析
└─ start_yolov8n_640.ps1         # 已验证的 YOLOv8n 640 训练入口
```

`Model_Traning` 是仓库已有目录名，虽然单词拼写不标准，但暂时保留以避免破坏现有脚本路径。

历史笔记可能包含旧环境、旧路径和失败尝试。`docs/archive/` 中的内容不能直接替代当前 README 和转换文档。

## 4. 克隆仓库与外部工作目录

### 4.1 克隆

```powershell
git clone https://github.com/QingYuan-Chen/YOLO_Traing.git
Set-Location .\YOLO_Traing
```

后续命令默认从仓库根目录执行。

### 4.2 为什么数据和产物放在仓库外

数据集、训练结果和模型文件体积大，而且可能包含无法重新生成的实验证据。将它们与代码仓库分开可以避免：

- 意外把数 GB 数据推到 GitHub。
- `git clean` 或仓库整理误删唯一权重。
- 多个实验把输出混到源码目录。
- 转换临时文件污染提交历史。

默认工作目录为：

```text
%USERPROFILE%\Desktop\YOLOTraining
├─ Datasets/                     # 数据集和 YAML
├─ Training_runs/                # 训练、验证和预测结果
├─ Module_conversion/            # ONNX、KModel、HEF
├─ Dependencies/                 # 必要的离线依赖
└─ Config/                       # Ultralytics 用户配置
```

使用其他位置时，设置：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

也可以在支持的脚本中显式传入：

```powershell
-WorkspaceRoot 'D:\YOLOTraining'
```

优先级是“命令行参数高于环境变量，环境变量高于默认桌面目录”。

## 5. 环境准备

### 5.1 基础工具

训练至少需要：

- Windows PowerShell。
- Git。
- Conda 或其他可隔离 Python 环境的工具。
- NVIDIA 驱动；使用 GPU 时还需要与显卡架构兼容的 PyTorch CUDA wheel。

模型转换还可能需要：

| 目标 | 额外要求 |
|---|---|
| ONNX | `onnx`、`onnxruntime` |
| K230 | Docker Desktop、已有 `k230-converter` 容器、nncase 2.9 |
| Hailo-8L | Docker Desktop、已有 `hailo-suite` 容器、Hailo Model Zoo 环境 |
| K230 板端 | 支持 `libs.AIBase`、`libs.AI2D` 和 `aidemo` 的 CanMV 固件 |

### 5.2 安装新版 YOLO 环境

仓库提供带版本默认值的环境脚本：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\yolo_new\Powershell\setup_yolo_new_env.ps1
```

脚本当前默认安装：

```text
Python          3.12
PyTorch         2.8.0 + cu128
torchvision     0.23.0
Ultralytics     8.4.60
ONNX            1.17.0
ONNX Runtime    1.26.0
```

这些版本是脚本的可复现默认值，不代表适用于所有显卡或所有未来模型。需要修改时通过参数传入，不要只写“安装最新版”而不记录最终版本。

激活并获取当前环境的可执行文件：

```powershell
conda activate yolo
$YoloExe = (Get-Command yolo).Source
$PythonExe = (Get-Command python).Source
```

仓库部分脚本保留了原验证机器上的绝对默认路径，例如 `E:\Anaconda_envs\...`。在其他电脑上应显式传入 `$YoloExe` 或 `$PythonExe`。

### 5.3 验证 CUDA 不是“只看得到”

```powershell
& $PythonExe -c @'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("supported arch:", torch.cuda.get_arch_list())

if torch.cuda.is_available():
    value = torch.ones(1, device="cuda") * 2
    print("GPU tensor:", value)
    print("device:", torch.cuda.get_device_name(0))
'@

nvidia-smi
```

`torch.cuda.is_available()` 为 `True` 只代表 PyTorch 能看到 CUDA。最小张量计算和真实训练日志中的显存占用都正常，才能进一步确认当前 wheel 支持这张显卡。

## 6. 数据集准备

### 6.1 推荐目录

标准检测数据集可以组织为：

```text
dataset/
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

每张有标签的图片应有一个同名 `.txt` 文件：

```text
images/000123.jpg
labels/000123.txt
```

单行 YOLO 检测标签格式为：

```text
class_id x_center y_center width height
```

例如：

```text
0 0.512500 0.483333 0.225000 0.316667
```

要求：

- `class_id` 从 `0` 开始。
- 坐标和宽高归一化到 `0~1`。
- 一张图片有多个目标时写多行。
- 无目标图片可以使用空标签文件，但应明确保留它的目的。

### 6.2 数据集 YAML

复制 [configs/bottle.example.yaml](configs/bottle.example.yaml)并修改路径：

```yaml
path: C:/Users/your-name/Desktop/YOLOTraining/Datasets/bottle/dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['plastic-bottle']
```

注意：

- YAML 键叫 `val`，实际目录可以叫 `valid`。
- Windows 路径建议使用正斜杠 `/`。
- `nc` 必须等于 `names` 的数量。
- `names` 的顺序决定部署时的类别 ID 映射。
- 不要把本机绝对数据路径提交成所有人都必须使用的默认配置。

### 6.3 训练前最低检查

在开始数小时训练前，至少确认：

1. `train`、`val`、`test` 路径存在。
2. 图片可解码，不是损坏或零字节文件。
3. 图片与标签基本一一对应。
4. 标签列数为 5，数值范围合法。
5. 类别编号小于 `nc`。
6. 训练集和验证集没有明显重复或数据泄漏。
7. 类别数量、每类样本量、空标签和缺失标签数量已记录。
8. 数据集版本或来源可追溯。

## 7. 路线一：训练新版 Ultralytics YOLO

### 7.1 快速启动已验证的 YOLOv8n 640 示例

准备变量：

```powershell
conda activate yolo

$WorkspaceRoot = if ($env:YOLO_WORKSPACE_ROOT) {
    $env:YOLO_WORKSPACE_ROOT
} else {
    Join-Path $env:USERPROFILE 'Desktop\YOLOTraining'
}

$YoloExe = (Get-Command yolo).Source
$DataYaml = Join-Path $WorkspaceRoot 'Datasets\bottle\bottle_plastic.yaml'
$ModelPath = Join-Path $PWD 'Model_Traning\weights\yolov8\yolov8n.pt'
$ProjectDir = Join-Path $WorkspaceRoot 'Training_runs\bottle'
```

开始训练：

```powershell
.\start_yolov8n_640.ps1 `
  -WorkspaceRoot $WorkspaceRoot `
  -YoloExe $YoloExe `
  -ModelPath $ModelPath `
  -DataYaml $DataYaml `
  -ProjectDir $ProjectDir `
  -RunName 'yolov8n_640' `
  -ImageSize 640 `
  -BatchSize 16 `
  -Epochs 200 `
  -Device 0 `
  -Workers 0
```

脚本会：

1. 检查 `yolo.exe`、初始权重和数据 YAML。
2. 拒绝复用已经存在的 run 目录。
3. 把 Ultralytics 配置写到外部工作目录。
4. 执行 `detect train`。
5. 把控制台输出同时写入日志。
6. 原样返回 YOLO 进程退出码。

Windows 首次运行建议使用 `Workers 0`。训练稳定后再逐步增加，避免数据加载多进程造成卡住或重复启动。

### 7.2 训练输出

默认输出结构类似：

```text
Training_runs/bottle/yolov8n_640/
├─ weights/
│  ├─ best.pt
│  └─ last.pt
├─ args.yaml
├─ results.csv
├─ results.png
├─ confusion_matrix.png
└─ 其他训练图表
```

关键文件：

| 文件 | 用途 |
|---|---|
| `best.pt` | 按训练期间目标指标选择的最佳权重 |
| `last.pt` | 最后一个 epoch 的权重，通常用于断点续训 |
| `args.yaml` | 实际训练参数 |
| `results.csv` | 每个 epoch 的指标记录 |
| 控制台日志 | 环境、警告、异常和完整启动证据 |

进程启动不等于训练成功。至少看到完整 epoch 写入 `results.csv`，并确认权重和退出状态后再判断。

### 7.3 验证

```powershell
.\yolo_new\Powershell\val_yolo_new.ps1 `
  -WorkspaceRoot $WorkspaceRoot `
  -YoloExe $YoloExe `
  -Weights (Join-Path $ProjectDir 'yolov8n_640\weights\best.pt') `
  -DataYaml $DataYaml `
  -ImageSize 640 `
  -Split test `
  -Device 0
```

验证时应记录：

- 权重文件 SHA-256。
- 数据集版本和 split。
- 输入尺寸。
- Precision、Recall、mAP50、mAP50-95。
- 框架版本和完整命令。

### 7.4 预测

```powershell
.\yolo_new\Powershell\predict_yolo_new.ps1 `
  -WorkspaceRoot $WorkspaceRoot `
  -YoloExe $YoloExe `
  -Weights (Join-Path $ProjectDir 'yolov8n_640\weights\best.pt') `
  -Source (Join-Path $WorkspaceRoot 'Datasets\bottle\dataset\test\images') `
  -ProjectDir (Join-Path $ProjectDir 'predictions') `
  -RunName 'yolov8n_640_test' `
  -ImageSize 640 `
  -Confidence 0.25 `
  -Device 0
```

预测适合检查漏检、误检、标签顺序、边缘目标和困难场景，但不能替代正式验证指标。

### 7.5 断点续训

```powershell
.\yolo_new\Powershell\resume_yolo_new.ps1 `
  -WorkspaceRoot $WorkspaceRoot `
  -YoloExe $YoloExe `
  -LastWeights (Join-Path $ProjectDir 'yolov8n_640\weights\last.pt')
```

只使用同一次实验的 `last.pt`。`resume=True` 会恢复原训练配置和优化器状态，拿错权重可能把不同实验的配置混在一起。

## 8. 路线二：训练经典 YOLOv5

### 8.1 初始化环境和上游源码

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\yolov5_train\setup_yolov5_env.ps1
```

脚本会：

- 在 Git 忽略的本地目录 `.conda_envs/yolov5` 中创建 Python 3.9 环境。
- 克隆官方 `ultralytics/yolov5`。
- 默认锁定提交 `70b964b6d5067fff621f724c85d0e39e6b4c8e4e`。
- 安装上游 `requirements.txt`。

仓库不会提交完整的 YOLOv5 上游源码。不要为“更新依赖”直接切到最新 `master`，否则训练结构、导出行为和部署兼容性都可能变化。

### 8.2 开始训练

使用初始化脚本创建的解释器时，显式传入路径：

```powershell
$YoloV5Python = (Resolve-Path '.\.conda_envs\yolov5\python.exe').Path
$YoloV5Root = (Resolve-Path '.\yolov5_train\yolov5').Path
$YoloV5Weights = (Resolve-Path '.\Model_Traning\weights\yolov5\yolov5n.pt').Path

.\yolov5_train\train_yolov5.ps1 `
  -WorkspaceRoot $WorkspaceRoot `
  -PythonExe $YoloV5Python `
  -YoloV5Root $YoloV5Root `
  -DataYaml $DataYaml `
  -Weights $YoloV5Weights `
  -ProjectDir $ProjectDir `
  -RunName 'yolov5n_classic_320' `
  -ImageSize 320 `
  -BatchSize 16 `
  -Epochs 200 `
  -Device 0 `
  -Workers 0
```

脚本会先切换到 YOLOv5 上游根目录再运行 `train.py`，避免 Windows 不同盘符之间计算相对路径时出现错误。

### 8.3 验证经典 YOLOv5

本仓库没有额外封装 `val.py`，可在锁定的上游目录中执行：

```powershell
Push-Location $YoloV5Root
try {
    & $YoloV5Python .\val.py `
      --weights (Join-Path $ProjectDir 'yolov5n_classic_320\weights\best.pt') `
      --data $DataYaml `
      --img 320 `
      --task test `
      --device 0 `
      --workers 0
}
finally {
    Pop-Location
}
```

经典 YOLOv5 和 YOLOv8n 示例使用了不同输入尺寸，因此不能只拿两行 mAP 直接判断模型代际优劣。

## 9. 导出 ONNX

### 9.1 导出前先确认

记录：

- 源权重的真实模型类型和 SHA-256。
- 导出输入尺寸。
- batch 是否固定为 1。
- dynamic 是否关闭。
- 目标平台支持的 opset。
- 是否执行图简化。

导出成功后至少运行 `onnx.checker`，并使用 ONNX Runtime 创建 CPU 推理会话。只生成一个 `.onnx` 文件不算完成验证。

### 9.2 新版 YOLO：PT → ONNX

```powershell
$BestV8 = Join-Path $ProjectDir 'yolov8n_640\weights\best.pt'
$OnnxV8 = Join-Path $WorkspaceRoot 'Module_conversion\bottle\yolov8n.onnx'

.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath $BestV8 `
  -OutputPath $OnnxV8 `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify `
  -YoloExe $YoloExe `
  -PythonExe $PythonExe `
  -WorkspaceRoot $WorkspaceRoot
```

脚本在临时目录中导出，验证通过后才复制到目标路径。目标文件已存在时会停止，避免静默覆盖参考模型。

### 9.3 经典 YOLOv5：PT → ONNX

```powershell
$BestV5 = Join-Path $ProjectDir 'yolov5n_classic_320\weights\best.pt'
$OnnxV5 = Join-Path $WorkspaceRoot `
  'Module_conversion\bottle\yolov5n_bottle_320_k230.onnx'

.\Model_Conversion\export_yolov5_pt_to_onnx.ps1 `
  -ModelPath $BestV5 `
  -OutputPath $OnnxV5 `
  -ImageSize 320 `
  -Opset 12 `
  -PythonExe $YoloV5Python `
  -ValidationPython $PythonExe `
  -YoloV5Root $YoloV5Root
```

K230 当前参考链路使用经典 YOLOv5 导出器、静态 NCHW、batch 1 和 opset 12。不要只修改现有 ONNX 的 `ir_version` 来伪装兼容。

## 10. 部署到 K230

### 10.1 编译 KModel

当前流程使用 Docker 容器 `k230-converter` 中的 nncase 2.9：

```powershell
$CalibrationImages = Join-Path $WorkspaceRoot `
  'Datasets\bottle\dataset\valid\images'
$KModelPath = Join-Path $WorkspaceRoot `
  'Module_conversion\bottle\yolov5n_bottle_320_k230.kmodel'

.\Model_Conversion\K230\compile_onnx_to_kmodel.ps1 `
  -OnnxPath $OnnxV5 `
  -CalibrationImages $CalibrationImages `
  -OutputPath $KModelPath `
  -InputWidth 320 `
  -InputHeight 320 `
  -CalibrationCount 10
```

脚本会完成：

1. ONNX 静态输入和 checker 检查。
2. 等间隔抽取代表性校准图片。
3. 创建独立容器工作区。
4. 使用 nncase 编译 KModel。
5. 使用 KModel 模拟器执行加载检查。
6. 复制回 Windows。
7. 比较容器和 Windows 两侧 SHA-256。
8. 清理临时工作区。

当前参考模型契约：

```text
文件：yolov5n_bottle_320_k230.kmodel
输入：uint8 [1, 3, 320, 320]
输出：float32 [1, 6300, 6]
SHA-256：36F50EC97A9E1EF2AFD8BB0625CC6E1E8B6C388BD706F74F04A02385EA00920A
```

### 10.2 K230 板端运行

板端脚本：[K230_Run/yolov5.py](K230_Run/yolov5.py)

默认 SD 卡文件：

```text
/sdcard/yolov5n_bottle_320_k230.kmodel
/sdcard/yolov5.py
```

脚本会：

- 启动前检查 KModel 是否存在。
- 检查 KModel 是否为单输入、单输出。
- 首帧核对输出是否为 `[1, 6300, 6]`。
- 使用与 CanMV 后处理一致的左上对齐 Letterbox。
- 调用 `aidemo.yolov5_det_postprocess`。
- 绘制检测框、类别、置信度和 FPS。

`1920×1080 → 320×320` 时 padding 为：

```text
top=0, bottom=140, left=0, right=0
```

不要单独改成上下各填 70，否则预处理与坐标还原约定不一致，检测框可能发生纵向偏移。

完整上板说明见 [K230_Run/README.md](K230_Run/README.md)。

## 11. 部署到 Hailo-8L

当前已验证路线是：

```text
YOLOv8n best.pt
  → 静态 640×640 ONNX，opset 17
  → 1024 张代表性校准图片
  → Hailo Model Zoo 解析、优化、量化和编译
  → hailo8l HEF
```

定义路径：

```powershell
$BestV8 = Join-Path $ProjectDir 'yolov8n_640\weights\best.pt'
$OnnxV8 = Join-Path $WorkspaceRoot 'Module_conversion\bottle\yolov8n.onnx'
$CalibrationImages = Join-Path $WorkspaceRoot `
  'Datasets\bottle\dataset\valid\images'
$HefPath = Join-Path $WorkspaceRoot `
  'Module_conversion\bottle\yolov8n_bottle_640_hailo8l.hef'
```

导出 ONNX：

```powershell
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath $BestV8 `
  -OutputPath $OnnxV8 `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify `
  -YoloExe $YoloExe `
  -PythonExe $PythonExe `
  -WorkspaceRoot $WorkspaceRoot
```

准备容器工作区：

```powershell
.\Model_Conversion\Hailo\prepare_hailo_workspace.ps1 `
  -OnnxPath $OnnxV8 `
  -CalibrationImages $CalibrationImages `
  -WorkName 'yolov8n-bottle-640' `
  -CalibrationCount 1024 `
  -Container 'hailo-suite'
```

先做预检：

```powershell
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath $HefPath `
  -Classes 1 `
  -Container 'hailo-suite' `
  -PreflightOnly
```

预检通过后正式编译：

```powershell
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath $HefPath `
  -Classes 1 `
  -Container 'hailo-suite'
```

当前参考产物：

```text
文件：yolov8n_bottle_640_hailo8l.hef
架构：hailo8l
类别数：1
SHA-256：339083786AB96041C454C0AE17B3B82C24EB492D03417A59388E1AA13B4D1CD2
```

标准 YOLOv8n 检测头使用三个尺度，每个尺度包含回归和类别分支，因此当前脚本显式指定六个卷积端节点。更换模型代际、任务类型、检测头或导出版本后，必须重新检查 ONNX 图，不能盲目复用节点名称。

完整的 Docker 环境、端节点解释、校准集要求和故障排查见 [Hailo-8L 转换说明](Model_Conversion/Hailo/README.md)。

## 12. 当前已验证的参考结果

### 12.1 训练记录

| 模型 | 输入 | 最后记录的 epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| 经典 YOLOv5n | 320 | 199 | 0.93372 | 0.85354 | 0.91405 | 0.73853 |
| YOLOv8n | 640 | 200 | 0.94692 | 0.86287 | 0.91940 | 0.78375 |

这些结果来自不同输入尺寸和训练路线，不能当作严格的模型排行榜。完整来源和比较边界见 [已验证示例](docs/已验证示例.md)。

### 12.2 转换产物

| 产物 | 目标 | SHA-256 |
|---|---|---|
| `yolov5n_bottle_320_k230.kmodel` | K230 | `36F50EC97A9E1EF2AFD8BB0625CC6E1E8B6C388BD706F74F04A02385EA00920A` |
| `yolov8n_bottle_640_hailo8l.hef` | Hailo-8L | `339083786AB96041C454C0AE17B3B82C24EB492D03417A59388E1AA13B4D1CD2` |

哈希一致只说明文件身份一致，不等于板端预处理、推理和后处理已经正确。

## 13. 如何判断每一步真的成功

| 阶段 | 最低成功证据 | 还不能证明什么 |
|---|---|---|
| 环境 | 包能导入、GPU 最小计算成功 | 完整训练一定稳定 |
| 训练启动 | 进程存在、日志无立即报错 | 模型已经训练完成 |
| 训练完成 | 完整 epoch、指标、权重、退出码 | 部署模型一定正确 |
| ONNX 导出 | checker 通过、Runtime 会话可创建 | 目标编译器一定支持 |
| KModel/HEF 编译 | 编译成功、文件可加载、哈希一致 | 摄像头检测链路正确 |
| 板端推理 | 实机加载、输入输出匹配、检测正确 | 所有真实场景都可靠 |

正式实验建议保留：

- 完整启动命令。
- Git 提交哈希。
- Python、PyTorch、CUDA 和框架版本。
- 数据集版本和样本统计。
- 初始权重、最佳权重和最终产物 SHA-256。
- `args.yaml`、`results.csv` 和控制台日志。
- 验证命令、split 和指标。
- 转换工具版本和板端固件版本。

## 14. 常见问题速查

| 现象 | 优先检查 |
|---|---|
| `unrecognized arguments: ---img` | 参数应是两个横杠，例如 `--img` |
| 路径包含空格后参数错位 | 使用引号、`Join-Path` 和 `-LiteralPath` |
| YOLOv5 报跨盘符错误 | 先切换到 YOLOv5 上游目录 |
| CUDA 可见但 `no kernel image` | PyTorch wheel 是否支持显卡计算能力 |
| 显存没有变化 | `device`、训练进程命令行和 `nvidia-smi` |
| 输出出现 `train2`、`train3` | 显式设置唯一的 `project` 和 `name` |
| 断点续训参数异常 | 是否使用了同一次实验的 `last.pt` |
| 训练 640、导出却是 320 | 导出命令中的 `ImageSize` |
| ONNX 报 opset 或 IR 不支持 | 从 PT 按目标平台要求重新导出 |
| KModel/HEF 能加载但框错误 | RGB/BGR、Letterbox、张量布局和后处理 |
| 下载权重脚本返回 404 | 仓库尚未发布对应 GitHub Release 资产 |

详细原因和命令见 [常见问题与排查](docs/常见问题.md)。

## 15. 大文件和 Git 策略

以下内容默认不进入普通 Git 历史：

- 数据集图片和标签副本。
- `best.pt`、`last.pt` 和预训练权重。
- `.onnx`、`.kmodel`、`.hef`、TensorRT Engine。
- 训练日志、预测图片和缓存。
- Conda 环境、wheel 缓存和工具链工作目录。

`.gitignore` 已覆盖常见模型格式和输出目录。需要分享模型时，推荐：

1. 使用 GitHub Release 或受控的外部存储。
2. 记录文件名、大小、生成命令和 SHA-256。
3. 在文档中说明适用的模型结构、输入尺寸和目标硬件。
4. 不要覆盖已经作为实验依据的参考产物。

当前仓库尚未发布正式模型 Release。因此 `Model_Traning/download_assets.ps1` 目前是发布流程模板，而不是已经可用的公共下载服务。

## 16. 扩展到自己的项目

替换塑料瓶示例时，建议按以下顺序操作：

1. 新建独立的数据集目录和 YAML，不直接修改历史实验数据。
2. 确认类别数量和 `names` 顺序。
3. 选择模型代际和规模，使用新的 run name。
4. 用少量 epoch 做冒烟训练，确认数据和输出目录正常。
5. 完成正式训练并保存指标、日志和权重哈希。
6. 使用目标平台支持的输入尺寸和 opset 重新导出 ONNX。
7. 使用真实部署场景图片做量化校准。
8. 根据模型真实输出实现或选择板端后处理。
9. 分别验证空场景、单目标、多目标、遮挡、边缘和小目标。
10. 把最终模型契约、固件版本和失败样例补充到文档。

新增一个部署示例时，至少应说明：

```text
源权重身份
输入尺寸和布局
颜色空间和数据类型
预处理与量化方式
输出张量形状
后处理函数
类别顺序
目标硬件和工具版本
最终产物 SHA-256
验证范围和未验证范围
```

## 17. 验证边界

- 文档中的“已验证”只指存在明确命令、环境、输出、指标或哈希的流程。
- 脚本通过语法检查，不等于耗时训练或实体板端部署已经重新执行。
- K230 板端脚本已按当前模型契约和官方 CanMV 接口整理，但仍需实体开发板验证摄像头、显示、帧率和真实检测效果。
- Hailo 参考 HEF 已完成文件级哈希核对，文档整理本身没有重新运行耗时编译。
- 被 `.gitignore` 忽略的文件不一定是缓存；权重、转换产物和失败日志可能是唯一实验依据。

如需深入了解某一阶段，优先进入对应子目录 README，而不是从历史笔记复制命令。
