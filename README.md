# YOLO 学习、训练与模型转换

这个仓库面向希望系统学习 YOLO 的开发者，重点回答四类问题：

1. YOLO 是什么，不同代际的模型有什么区别。
2. 如何准备数据集并完成训练、验证、预测和断点续训。
3. Windows、CUDA、路径、输出目录和模型身份方面有哪些常见坑。
4. 如何把训练得到的 PyTorch 权重转换成 ONNX、K230 KModel 或 Hailo HEF。

仓库中的塑料瓶检测任务是一个经过实际训练和转换验证的示例，不是本仓库唯一支持的数据集。代码和文档可以替换成自己的模型、数据集和输出目录。

## 推荐阅读顺序

1. [YOLO 基础](docs/YOLO基础.md)
2. [训练指南](docs/训练指南.md)
3. [常见问题与排查](docs/常见问题.md)
4. [已验证示例](docs/已验证示例.md)
5. [模型转换总览](Model_Conversion/README.md)
6. [K230 转换](Model_Conversion/K230/README.md)
7. [K230 板端运行](K230_Run/README.md)
8. [Hailo-8L 转换](Model_Conversion/Hailo/README.md)

历史实验笔记保存在 `docs/archive/`。历史笔记中的路径、模型和结论不应直接当作当前操作步骤。

## 仓库结构

```text
E:\YOLO
├─ docs/                       # 原理、训练方法、常见坑和历史笔记
├─ yolo_new/                   # Ultralytics 新版 YOLO 训练示例
│  ├─ Powershell/              # PowerShell 入口
│  └─ Python/                  # Python API 入口
├─ yolov5_train/               # 经典 YOLOv5 训练和导出入口
├─ Model_Conversion/           # PT、ONNX、KModel、HEF 转换方法
├─ K230_Run/                   # K230 板端推理示例与适配边界
├─ Model_Traning/              # 预训练权重和离线资源管理
└─ start_yolov8n_640.ps1       # 已验证的 YOLOv8n 640 训练入口
```

`Model_Traning` 是仓库已有目录名，拼写暂时保留以避免破坏现有脚本路径。

数据集、训练结果、离线依赖和转换产物默认放在仓库外：

```text
%USERPROFILE%\Desktop\YOLOTraining
├─ Datasets/
├─ Training_runs/
├─ Module_conversion/
├─ Dependencies/
└─ Config/
```

可以通过环境变量更换这个目录：

```powershell
$env:YOLO_WORKSPACE_ROOT = 'D:\YOLOTraining'
```

## 两条训练路线

| 路线 | 入口 | 适合场景 | 关键区别 |
|---|---|---|---|
| 经典 YOLOv5 | `yolov5_train/` | 学习传统 YOLOv5、兼容已有部署链 | 使用独立的 `train.py`、`val.py`、`export.py` |
| Ultralytics 新版 YOLO | `yolo_new/` | YOLOv8、YOLO11、YOLO12、YOLO26 等 | 使用统一的 `yolo` CLI 或 `YOLO` Python API |

模型文件名不能可靠证明模型结构。部署前应结合模型元数据、网络结构、输入输出张量和文件哈希确认模型身份。

## 快速开始

### 1. 准备数据集 YAML

以单类别塑料瓶数据集为例：

模板文件：[configs/bottle.example.yaml](configs/bottle.example.yaml)

```yaml
path: C:/Users/your-name/Desktop/YOLOTraining/Datasets/bottle/dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['plastic-bottle']
```

检查事项：

- `train`、`val`、`test` 对应目录真实存在。
- 每张图片应有同名的 YOLO 标签文件。
- 标签格式为 `class x_center y_center width height`，坐标归一化到 `0~1`。
- `nc` 必须和 `names` 的类别数量一致。

### 2. 训练 Ultralytics YOLOv8n

当前已验证入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_yolov8n_640.ps1
```

默认示例参数：

```text
model   = yolov8n.pt
imgsz   = 640
batch   = 16
epochs  = 200
device  = 0
workers = 0
name    = yolov8n_640
```

更通用的 PowerShell 和 Python 示例位于：

```text
yolo_new/Powershell/
yolo_new/Python/
```

### 3. 训练经典 YOLOv5

先准备官方 YOLOv5 源码和环境，再执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\yolov5_train\train_yolov5.ps1
```

仓库不会跟踪完整的 `ultralytics/yolov5` 上游源码。初始化方法和锁定版本说明见 [经典 YOLOv5 说明](yolov5_train/README.md)。

### 4. 验证、预测和导出

Ultralytics 新版 YOLO：

```text
yolo_new/Powershell/val_yolo_new.ps1
yolo_new/Powershell/predict_yolo_new.ps1
yolo_new/Powershell/resume_yolo_new.ps1
Model_Conversion/export_pt_to_onnx.ps1
```

经典 YOLOv5：

```text
yolov5_train/export_yolov5_onnx.ps1
Model_Conversion/export_yolov5_pt_to_onnx.ps1
```

## 模型转换

转换链路不能只看文件扩展名：

```text
训练权重 .pt
    └─ 导出并验证 ONNX
         ├─ PC / ONNX Runtime
         ├─ K230 / nncase → .kmodel
         └─ Hailo Dataflow Compiler → .hef
```

转换时必须确认：

- 模型代际和检测头类型。
- 输入尺寸、batch、动态或静态输入。
- ONNX opset、IR version 和算子兼容性。
- 量化数据的预处理方式和代表性。
- 板端后处理与模型输出格式一致。

详细命令和安全封装脚本见 [Model_Conversion/README.md](Model_Conversion/README.md)。经典 YOLOv5n 320 KModel 的摄像头推理方法见 [K230 板端运行说明](K230_Run/README.md)。

### YOLOv8n 转 Hailo-8L

当前已验证方法使用静态 640 ONNX、opset 17、1024 张校准图片和 `hailo8l` 架构：

```powershell
# 1. best.pt → 静态 ONNX
.\Model_Conversion\export_pt_to_onnx.ps1 `
  -ModelPath '<训练结果>\weights\best.pt' `
  -OutputPath '<转换目录>\yolov8n.onnx' `
  -ImageSize 640 `
  -Opset 17 `
  -Simplify

# 2. 准备容器工作区和校准集
.\Model_Conversion\Hailo\prepare_hailo_workspace.ps1 `
  -OnnxPath '<转换目录>\yolov8n.onnx' `
  -CalibrationImages '<数据集>\valid\images' `
  -WorkName 'yolov8n-bottle-640' `
  -CalibrationCount 1024

# 3. 先预检
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath '<转换目录>\yolov8n_bottle_640_hailo8l.hef' `
  -Classes 1 `
  -PreflightOnly

# 4. 去掉 -PreflightOnly 后正式编译
.\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 `
  -WorkName 'yolov8n-bottle-640' `
  -OnnxFileName 'yolov8n.onnx' `
  -OutputPath '<转换目录>\yolov8n_bottle_640_hailo8l.hef' `
  -Classes 1
```

完整的 Docker、端节点、校准集、哈希验证和故障排查方法见 [YOLOv8n 转 Hailo-8L HEF](Model_Conversion/Hailo/README.md)。

## 已验证示例

当前仓库外保存有以下塑料瓶示例结果：

| 项目 | 位置或标识 |
|---|---|
| 经典 YOLOv5n 320 | `Training_runs/bottle/yolov5n_classic_320` |
| YOLOv8n 640 | `Training_runs/bottle/yolov8n_640` |
| K230 KModel | `yolov5n_bottle_320_k230.kmodel` |
| Hailo-8L HEF | `yolov8n_bottle_640_hailo8l.hef` |

转换文档中记录的 KModel 和 HEF SHA-256 已与本地参考产物核对。训练结果和转换产物不直接提交到普通 Git 历史。

训练指标、参数来源和比较边界见 [已验证示例](docs/已验证示例.md)。

## 最容易踩的坑

- 把 YOLOv8 类模型误当成经典 YOLOv5。
- 数据集 YAML 路径、类别数或目录名不一致。
- Windows 路径包含空格却没有正确引用。
- 在另一个盘符启动 YOLOv5，触发跨盘符相对路径错误。
- CUDA 可见，但 PyTorch 构建不支持显卡计算能力。
- 同时运行两个训练任务争抢同一张 GPU。
- 训练、验证和导出使用了不同的 `imgsz` 或不同权重。
- 复用已有输出目录，导致自动编号、混入旧结果或覆盖证据。
- 只修改 ONNX `ir_version`，却没有解决 opset 或算子不兼容。
- K230/Hailo 板端后处理与模型输出张量不匹配。

完整排查方法见 [常见问题与排查](docs/常见问题.md)。

## 大文件策略

以下内容不应直接进入普通 Git 历史：

- 数据集图片和标签副本。
- `best.pt`、`last.pt` 和预训练权重。
- ONNX、KModel、HEF、TensorRT Engine。
- 训练日志、预测图片和缓存。
- 离线 wheel 和转换工具链工作目录。

`.gitignore` 负责阻止常见产物误提交。需要共享的大文件应使用 GitHub Release 或其他外部存储，并记录文件大小和 SHA-256。

当前 GitHub 仓库尚未发布正式 Release。在 Release 资产上传完成前，`Model_Traning/download_assets.ps1` 中的下载流程只能作为发布方案，不能视为已可用的下载服务。

## 验证边界

- 文档中的“已验证”只指明确记录了命令、环境、输出或哈希的流程。
- 脚本通过语法检查不等于训练或板端部署已经成功。
- 软件训练成功不等于 ONNX、KModel、HEF 或板端运动链路成功。
- 整理仓库时不得把历史输出目录当成可随意删除的缓存。
