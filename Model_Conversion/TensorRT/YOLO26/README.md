# YOLO26 实例分割与单目深度 TensorRT 优化部署

本目录记录已经在 Windows + NVIDIA GeForce RTX 5060 Laptop GPU 上完成的
YOLO26s-seg、YOLO26m-seg 和 YOLO26s-depth TensorRT 转换、数值验证与推理优化方案。
大模型文件不写入 Git 历史，统一放在 GitHub Release 资产中。

YOLO26s-depth 是单目度量深度模型，输出每个像素的距离（单位：米）。Ultralytics
发布权重推荐使用 768 输入；实例分割 engine 使用 640 输入。两类任务不能混用输入尺寸或
后处理。上游说明见 [YOLO26 文档](https://docs.ultralytics.com/models/yolo26/) 和
[Depth 任务文档](https://docs.ultralytics.com/tasks/depth/)。

## 下载

最新版整合 Release：
[yolo26-seg-depth-trt-rtx5060-20260830](https://github.com/QingYuan-Chen/YOLO_Traing/releases/tag/yolo26-seg-depth-trt-rtx5060-20260830)

下载并解压 `YOLO26-seg-depth-TensorRT-RTX5060-Windows-20260830.zip` 后，主要目录为：

```text
YOLO26-seg-depth-TensorRT-RTX5060-Windows-20260830/
├─ README_CN.md
├─ BENCHMARK_RESULTS.md
├─ ENVIRONMENT.json
├─ LICENSE_NOTICE.md
├─ MANIFEST.sha256
├─ requirements-runtime.txt
├─ requirements-validation.txt
├─ models/
│  ├─ pytorch/
│  │  ├─ yolo26s-seg.pt
│  │  ├─ yolo26m-seg.pt
│  │  └─ yolo26s-depth.pt
│  └─ tensorrt/
│     ├─ yolo26s-seg-fp16-b1-640.engine
│     ├─ yolo26m-seg-fp16-b1-640.engine
│     └─ yolo26s-depth-fp16-b1-768.engine
├─ scripts/
├─ sample_data/
└─ results/
```

## 兼容性边界

TensorRT engine 不是通用模型格式。本包使用 TensorRT 10.13.3.9.post1、CUDA 12.8、
静态 batch 1 和内部 FP16 构建，只在 RTX 5060 Laptop GPU、Windows、驱动 610.88
上完成实机验证。不同 GPU 架构、TensorRT 版本、驱动或操作系统可能无法反序列化。
遇到这种情况应使用包内 `.pt` 权重在目标机器重新构建，不要把加载失败误判为权重损坏。

优化推理器使用 CUDA Graph，因此捕获后输入图像的 HWC 分辨率必须保持不变；摄像头
分辨率变化时应重新创建并捕获 pipeline。实例分割的 `--max-masks` 也是固定分配值，
默认 20，实际实例更多时需要调大。

## 环境准备

本次验证环境见 `ENVIRONMENT.json`。全新 Python 3.12 虚拟环境可尝试：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements-runtime.txt'
```

TensorRT Python 依赖包含约 1.5 GB 运行库，完整虚拟环境实测约 2.17 GiB；安装前建议
预留至少 4 GiB。若接收方已有匹配版本的 CUDA、PyTorch 和 TensorRT，应优先复用。

```powershell
& '.\.venv\Scripts\python.exe' -c "import torch,tensorrt,ultralytics; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), tensorrt.__version__, ultralytics.__version__)"
& '.\.venv\Scripts\python.exe' -m pip check
```

## 在目标机器重新转换

以下命令使用静态 batch 1、内部 FP16、FP32 I/O、4 GiB TensorRT workspace。构建过程
可能持续数分钟，且必须在目标 NVIDIA GPU 上执行。

深度模型：

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\scripts\export_yolo26_trt_engine.py' `
  '.\models\pytorch\yolo26s-depth.pt' `
  --task depth `
  --imgsz 768 `
  --workspace 4 `
  --output '.\models\tensorrt\yolo26s-depth-fp16-b1-768.engine'
```

实例分割模型将 `--task` 改为 `segment`、`--imgsz` 改为 `640`，并替换权重和输出文件名。
脚本默认不覆盖已有输出；确认需要覆盖时增加 `--force`。

## 推荐推理命令

### 单目深度

`exact-cpu` 用 OpenCV 做 letterbox，和本次数值验证的参考预处理一致。输出深度图会去除
padding，并恢复到原图分辨率。

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\scripts\infer_yolo26_depth_trt_optimized.py' `
  '.\models\tensorrt\yolo26s-depth-fp16-b1-768.engine' `
  '.\sample_data\bus.jpg' `
  --preprocess exact-cpu `
  --iterations 300 `
  --output-json '.\depth_speed.json' `
  --save-preview '.\depth_preview.jpg' `
  --save-depth-npy '.\depth_meters.npy'
```

`.npy` 是 `float32` 原图大小数组，数值单位为米；彩色预览仅用于观察，不应用于数值计算。

### 实例分割

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\scripts\infer_yolo26_trt_optimized.py' `
  '.\models\tensorrt\yolo26s-seg-fp16-b1-640.engine' `
  '.\sample_data\bus.jpg' `
  --preprocess exact-cpu `
  --confidence 0.25 `
  --max-masks 20 `
  --iterations 300 `
  --save-overlay '.\seg_result.jpg'
```

使用 m 模型时只需换成 `yolo26m-seg-fp16-b1-640.engine`。

## 深度模型公平比较的关键

静态 768 engine 必须接收 768 x 768 正方形 letterbox。Ultralytics 的 PyTorch 推理在
单张图时可能默认使用矩形最小填充；如果直接拿默认 PyTorch 结果与静态 engine 结果比较，
两端看到的输入不同，会产生虚假的深度偏差。本次先观察到约 0.53 m MAE，随后通过原始张量
对比定位到该问题。统一设置 `rect=False` 后，原图深度图 MAE 为 0.00316 m、平均相对误差
0.0671%、δ1 一致率 100%。

可复现命令：

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\scripts\benchmark_yolo26_depth_backends.py' `
  --weights '.\models\pytorch\yolo26s-depth.pt' `
  --engine '.\models\tensorrt\yolo26s-depth-fp16-b1-768.engine' `
  --image '.\sample_data\bus.jpg' `
  --imgsz 768 `
  --output '.\depth_pt_vs_trt.json'
```

`diagnose_yolo26_depth_export.py` 还可以在有导出中间 ONNX 文件时，对完全相同输入的
PT、ONNX 和 TensorRT 原始输出做逐像素比较。该可选诊断需要额外安装
`requirements-validation.txt`，日常推理不需要 ONNX Runtime。

## 已验证的有效优化

1. TensorRT runtime、engine 和 execution context 常驻，不逐帧反序列化。
2. GPU 输入、输出和 pinned host 缓冲区预分配。
3. 使用 `execute_async_v3` 和独立 CUDA stream。
4. 将上传、归一化、TensorRT 推理及任务后处理捕获到 CUDA Graph。
5. 深度任务在 GPU 上裁掉 letterbox padding，并恢复到原图大小，只在需要时异步回传 CPU。
6. 分割任务固定处理置信度最高的 `max_masks` 个候选，避免动态索引带来的同步。
7. 多流吞吐测试中 4 contexts 优于 8 contexts；单路使用 1 context，多路优先测试 4 contexts。

没有纳入正式交付的方案：

- 单独 FP16 I/O 没有稳定提速，因此 engine 继续使用内部 FP16、FP32 I/O。
- 深度 FP32 engine 没有必要：统一预处理后，FP16 数值误差已经远低于先前的伪差异；
  再增加一个更大、更慢的 engine 不改善当前问题。
- `fast-gpu` 会产生轻微插值差异，在本机也没有优于默认 `exact-cpu`，只保留作实验选项。

详细数据见 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)。

## 校验文件

在解压目录执行：

```powershell
Get-Content -LiteralPath '.\MANIFEST.sha256' -Encoding UTF8
Get-FileHash -Algorithm SHA256 '.\models\pytorch\*.pt', '.\models\tensorrt\*.engine'
```

模型哈希必须与 `MANIFEST.sha256` 一致。压缩包本身另有 `.sha256` 文件，并在 GitHub
Release 资产信息中记录。

## 精度边界

本包验证了转换、单图数值一致性与运行链路，不等于完成业务数据集精度验收。实例分割
engine 相对 PyTorch 仍观察到置信度漂移；正式研究或生产使用前必须在目标数据集重新计算
box/mask mAP。深度模型也应在带真值的目标数据集计算 AbsRel、RMSE 和 δ 指标，并确认
摄像头成像域与模型训练域匹配。
