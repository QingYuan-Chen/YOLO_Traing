# YOLO26 实例分割 TensorRT 优化部署

本目录记录已经在 Windows + NVIDIA GeForce RTX 5060 Laptop GPU 上完成的
YOLO26s-seg / YOLO26m-seg TensorRT 转换和推理优化方案。模型权重与引擎不写入
Git 历史，统一放在 GitHub Release 资产中。

## 下载

Release：
[yolo26-seg-trt-rtx5060-20260822](https://github.com/QingYuan-Chen/YOLO_Traing/releases/tag/yolo26-seg-trt-rtx5060-20260822)

下载并解压 `YOLO26-seg-TensorRT-RTX5060-Windows-20260822.zip` 后，目录应为：

```text
YOLO26-seg-TensorRT-RTX5060-Windows-20260822/
├─ README_CN.md
├─ BENCHMARK_RESULTS.md
├─ ENVIRONMENT.json
├─ LICENSE_NOTICE.md
├─ MANIFEST.sha256
├─ requirements-runtime.txt
├─ models/
│  ├─ pytorch/
│  │  ├─ yolo26s-seg.pt
│  │  └─ yolo26m-seg.pt
│  └─ tensorrt/
│     ├─ yolo26s-seg-fp16-b1-640.engine
│     └─ yolo26m-seg-fp16-b1-640.engine
├─ scripts/
│  ├─ infer_yolo26_trt_optimized.py
│  ├─ benchmark_yolo26_trt_native.py
│  └─ benchmark_yolo26_trt_concurrency.py
├─ sample_data/
│  └─ bus.jpg
└─ results/
```

## 兼容性边界

TensorRT engine 不是通用模型格式。这里的 engine 使用 TensorRT
10.13.3.9.post1、CUDA 12.8、静态 batch 1、640 x 640 输入和内部 FP16 构建，
只在 RTX 5060 Laptop GPU 上做过实机验证。不同 GPU 架构、TensorRT 版本、驱动或
操作系统可能无法反序列化；遇到这种情况应使用包内 `.pt` 权重在目标机器重新构建，
不要把 engine 加载失败误判为权重损坏。

当前优化推理器还有两个静态约束：

- CUDA Graph 捕获后，输入图像的 HWC 分辨率必须保持不变；分辨率变化时重新创建并
  捕获 `OptimizedSegmentationPipeline`。
- `--max-masks` 是捕获时固定分配的 mask 数量，默认 20。场景实例更多时需要增大；
  场景实例较少时可减小以降低 CPU 回传量。

## 环境准备

本次验证环境见 `ENVIRONMENT.json`。全新 Python 3.12 虚拟环境可尝试：

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements-runtime.txt'
```

TensorRT Python 依赖包含约 1.5 GB 的运行库，完整虚拟环境本次实测约 2.17 GiB；
安装前请预留至少 4 GiB 空间。若接收方已有匹配版本的 CUDA、PyTorch 和 TensorRT，
优先复用已有环境。

安装后检查：

```powershell
& '.\.venv\Scripts\python.exe' -c "import torch,tensorrt; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), tensorrt.__version__)"
& '.\.venv\Scripts\python.exe' -m pip check
```

## 推荐推理命令

以下命令在解压目录内运行。`exact-cpu` 使用 OpenCV letterbox，已验证输入张量与
Ultralytics/OpenCV 基准逐像素一致，同时在本机比 Torch GPU resize 更快。

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\scripts\infer_yolo26_trt_optimized.py' `
  '.\models\tensorrt\yolo26s-seg-fp16-b1-640.engine' `
  '.\your_image.jpg' `
  --preprocess exact-cpu `
  --confidence 0.25 `
  --max-masks 20 `
  --iterations 300 `
  --save-overlay '.\s_result.jpg'
```

使用 m 模型时只替换 engine 文件名：

```text
models\tensorrt\yolo26m-seg-fp16-b1-640.engine
```

包内示例图可直接替换 `your_image.jpg`：

```text
sample_data\bus.jpg
```

优化推理器输出的是 640 x 640 letterbox 模型坐标和 mask。如果业务需要原始图像
坐标，应在消费端按 letterbox 比例和 padding 做反变换。

## 优化方案

已保留的有效优化：

1. TensorRT runtime、engine 和 execution context 常驻，不逐帧反序列化。
2. GPU 输入、输出和 pinned host 缓冲区预分配。
3. 使用 `execute_async_v3` 和独立 CUDA stream。
4. 将上传、归一化、TensorRT 推理、置信度过滤、mask 矩阵乘法、裁剪和上采样
   捕获到一个 CUDA Graph。
5. YOLO26 end-to-end 输出按置信度降序，固定处理前 `max_masks` 个候选，避免动态
   boolean indexing 引起的 GPU/CPU 同步。
6. CPU 回传使用固定 pinned buffers 和 non-blocking copy。

未采纳的实验：

- FP16 I/O binding 没有稳定提速，因此 Release 只包含内部 FP16、FP32 I/O 的
  已验证 engine。
- 8 路并发吞吐低于 4 路，因此单相机使用 1 个 context，多路视频最多优先测试
  4 个 context。
- `fast-gpu` Torch resize 与 OpenCV 存在细微像素差异，且本机略慢于
  `exact-cpu`，所以不作为默认方案。

详细数据见 [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)。

## 校验文件

在解压目录执行：

```powershell
Get-Content '.\MANIFEST.sha256'
Get-FileHash -Algorithm SHA256 '.\models\pytorch\*.pt', '.\models\tensorrt\*.engine'
```

四个模型文件的 SHA-256 必须与 `MANIFEST.sha256` 一致。压缩包本身另有同名
`.sha256` 文件，并在 GitHub Release 资产信息中记录。

## 精度边界

本次已经验证：

- `exact-cpu` 输入与 Ultralytics/OpenCV 输入张量最大绝对误差为 0。
- GPU mask 后处理与 `ultralytics.utils.ops.process_mask` 对同一 engine 输出
  逐像素一致，mask IoU 为 1.0。

但 TensorRT engine 相对 `.pt` 权重仍存在置信度漂移。bus 图上 s engine 在
`conf=0.25` 时为 4 个实例，而 PyTorch 权重曾得到 5 个实例；m engine 为 5 个，
但置信度也有漂移。因此本包证明的是转换和运行链路成功，不代表部署精度已经验收。
正式研究或生产使用前，必须在目标数据集上重新计算 box mAP 和 mask mAP。
