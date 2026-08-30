# YOLO26 TensorRT 优化实测记录

实例分割测试日期为 2026-08-22，单目深度补测日期为 2026-08-30。固定环境为 Windows、
NVIDIA GeForce RTX 5060 Laptop GPU、TensorRT 10.13.3.9.post1、PyTorch
2.8.0+cu128。测试图片为 Ultralytics `bus.jpg`，batch 1。所有测试均经过 warmup；
GPU 时间使用 CUDA Event，主机延迟包含对应路径同步。

## YOLO26s-depth：PyTorch 与标准 TensorRT 接口

公平比较条件固定为 `imgsz=768`、`rect=False`、100 次计时。`rect=False` 很重要：静态
engine 使用 768 x 768 正方形输入，而 PyTorch 默认矩形最小填充会改变网络实际输入。

| 后端 | 端到端中位数 | FPS | Ultralytics 推理阶段 |
| --- | ---: | ---: | ---: |
| PyTorch FP32 | 13.550 ms | 73.8 | 10.602 ms |
| TensorRT 内部 FP16 | 6.342 ms | 157.7 | 2.741 ms |

标准 Ultralytics 端到端中位延迟提升为 **2.14 倍**。

原图大小深度图的 TensorRT 相对 PyTorch 数值差异：

| 指标 | 结果 |
| --- | ---: |
| MAE | 0.003161 m |
| RMSE | 0.004817 m |
| 平均绝对相对误差 | 0.0671% |
| δ1 一致率 | 100% |
| 最大绝对误差 | 0.037798 m |

原始 JSON：`results/depth_pt_vs_trt_ultralytics.json`。

## YOLO26s-depth：优化 CUDA Graph 链路

`exact-cpu` 完整 GPU Graph 包含上传、归一化、TensorRT 推理、去 letterbox padding，
以及恢复到 1080 x 810 原图大小的 float32 深度图。OpenCV CPU letterbox 在 Graph 外执行，
但“主机帧到结果”计时包含它。

| 测试路径 | 中位/平均延迟 | 等效 FPS |
| --- | ---: | ---: |
| CUDA Graph GPU 链路（平均） | 2.934 ms | 340.8 |
| 主机帧到 GPU 原图深度图（中位） | 3.428 ms | 291.7 |
| 主机帧到 CPU 原图深度图（中位） | 3.708 ms | 269.7 |

相对于 PyTorch 13.550 ms 的端到端中位数，推荐的主机帧到 GPU 深度图路径约提升
**3.95 倍**。`fast-gpu` 中位数为 3.386 ms，但会产生轻微插值差异，且整体没有形成
稳定优势，所以 `exact-cpu` 仍作为默认。

原始结果：

- `results/depth_optimized_exact.json`
- `results/depth_optimized_fast_gpu.json`
- `results/depth_diagnostic_pt_onnx_engine.json`
- `results/depth_preview.jpg`

## 深度转换偏差的根因验证

初次比较曾出现约 0.53 m MAE。对同一 768 x 768 输入的原始输出进一步比较后，确认导出
链路本身正常：

| 比较 | MAE | 平均绝对相对误差 | δ1 一致率 |
| --- | ---: | ---: | ---: |
| ONNX vs PyTorch | 0.000511 m | 0.0115% | 100% |
| TensorRT FP16 vs PyTorch | 0.002544 m | 0.0708% | 100% |

因此，大偏差来自 PyTorch 与静态 engine 使用了不同 letterbox 形状，而不是 FP16 精度
损失。统一正方形预处理后，最终原图结果也保持一致。基于这个证据，没有额外构建 FP32
engine。

## YOLO26 实例分割：推荐 `exact-cpu` 链路

分割完整 GPU 链路包含 OpenCV letterbox 后的上传/归一化、TensorRT 推理、置信度过滤、
原型 mask 矩阵乘法、裁剪以及 20 个固定 640 x 640 mask 输出。

| 模型 | CUDA Graph GPU 链路 | 主机帧到 GPU 结果，中位数 | 固定 boxes/masks 回传 CPU，中位数 |
| --- | ---: | ---: | ---: |
| YOLO26s-seg | 2.484 ms / 402.6 FPS | 2.819 ms | 3.693 ms |
| YOLO26m-seg | 5.533 ms / 180.7 FPS | 5.935 ms | 6.879 ms |

bus 图在 `conf=0.25` 下，s engine 得到 4 个实例，m engine 得到 5 个实例。

原始 JSON：

- `results/s_optimized_exact_final.json`
- `results/m_optimized_exact_final.json`

## 实例分割多 context 吞吐

下表是并发 batch-1 context 的 engine-only CUDA Graph 吞吐，不是 dynamic batch engine，
也不包含完整 mask 后处理。

| 模型 | 1 context | 4 contexts | 8 contexts |
| --- | ---: | ---: | ---: |
| YOLO26s-seg | 469.4 FPS | 542.3 FPS | 508.9 FPS |
| YOLO26m-seg | 177.5 FPS | 188.9 FPS | 175.5 FPS |

4 contexts 是这张 GPU 的吞吐甜点位；8 contexts 没有继续提升。原始 JSON：

- `results/s_concurrency_1_4_8.json`
- `results/m_concurrency_1_4_8.json`

## Engine I/O 契约

YOLO26s-depth：

- 输入 `images`：`(1, 3, 768, 768)`，FP32。
- 输出 `output0`：`(1, 1, 768, 768)`，FP32，单位为米。

YOLO26s/m-seg：

- 输入 `images`：`(1, 3, 640, 640)`，FP32。
- 输出 `output0`：`(1, 300, 38)`，FP32。
- 输出 `output1`：`(1, 32, 160, 160)`，FP32。

所有正式 engine 内部使用 FP16 tactic，但绑定保持 FP32。分割 FP16 I/O 实验没有稳定
改善，因此未纳入 Release。
