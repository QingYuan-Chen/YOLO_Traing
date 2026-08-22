# YOLO26 TensorRT 优化实测记录

测试日期：2026-08-22。

固定环境：Windows、NVIDIA GeForce RTX 5060 Laptop GPU、640 x 640、batch 1、
TensorRT 10.13.3.9.post1、PyTorch 2.8.0+cu128。测试图片为 Ultralytics
`bus.jpg`。所有延迟均经过 warmup；GPU 时间使用 CUDA Event，主机延迟包含对应
路径的同步。

## 推荐 `exact-cpu` 完整分割链路

完整 GPU 链路包含 OpenCV letterbox 后的上传/归一化、TensorRT 推理、置信度
过滤、原型 mask 矩阵乘法、裁剪以及 20 个固定 640 x 640 mask 输出。

| 模型 | CUDA Graph GPU 链路 | 主机帧到 GPU 结果，中位数 | 固定 boxes/masks 回传 CPU，中位数 |
| --- | ---: | ---: | ---: |
| YOLO26s-seg | 2.484 ms / 402.6 FPS | 2.819 ms | 3.693 ms |
| YOLO26m-seg | 5.533 ms / 180.7 FPS | 5.935 ms | 6.879 ms |

bus 图在 `conf=0.25` 下，s engine 得到 4 个实例，m engine 得到 5 个实例。

原始 JSON：

- `results/s_optimized_exact_final.json`
- `results/m_optimized_exact_final.json`

## 多 context 吞吐

下表是并发 batch-1 context 的 engine-only CUDA Graph 吞吐，不是 dynamic batch
engine，也不包含完整 mask 后处理。

| 模型 | 1 context | 4 contexts | 8 contexts |
| --- | ---: | ---: | ---: |
| YOLO26s-seg | 469.4 FPS | 542.3 FPS | 508.9 FPS |
| YOLO26m-seg | 177.5 FPS | 188.9 FPS | 175.5 FPS |

4 contexts 是这张 GPU 的吞吐甜点位；8 contexts 没有继续提升。原始 JSON：

- `results/s_concurrency_1_4_8.json`
- `results/m_concurrency_1_4_8.json`

## 数值验证

- `exact-cpu` 预处理与 Ultralytics/OpenCV 输入张量逐像素相等，最大绝对误差 0。
- 固定候选 GPU mask 解码与 Ultralytics `process_mask` 对相同输出逐像素相等，
  两个 engine 的最小 mask IoU 都为 1.0。
- YOLO26 end-to-end engine 的 300 个候选置信度在验证图片上按降序排列。
- s/m engine 均可反序列化，I/O 为：
  - 输入 `images`：`(1, 3, 640, 640)`，FP32。
  - 输出 `output0`：`(1, 300, 38)`，FP32。
  - 输出 `output1`：`(1, 32, 160, 160)`，FP32。

engine 内部使用 FP16 tactic，但绑定仍为 FP32。单独重建 FP16 I/O s engine 后，
重复结果没有稳定改善，因此没有纳入 Release，也没有继续构建 m 版本。
