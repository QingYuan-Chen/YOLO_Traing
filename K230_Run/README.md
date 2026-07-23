# K230 板端示例

本目录保存 CanMV/K230 Python 推理示例。板端脚本必须和 KModel 的输入、输出、预处理及后处理严格匹配，不能仅根据 `yolov5.py` 这个文件名判断模型兼容性。

## 文件定位

| 文件 | 当前定位 |
|---|---|
| `yolov5.py` | 已验证转换产物对应的经典 YOLOv5n 320 塑料瓶检测脚本 |
| `yolov8.py` | 早期 `ball` 实验脚本，仅保留作历史参考 |
| `yolov12.py` | 早期 `ball` 实验脚本，仅保留作历史参考 |

## YOLOv5n 模型契约

`yolov5.py` 默认只适配以下 KModel：

| 项目 | 值 |
|---|---|
| KModel 路径 | `/sdcard/yolov5n_bottle_320_k230.kmodel` |
| 输入 | `uint8 [1, 3, 320, 320]`，NCHW |
| 输出 | `float32 [1, 6300, 6]` |
| 参考文件 SHA-256 | `36F50EC97A9E1EF2AFD8BB0625CC6E1E8B6C388BD706F74F04A02385EA00920A` |
| 类别 | `plastic-bottle` |
| 置信度阈值 | `0.50` |
| NMS 阈值 | `0.45` |
| 后处理 | `aidemo.yolov5_det_postprocess` |

输出最后一维的 6 个值对应 `x, y, w, h, objectness, class_score`。如果换成多类别模型，最后一维应为 `5 + 类别数`，同时必须更新脚本顶部的 `LABELS`。

该模型的预处理采用：

- RGB888 planar 输入。
- AI2D 输入和输出均为 `uint8`；KModel 推理输出为 `float32`。
- `tf_bilinear + half_pixel` 缩放。
- 填充值 `[128, 128, 128]`。
- 图像放在左上角，只在右侧和下侧补边。

最后一项看起来不同于常见的居中 Letterbox，但它与 CanMV 的 `letterbox_pad_param` 和 `aidemo.yolov5_det_postprocess` 坐标还原方式一致。以 `1920×1080 → 320×320` 为例，padding 应为 `top=0, bottom=140, left=0, right=0`。不要单独把它改成上下各补 70，否则检测框可能发生纵向偏移。

## 上板运行

### 1. 准备文件

根据 [K230 KModel 转换说明](../Model_Conversion/K230/README.md)生成模型，并将以下文件放到 SD 卡：

```text
/sdcard/yolov5n_bottle_320_k230.kmodel
/sdcard/yolov5.py
```

复制前可在 Windows PowerShell 中核对参考模型身份：

```powershell
Get-FileHash .\yolov5n_bottle_320_k230.kmodel -Algorithm SHA256
```

`libs.PipeLine`、`libs.AIBase`、`libs.AI2D`、`nncase_runtime` 和 `aidemo` 由支持 AI 示例的 CanMV 固件提供，不需要从本仓库复制。若导入失败，应先核对固件版本和固件内置资源。

### 2. 修改显示方式

脚本顶部默认使用 LCD：

```python
DISPLAY_MODE = "lcd"
DISPLAY_SIZE = [640, 480]
```

使用 HDMI 时改为：

```python
DISPLAY_MODE = "hdmi"
DISPLAY_SIZE = [1920, 1080]
```

摄像头输入、模型路径、类别、阈值和调试输出也集中在脚本顶部。`DEBUG_MODE = 1` 会逐阶段打印耗时，正常运行建议保持为 `0`，避免串口输出影响帧率。

### 3. 启动脚本

可在 CanMV IDE 中打开并运行 `yolov5.py`，也可将它设置为板端启动脚本。首次推理前后应看到类似日志：

```text
[KPU] loaded: /sdcard/yolov5n_bottle_320_k230.kmodel
[KPU] expected input=uint8 [1, 3, 320, 320], output=float32 [1, 6300, 6]
[AI2D] source=1920x1080, model=320x320, scale=0.166667, padding=(top 0, bottom 140, left 0, right 0)
[KPU] output[0] shape verified: (1, 6300, 6)
```

脚本会在初始化前检查模型文件，在第一次推理时检查输出张量形状。检查失败时会直接给出模型、输入尺寸或类别不匹配的提示，避免把错误结果继续送入 C 后处理。

## 更换自己的模型

至少同步修改并核对：

1. `KMODEL_PATH`。
2. `MODEL_INPUT_SIZE`。
3. `LABELS`，顺序必须和训练数据集一致。
4. `EXPECTED_PREDICTIONS` 和模型真实输出。
5. 输入布局、RGB/BGR、数据类型、量化范围和 Letterbox 方式。
6. 检测头是否仍是 `aidemo.yolov5_det_postprocess` 支持的经典 YOLOv5 解码输出。

如果输出不是单个 `[1, N, 5 + 类别数]` 张量，不要绕过脚本的契约检查；应先按模型真实输出重新实现后处理。

## 常见故障

| 现象 | 优先检查 |
|---|---|
| `KModel not found` | SD 卡路径、文件名和挂载状态 |
| `expected 1 input and 1 output` | 是否误用了其他导出结构或其他 YOLO 代际模型 |
| `output mismatch` | KModel 身份、输入尺寸、类别数和导出方式 |
| 能出框但位置偏移 | 是否改成了居中 Letterbox，显示尺寸是否传递正确 |
| 类别名称错误 | `LABELS` 顺序是否与训练配置一致 |
| 无目标或误检多 | 量化数据、RGB/BGR、阈值、数据域和训练质量 |
| `aidemo` 没有对应函数 | CanMV 固件版本或固件 AI 扩展不匹配 |
| 帧率低且串口刷屏 | 将 `DEBUG_MODE` 设为 `0` |

模型能够加载只证明 KPU 可以读取文件，不等于检测链路正确。上板后仍应分别验证空场景、单目标、多目标、画面边缘目标和不同尺度目标，并保存失败画面及串口日志。

## 相关说明

- [PT/ONNX 转换总览](../Model_Conversion/README.md)
- [K230 KModel 转换](../Model_Conversion/K230/README.md)
- [CanMV-K230 官方 YOLO 实现](https://github.com/kendryte/canmv_k230/blob/canmv_k230/resources/libs/YOLO.py)
- [CanMV K230 官方教程](https://github.com/kendryte/k230_docs/blob/main/en/CanMV_K230_Tutorial.md)

当前仓库完成的是脚本静态检查、模型契约核对和转换产物身份核对；没有在本次整理中连接实体 K230 开发板，因此实际摄像头、显示、性能和检测精度仍需板端验证。
