# ONNX 转 K230 KModel

本流程使用现有 Docker 容器 `k230-converter` 中的 nncase 2.9。Docker Desktop 必须已经启动。

转换完成后，使用仓库中的 [K230 YOLOv5n 板端脚本](../../K230_Run/README.md)进行摄像头、显示、输出张量和检测框验证。

## 输入要求

- 静态 NCHW 输入，batch 为 1。
- 当前经典 YOLOv5n 320 模型应从 PT 重新导出为 opset 12。
- 不要只修改 ONNX 的 `ir_version`；opset 17 仍超出该容器内旧 ONNX Runtime 的支持范围。
- 校准图片应来自真实验证集，脚本会等间隔抽取，且不会修改原数据集。

## 推荐：一条 PowerShell 命令

在 Windows PowerShell 中，从 `E:\YOLO` 执行：

```powershell
.\Model_Conversion\K230\compile_onnx_to_kmodel.ps1 `
  -OnnxPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov5n_bottle_320_k230.onnx' `
  -CalibrationImages 'C:\Users\MECHREU\Desktop\YOLOTraining\Datasets\bottle\Plastic Bottle 2.0.v39i.yolov8\valid\images' `
  -OutputPath 'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\yolov5n_bottle_320_k230.kmodel' `
  -InputWidth 320 `
  -InputHeight 320 `
  -CalibrationCount 10
```

脚本会完成 ONNX 检查、等间隔抽取校准图片、创建独立容器工作区、nncase 编译、KModel 模拟器加载检查、复制回 Windows、两侧 SHA-256 对比和临时工作区清理。

为保护已有模型，目标 `.kmodel` 已存在时脚本会停止；请改用新文件名，或确认后自行处理旧文件。

## 等价的容器核心命令

假设已经把 ONNX 放到 `/workspace/k230-conversion/model.onnx`，校准图片放到 `/workspace/k230-conversion/calibration`：

```powershell
docker start k230-converter

docker exec -w /workspace/k230-conversion k230-converter `
  python3 /home/user/model_converter/convert_model.py `
  --model /workspace/k230-conversion/model.onnx `
  --dataset_path /workspace/k230-conversion/calibration `
  --input_width 320 `
  --input_height 320 `
  --target k230 `
  --ptq_option 0
```

`--ptq_option 0` 对应 NoClip 的 uint8 PTQ。编译成功后，输出文件与 ONNX 同名，扩展名变为 `.kmodel`。

复制回 Windows：

```powershell
docker cp `
  k230-converter:/workspace/k230-conversion/model.kmodel `
  'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\model.kmodel'

Get-FileHash -Algorithm SHA256 `
  'C:\Users\MECHREU\Desktop\YOLOTraining\Module_conversion\bottle\model.kmodel'
```

## 验证要点

当前已成功转换的 YOLOv5n 320 参考模型：

```text
输入：uint8 [1, 3, 320, 320]
输出：float32 [1, 6300, 6]
SHA-256：36F50EC97A9E1EF2AFD8BB0625CC6E1E8B6C388BD706F74F04A02385EA00920A
```

如果编译报告 opset 不受支持，应重新运行上级目录的 `export_yolov5_pt_to_onnx.ps1`，不要对现有 ONNX 直接改版本号。
