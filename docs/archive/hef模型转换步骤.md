# HEF 模型转换步骤（历史记录）

> 本文保留早期 PingPong、旧模型路径和转换排查记录。当前转换流程请使用
> [`Model_Conversion/README.md`](../../Model_Conversion/README.md) 和
> [`Model_Conversion/Hailo/README.md`](../../Model_Conversion/Hailo/README.md)。

本文档记录当前 Windows + WSL2 Ubuntu-22.04 + Docker + Hailo AI Software Suite 的 YOLO `.pt` 转 `.hef` 流程。

## 1. 常用路径

```text
Windows YOLO 环境:
E:\Anaconda_envs\envs\yolo

Ultralytics 配置目录:
C:\Users\MECHREU\Desktop\树莓派5\ultralytics_config

WSL Hailo 工作目录:
~/hailo-work

Hailo 容器工作目录:
/workspace

校准图片来源:
E:\YOLO\merged\test\images

树莓派 5:
a@192.168.5.115
```

## 2. 启动或创建 Hailo Docker 容器

在 Ubuntu-22.04 WSL 里执行。

查看已有容器：

```bash
docker ps -a
```

启动已有容器：

```bash
docker start -ai hailo-suite
```

如果容器不存在，创建普通容器：

```bash
docker run -it --name hailo-suite \
  -v ~/hailo-work:/workspace \
  hailo8_ai_sw_suite_2025-10:1 \
  bash
```

如果要让容器能看到 NVIDIA GPU：

```bash
docker run -it --gpus all --name hailo-suite \
  -v ~/hailo-work:/workspace \
  hailo8_ai_sw_suite_2025-10:1 \
  bash
```

删除旧容器：

```bash
docker rm -f hailo-suite
```

## 3. 准备校准集

在 Ubuntu-22.04 WSL 里执行。

```bash
mkdir -p ~/hailo-work/calib_images
cp /mnt/e/YOLO/merged/test/images/* ~/hailo-work/calib_images/
ls ~/hailo-work/calib_images | wc -l
```

当前校准集约 298 张图片。

## 4. 从 PT 导出 ONNX

在 Windows PowerShell 里执行。

640 输入：

```powershell
$env:YOLO_CONFIG_DIR='C:\Users\MECHREU\Desktop\树莓派5\ultralytics_config'
& 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe' export model='E:\YOLO\PingPong\YOLOV8.pt' format=onnx imgsz=640 opset=11 simplify=True
```

320 输入：

```powershell
$env:YOLO_CONFIG_DIR='C:\Users\MECHREU\Desktop\树莓派5\ultralytics_config'
& 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe' export model='E:\YOLO\PingPong\YOLOV8.pt' format=onnx imgsz=320 opset=11 simplify=True
```

如果转换其它模型，只改 `model='...'` 和后续文件名。

## 5. 复制 ONNX 到 WSL 工作目录

在 Ubuntu-22.04 WSL 里执行。

```bash
cp /mnt/e/YOLO/PingPong/YOLOV8.onnx ~/hailo-work/
ls -lh ~/hailo-work
```

## 6. 在 Hailo 容器里编译 HEF

进入 Hailo 容器后执行。

```bash
cd /workspace
chmod -R a+rwX /workspace
```

测试 Hailo DFC：

```bash
python -c "from hailo_sdk_client import ClientRunner; print('Hailo DFC OK')"
```

编译 YOLOv8n / YOLOv8 类结构模型：

```bash
hailomz compile yolov8n \
  --ckpt /workspace/YOLOV8.onnx \
  --calib-path /workspace/calib_images \
  --hw-arch hailo8l \
  --classes 1 \
  --performance
```

说明：

```text
--hw-arch hailo8l   Raspberry Pi AI Kit / AI HAT+ 常用 Hailo-8L
--classes 1         乒乓球模型只有 1 类时使用 1
--performance       启用性能优化
```

编译完成后确认输出：

```bash
ls -lh /workspace/*.hef /workspace/*.har
find /workspace -name "*.hef" -o -name "*.har"
```

成功时会看到类似：

```text
HEF file written to yolov8n.hef
Saved HAR to: /workspace/yolov8n.har
```

## 7. YOLOv5 注意事项

当前 `E:\YOLO\PingPong\YOLOV5.pt` 实际检测头包含 `cv2/cv3/dfl`，更像 YOLOv8 类结构，不是传统 YOLOv5 Detect 头。

因此它转换失败时不要强行用：

```bash
hailomz compile yolov5s ...
```

优先按 YOLOv8 类结构编译：

```bash
hailomz compile yolov8n \
  --ckpt /workspace/YOLOV5.onnx \
  --calib-path /workspace/calib_images \
  --hw-arch hailo8l \
  --classes 1 \
  --performance
```

为了避免输出文件名覆盖，可以在编译前备份旧结果：

```bash
mv /workspace/yolov8n.hef /workspace/YOLOV8_yolov8n.hef 2>/dev/null || true
mv /workspace/yolov8n.har /workspace/YOLOV8_yolov8n.har 2>/dev/null || true
```

编译后重命名：

```bash
mv /workspace/yolov8n.hef /workspace/YOLOV5_yolov8n.hef
mv /workspace/yolov8n.har /workspace/YOLOV5_yolov8n.har
```

## 8. 复制 HEF 到 Windows 桌面

先退出容器：

```bash
exit
```

在 Ubuntu-22.04 WSL 里执行：

```bash
cp ~/hailo-work/yolov8n.hef /mnt/c/Users/MECHREU/Desktop/
```

复制到当前工作区：

```bash
cp ~/hailo-work/yolov8n.hef "/mnt/c/Users/MECHREU/Desktop/树莓派5/"
```

## 9. 复制 HEF 到树莓派 5

在 Ubuntu-22.04 WSL 里执行：

```bash
scp -i /mnt/c/Users/MECHREU/.ssh/id_ed25519 ~/hailo-work/yolov8n.hef a@192.168.5.115:/home/a/
```

测试树莓派 SSH：

```bash
ssh -i /mnt/c/Users/MECHREU/.ssh/id_ed25519 a@192.168.5.115 "hostname; whoami; uname -a"
```

## 10. 清理转换产物

在容器里删除 HEF/HAR：

```bash
rm -f /workspace/*.hef /workspace/*.har
```

在 WSL 里删除 HEF/HAR：

```bash
rm -f ~/hailo-work/*.hef ~/hailo-work/*.har
```

不要误删：

```text
YOLOV5.pt
YOLOV5.onnx
YOLOV8.pt
YOLOV8.onnx
calib_images/
```

## 11. 常见问题

### yolo: command not found

Hailo 容器里不要安装 Ultralytics。请在 Windows 主机的 conda `yolo` 环境里导出 ONNX，Hailo 容器只做 ONNX 到 HEF。

### Permission denied: yolov8n.har

`/workspace` 没有写权限。退出容器后在 WSL 执行：

```bash
chmod -R a+rwX ~/hailo-work
```

再进入容器重新编译。

### No GPU chosen and no suitable GPU found

容器没有使用 GPU 或容器内 CUDA/cuDNN 不满足 Hailo DFC 推荐版本。CPU 也可以编译，只是更慢。

### 198.18.0.x 或 .local 解析异常

可能是 Clash / TUN / Fake-IP 影响。当前树莓派 5 已验证 IP 是：

```text
192.168.5.115
```
