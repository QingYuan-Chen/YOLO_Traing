[CmdletBinding()]
param(
    [string]$EnvironmentName = 'yolo',
    [string]$PythonVersion = '3.12',
    [string]$TorchVersion = '2.8.0',
    [string]$TorchVisionVersion = '0.23.0',
    [string]$UltralyticsVersion = '8.4.60',
    [string]$OnnxVersion = '1.17.0',
    [string]$OnnxRuntimeVersion = '1.26.0'
)

$ErrorActionPreference = 'Stop'
$env:CONDA_NO_PLUGINS = 'true'
$env:CONDA_NUMBER_CHANNEL_NOTICES = '0'

$environmentList = conda env list --json | ConvertFrom-Json
$environmentExists = $environmentList.envs | Where-Object {
    (Split-Path -Leaf $_) -eq $EnvironmentName
}

if (-not $environmentExists) {
    conda create -n $EnvironmentName "python=$PythonVersion" -y
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create Conda environment: $EnvironmentName"
    }
}

conda run -n $EnvironmentName python -m pip install --upgrade pip
conda run -n $EnvironmentName python -m pip install `
    "torch==$TorchVersion" `
    "torchvision==$TorchVisionVersion" `
    --index-url https://download.pytorch.org/whl/cu128
conda run -n $EnvironmentName python -m pip install `
    "ultralytics==$UltralyticsVersion" `
    "onnx==$OnnxVersion" `
    "onnxruntime==$OnnxRuntimeVersion"

conda run -n $EnvironmentName python -c @'
import onnx
import onnxruntime
import torch
import ultralytics

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
'@

if ($LASTEXITCODE -ne 0) {
    throw 'Environment import check failed.'
}
