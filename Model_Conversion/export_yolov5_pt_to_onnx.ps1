[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [ValidateRange(32, 4096)]
    [int]$ImageSize = 320,

    [ValidateRange(7, 14)]
    [int]$Opset = 12,

    [string]$PythonExe = 'E:\Anaconda_envs\envs\yolov5gpu128\python.exe',
    [string]$ValidationPython = 'E:\Anaconda_envs\envs\yolo\python.exe',
    [string]$YoloV5Root
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $YoloV5Root) {
    $YoloV5Root = Join-Path $repoRoot 'yolov5_train\yolov5'
}

$model = (Resolve-Path -LiteralPath $ModelPath).Path
$output = [IO.Path]::GetFullPath($OutputPath)
if ([IO.Path]::GetExtension($model) -ne '.pt') {
    throw "Model must be a .pt file: $model"
}
if ([IO.Path]::GetExtension($output) -ne '.onnx') {
    throw "OutputPath must end with .onnx: $output"
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists: $output"
}

foreach ($required in @($PythonExe, $ValidationPython, (Join-Path $YoloV5Root 'export.py'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}

$stageRoot = Join-Path $env:TEMP ('yolov5-onnx-export-' + [guid]::NewGuid().ToString('N'))
$stageModel = Join-Path $stageRoot ([IO.Path]::GetFileName($model))
$stageOnnx = [IO.Path]::ChangeExtension($stageModel, '.onnx')

try {
    New-Item -ItemType Directory -Path $stageRoot | Out-Null
    New-Item -ItemType Directory -Path (Split-Path -Parent $output) -Force | Out-Null
    Copy-Item -LiteralPath $model -Destination $stageModel

    $env:YOLOV5_EXPORT_MODEL = $stageModel
    $env:YOLOV5_EXPORT_SIZE = [string]$ImageSize
    $env:YOLOV5_EXPORT_OPSET = [string]$Opset
    Push-Location -LiteralPath $YoloV5Root
    try {
        @'
import os
import torch

original_export = torch.onnx.export

def legacy_export(*args, **kwargs):
    kwargs["dynamo"] = False
    return original_export(*args, **kwargs)

torch.onnx.export = legacy_export

import export

size = int(os.environ["YOLOV5_EXPORT_SIZE"])
export.run(
    weights=os.environ["YOLOV5_EXPORT_MODEL"],
    imgsz=(size, size),
    batch_size=1,
    device="cpu",
    include=("onnx",),
    opset=int(os.environ["YOLOV5_EXPORT_OPSET"]),
    simplify=False,
)
'@ | & $PythonExe -
    }
    finally {
        Pop-Location
        Remove-Item Env:YOLOV5_EXPORT_MODEL -ErrorAction SilentlyContinue
        Remove-Item Env:YOLOV5_EXPORT_SIZE -ErrorAction SilentlyContinue
        Remove-Item Env:YOLOV5_EXPORT_OPSET -ErrorAction SilentlyContinue
    }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $stageOnnx -PathType Leaf)) {
        throw 'Classic YOLOv5 ONNX export failed.'
    }

    & $ValidationPython -c "import onnx,onnxruntime as ort,sys; p=sys.argv[1]; m=onnx.load(p); onnx.checker.check_model(m); s=ort.InferenceSession(p,providers=['CPUExecutionProvider']); print('IR',m.ir_version,'OPSET',[(x.domain,x.version) for x in m.opset_import]); print('INPUTS',[(x.name,x.shape,x.type) for x in s.get_inputs()]); print('OUTPUTS',[(x.name,x.shape,x.type) for x in s.get_outputs()])" $stageOnnx
    if ($LASTEXITCODE -ne 0) {
        throw 'ONNX validation failed.'
    }

    Copy-Item -LiteralPath $stageOnnx -Destination $output
    Get-Item -LiteralPath $output | Select-Object FullName, Length, LastWriteTime
    Get-FileHash -LiteralPath $output -Algorithm SHA256
}
finally {
    if (Test-Path -LiteralPath $stageRoot) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}
