[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [ValidateRange(32, 4096)]
    [int]$ImageSize = 640,

    [ValidateRange(7, 21)]
    [int]$Opset = 17,

    [switch]$Simplify,

    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$PythonExe = 'E:\Anaconda_envs\envs\yolo\python.exe',
    [string]$WorkspaceRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

$model = (Resolve-Path -LiteralPath $ModelPath).Path
if ([IO.Path]::GetExtension($model) -ne '.pt') {
    throw "Model must be a .pt file: $model"
}
if (-not (Test-Path -LiteralPath $YoloExe -PathType Leaf)) {
    throw "Ultralytics yolo.exe not found: $YoloExe"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python not found: $PythonExe"
}

$output = [IO.Path]::GetFullPath($OutputPath)
if ([IO.Path]::GetExtension($output) -ne '.onnx') {
    throw "OutputPath must end with .onnx: $output"
}
if (Test-Path -LiteralPath $output) {
    throw "Output already exists: $output"
}

$stageRoot = Join-Path $env:TEMP ('yolo-onnx-export-' + [guid]::NewGuid().ToString('N'))
$stageModel = Join-Path $stageRoot ([IO.Path]::GetFileName($model))
$stageOnnx = [IO.Path]::ChangeExtension($stageModel, '.onnx')
$outputDirectory = Split-Path -Parent $output

try {
    New-Item -ItemType Directory -Path $stageRoot | Out-Null
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    Copy-Item -LiteralPath $model -Destination $stageModel

    $env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
    $simplifyValue = if ($Simplify) { 'True' } else { 'False' }

    & $YoloExe export `
        "model=$stageModel" `
        'format=onnx' `
        "imgsz=$ImageSize" `
        'batch=1' `
        "opset=$Opset" `
        "simplify=$simplifyValue" `
        'dynamic=False' `
        'device=cpu'

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $stageOnnx -PathType Leaf)) {
        throw 'Ultralytics ONNX export failed.'
    }

    & $PythonExe -c "import onnx,onnxruntime as ort,sys; p=sys.argv[1]; m=onnx.load(p); onnx.checker.check_model(m); s=ort.InferenceSession(p,providers=['CPUExecutionProvider']); print('IR',m.ir_version,'OPSET',[(x.domain,x.version) for x in m.opset_import]); print('INPUTS',[(x.name,x.shape,x.type) for x in s.get_inputs()]); print('OUTPUTS',[(x.name,x.shape,x.type) for x in s.get_outputs()])" $stageOnnx
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
