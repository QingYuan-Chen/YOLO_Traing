[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$Weights,
    [string]$Source,
    [string]$ProjectDir,
    [string]$RunName = 'yolov8n_640',
    [int]$ImageSize = 640,
    [double]$Confidence = 0.25,
    [int]$Device = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $Weights) {
    $Weights = Join-Path $workspace 'Training_runs\bottle\yolov8n_640\weights\best.pt'
}
if (-not $Source) {
    $Source = Join-Path $workspace 'Datasets\bottle\Plastic Bottle 2.0.v39i.yolov8\test\images'
}
if (-not $ProjectDir) {
    $ProjectDir = Join-Path $workspace 'Training_runs\bottle\predictions'
}
foreach ($requiredPath in @($YoloExe, $Weights, $Source)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path not found: $requiredPath"
    }
}

$env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
Set-Location -LiteralPath $repoRoot

& $YoloExe detect predict `
    "model=$Weights" `
    "source=$Source" `
    "imgsz=$ImageSize" `
    "conf=$Confidence" `
    "device=$Device" `
    "project=$ProjectDir" `
    "name=$RunName" `
    'save=True'

exit $LASTEXITCODE
