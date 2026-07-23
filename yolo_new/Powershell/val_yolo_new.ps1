[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$Weights,
    [string]$DataYaml,
    [int]$ImageSize = 640,
    [ValidateSet('train', 'val', 'test')]
    [string]$Split = 'test',
    [int]$Device = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $Weights) {
    $Weights = Join-Path $workspace 'Training_runs\bottle\yolov8n_640\weights\best.pt'
}
if (-not $DataYaml) {
    $DataYaml = Join-Path $workspace 'Datasets\bottle\bottle_plastic.yaml'
}
foreach ($requiredFile in @($YoloExe, $Weights, $DataYaml)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}

$env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
Set-Location -LiteralPath $repoRoot

& $YoloExe detect val `
    "model=$Weights" `
    "data=$DataYaml" `
    "imgsz=$ImageSize" `
    "split=$Split" `
    "device=$Device"

exit $LASTEXITCODE
