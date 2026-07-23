[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$LastWeights
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $LastWeights) {
    $LastWeights = Join-Path $workspace 'Training_runs\bottle\yolov8n_640\weights\last.pt'
}
foreach ($requiredFile in @($YoloExe, $LastWeights)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}

$env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
Set-Location -LiteralPath $repoRoot

& $YoloExe detect train "model=$LastWeights" 'resume=True'
exit $LASTEXITCODE
