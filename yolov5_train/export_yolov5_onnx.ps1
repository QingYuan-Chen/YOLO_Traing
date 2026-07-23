[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$Weights,
    [string]$OutputPath,
    [int]$ImageSize = 320,
    [int]$Opset = 12
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $Weights) {
    $Weights = Join-Path $workspace 'Training_runs\bottle\yolov5n_classic_320\weights\best.pt'
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $workspace 'Module_conversion\bottle\yolov5n_bottle_320_k230.onnx'
}

& (Join-Path $repoRoot 'Model_Conversion\export_yolov5_pt_to_onnx.ps1') `
    -ModelPath $Weights `
    -OutputPath $OutputPath `
    -ImageSize $ImageSize `
    -Opset $Opset

exit $LASTEXITCODE
