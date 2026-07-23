[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$Weights,
    [string]$OutputPath,
    [int]$ImageSize = 640,
    [int]$Opset = 17,
    [switch]$Simplify
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $Weights) {
    $Weights = Join-Path $workspace 'Training_runs\bottle\yolov8n_640\weights\best.pt'
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $workspace 'Module_conversion\bottle\yolov8n.onnx'
}

$exportScript = Join-Path $repoRoot 'Model_Conversion\export_pt_to_onnx.ps1'
$arguments = @{
    ModelPath = $Weights
    OutputPath = $OutputPath
    ImageSize = $ImageSize
    Opset = $Opset
    WorkspaceRoot = $workspace
}
if ($Simplify) {
    $arguments.Simplify = $true
}

& $exportScript @arguments
exit $LASTEXITCODE
