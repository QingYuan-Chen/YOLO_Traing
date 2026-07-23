[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$ModelPath,
    [string]$DataYaml,
    [string]$ProjectDir,
    [string]$RunName = 'yolov8n_640_manual',
    [int]$Epochs = 300,
    [int]$ImageSize = 640,
    [int]$BatchSize = 32,
    [int]$Device = 0,
    [int]$Workers = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$launcher = Join-Path $repoRoot 'start_yolov8n_640.ps1'

$arguments = @{
    WorkspaceRoot = $WorkspaceRoot
    YoloExe = $YoloExe
    RunName = $RunName
    Epochs = $Epochs
    ImageSize = $ImageSize
    BatchSize = $BatchSize
    Device = $Device
    Workers = $Workers
}
if ($ModelPath) { $arguments.ModelPath = $ModelPath }
if ($DataYaml) { $arguments.DataYaml = $DataYaml }
if ($ProjectDir) { $arguments.ProjectDir = $ProjectDir }

& $launcher @arguments
exit $LASTEXITCODE
