[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$YoloExe = 'E:\Anaconda_envs\envs\yolo\Scripts\yolo.exe',
    [string]$ModelPath,
    [string]$DataYaml,
    [string]$ProjectDir,
    [string]$RunName = 'yolov8n_640',
    [ValidateRange(32, 4096)]
    [int]$ImageSize = 640,
    [ValidateRange(1, 1024)]
    [int]$BatchSize = 16,
    [ValidateRange(1, 10000)]
    [int]$Epochs = 200,
    [int]$Device = 0,
    [ValidateRange(0, 128)]
    [int]$Workers = 0
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'scripts\project_paths.ps1')

$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot
if (-not $ModelPath) {
    $ModelPath = Join-Path $PSScriptRoot 'Model_Traning\weights\yolov8\yolov8n.pt'
}
if (-not $DataYaml) {
    $DataYaml = Join-Path $workspace 'Datasets\bottle\bottle_plastic.yaml'
}
if (-not $ProjectDir) {
    $ProjectDir = Join-Path $workspace 'Training_runs\bottle'
}

$runDir = Join-Path $ProjectDir $RunName
$logPath = Join-Path $ProjectDir ("{0}_console.log" -f $RunName)

foreach ($requiredFile in @($YoloExe, $ModelPath, $DataYaml)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}
if (Test-Path -LiteralPath $runDir) {
    throw "Run directory already exists. Use a new -RunName or resume explicitly: $runDir"
}

New-Item -ItemType Directory -Path $ProjectDir -Force | Out-Null
$env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
Set-Location -LiteralPath $PSScriptRoot

"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting $RunName." |
    Tee-Object -FilePath $logPath -Append

& $YoloExe detect train `
    "model=$ModelPath" `
    "data=$DataYaml" `
    "imgsz=$ImageSize" `
    "batch=$BatchSize" `
    "epochs=$Epochs" `
    "device=$Device" `
    "workers=$Workers" `
    "project=$ProjectDir" `
    "name=$RunName" 2>&1 | Tee-Object -FilePath $logPath -Append

$trainExitCode = $LASTEXITCODE
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] YOLO exited with code $trainExitCode." |
    Tee-Object -FilePath $logPath -Append

exit $trainExitCode
