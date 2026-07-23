[CmdletBinding()]
param(
    [string]$WorkspaceRoot,
    [string]$PythonExe = 'E:\Anaconda_envs\envs\yolov5gpu128\python.exe',
    [string]$YoloV5Root,
    [string]$DataYaml,
    [string]$Weights,
    [string]$ProjectDir,
    [string]$RunName = 'yolov5n_classic_320',
    [int]$ImageSize = 320,
    [int]$BatchSize = 16,
    [int]$Epochs = 200,
    [int]$Device = 0,
    [int]$Workers = 0
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $repoRoot 'scripts\project_paths.ps1')
$workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot

if (-not $YoloV5Root) {
    $YoloV5Root = Join-Path $PSScriptRoot 'yolov5'
}
if (-not $DataYaml) {
    $DataYaml = Join-Path $workspace 'Datasets\bottle\bottle_plastic.yaml'
}
if (-not $Weights) {
    $Weights = Join-Path $repoRoot 'Model_Traning\weights\yolov5\yolov5n.pt'
}
if (-not $ProjectDir) {
    $ProjectDir = Join-Path $workspace 'Training_runs\bottle'
}

$trainScript = Join-Path $YoloV5Root 'train.py'
$runDir = Join-Path $ProjectDir $RunName
foreach ($requiredFile in @($PythonExe, $trainScript, $DataYaml, $Weights)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required file not found: $requiredFile"
    }
}
if (Test-Path -LiteralPath $runDir) {
    throw "Run directory already exists. Use a new -RunName or resume explicitly: $runDir"
}

$env:YOLO_CONFIG_DIR = Join-Path $workspace 'Config\ultralytics'
Set-Location -LiteralPath $YoloV5Root

& $PythonExe train.py `
    --img $ImageSize `
    --batch-size $BatchSize `
    --epochs $Epochs `
    --data $DataYaml `
    --weights $Weights `
    --workers $Workers `
    --device $Device `
    --project $ProjectDir `
    --name $RunName

exit $LASTEXITCODE
