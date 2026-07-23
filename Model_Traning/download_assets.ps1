param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("wheels", "weights")]
    [string]$Target,

    [string]$Repo = "QingYuan-Chen/Yolo_Traing",
    [string]$ReleaseTag,
    [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir '..')).Path
. (Join-Path $RepoRoot 'scripts\project_paths.ps1')
$Workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot
$ExternalAssetRoot = Join-Path $Workspace "Dependencies"
$TempDir = Join-Path $env:TEMP "yolo-asset-download"

New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

function Download-Asset {
    param(
        [string]$BaseUrl,
        [string]$Name
    )

    $targetPath = Join-Path $TempDir $Name
    $url = "$BaseUrl/$Name"

    Write-Host "Downloading $Name"
    Invoke-WebRequest -Uri $url -OutFile $targetPath
    return $targetPath
}

if ($Target -eq "wheels") {
    if (-not $ReleaseTag) {
        $ReleaseTag = "wheels-cu128"
    }

    $BaseUrl = "https://github.com/$Repo/releases/download/$ReleaseTag"
    $WheelDir = Join-Path $ExternalAssetRoot "wheels"
    $TorchName = "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"
    $TorchPath = Join-Path $WheelDir $TorchName

    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null

    $nonTorchZip = Download-Asset -BaseUrl $BaseUrl -Name "wheels-non-torch.zip"
    Expand-Archive -LiteralPath $nonTorchZip -DestinationPath $WheelDir -Force

    Remove-Item -LiteralPath $TorchPath -Force -ErrorAction SilentlyContinue
    $partNames = @(
        "$TorchName.part01",
        "$TorchName.part02"
    )

    $outputStream = [System.IO.File]::Create($TorchPath)
    try {
        foreach ($partName in $partNames) {
            $partPath = Download-Asset -BaseUrl $BaseUrl -Name $partName
            $inputStream = [System.IO.File]::OpenRead($partPath)
            try {
                $inputStream.CopyTo($outputStream)
            }
            finally {
                $inputStream.Dispose()
            }
        }
    }
    finally {
        $outputStream.Dispose()
    }

    Write-Host ""
    Write-Host "Wheels downloaded to: $WheelDir"
    Write-Host "Offline install example:"
    Write-Host "pip install --no-index --find-links `"$WheelDir`" -r `"$ScriptDir\requirements.txt`""
    exit 0
}

if (-not $ReleaseTag) {
    $ReleaseTag = "weights-pretrained"
}

$BaseUrl = "https://github.com/$Repo/releases/download/$ReleaseTag"
$ZipPath = Download-Asset -BaseUrl $BaseUrl -Name "weights.zip"

Expand-Archive -LiteralPath $ZipPath -DestinationPath $ScriptDir -Force

Write-Host ""
Write-Host "Weights downloaded to: $(Join-Path $ScriptDir "weights")"
