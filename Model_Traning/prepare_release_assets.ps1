param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("wheels", "weights")]
    [string]$Target,

    [string]$OutputDir,
    [int64]$PartSizeBytes = 1900MB,
    [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir '..')).Path
. (Join-Path $RepoRoot 'scripts\project_paths.ps1')
$Workspace = Resolve-YoloWorkspaceRoot -WorkspaceRoot $WorkspaceRoot
$ExternalAssetRoot = Join-Path $Workspace "Dependencies"

function Write-AssetTable {
    param([string]$Path)

    Get-ChildItem -LiteralPath $Path -File | Sort-Object Name |
        Select-Object Name,@{Name="MB";Expression={[Math]::Round($_.Length / 1MB, 2)}},@{Name="SHA256";Expression={(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}} |
        Format-Table -AutoSize
}

function Split-File {
    param(
        [string]$SourcePath,
        [string]$DestinationDir,
        [int64]$PartSize
    )

    $buffer = New-Object byte[] (8MB)
    $inputStream = [System.IO.File]::OpenRead($SourcePath)
    try {
        $partIndex = 1
        $baseName = Split-Path -Leaf $SourcePath

        while ($inputStream.Position -lt $inputStream.Length) {
            $partPath = Join-Path $DestinationDir ("{0}.part{1:D2}" -f $baseName, $partIndex)
            $outputStream = [System.IO.File]::Create($partPath)
            try {
                $written = 0L
                while ($written -lt $PartSize -and $inputStream.Position -lt $inputStream.Length) {
                    $remaining = [Math]::Min($buffer.Length, $PartSize - $written)
                    $read = $inputStream.Read($buffer, 0, [int]$remaining)
                    if ($read -le 0) {
                        break
                    }
                    $outputStream.Write($buffer, 0, $read)
                    $written += $read
                }
            }
            finally {
                $outputStream.Dispose()
            }
            $partIndex++
        }
    }
    finally {
        $inputStream.Dispose()
    }
}

if ($Target -eq "wheels") {
    if (-not $OutputDir) {
        $OutputDir = "$env:USERPROFILE\Desktop\yolo-wheel-release-assets"
    }

    $WheelDir = Join-Path $ExternalAssetRoot "wheels"
    $TorchName = "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"
    $TorchWheel = Join-Path $WheelDir $TorchName
    $NonTorchZip = Join-Path $OutputDir "wheels-non-torch.zip"

    if (-not (Test-Path -LiteralPath $WheelDir)) {
        throw "Wheel directory not found: $WheelDir"
    }
    if (-not (Test-Path -LiteralPath $TorchWheel)) {
        throw "Torch wheel not found: $TorchWheel"
    }

    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    Remove-Item -LiteralPath $NonTorchZip -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $OutputDir "$TorchName.part*") -Force -ErrorAction SilentlyContinue

    $NonTorchWheels = Get-ChildItem -LiteralPath $WheelDir -File -Filter "*.whl" |
        Where-Object { $_.Name -ne $TorchName }

    Compress-Archive -LiteralPath $NonTorchWheels.FullName -DestinationPath $NonTorchZip -CompressionLevel Optimal
    Split-File -SourcePath $TorchWheel -DestinationDir $OutputDir -PartSize $PartSizeBytes

    Write-AssetTable -Path $OutputDir
    Write-Host ""
    Write-Host "Upload these files to a GitHub Release named/tagged wheels-cu128."
    Write-Host "OutputDir: $OutputDir"
    exit 0
}

if (-not $OutputDir) {
    $OutputDir = "$env:USERPROFILE\Desktop\yolo-weight-release-assets"
}

$WeightDir = Join-Path $ScriptDir "weights"
$WeightsZip = Join-Path $OutputDir "weights.zip"

if (-not (Test-Path -LiteralPath $WeightDir)) {
    throw "Weight directory not found: $WeightDir"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Remove-Item -LiteralPath $WeightsZip -Force -ErrorAction SilentlyContinue

$TempRoot = Join-Path $env:TEMP ("yolo-weight-assets-" + [guid]::NewGuid().ToString("N"))
$TempWeights = Join-Path $TempRoot "weights"

try {
    New-Item -ItemType Directory -Force -Path $TempWeights | Out-Null
    Copy-Item -LiteralPath (Join-Path $WeightDir "*") -Destination $TempWeights -Recurse -Force
    Compress-Archive -LiteralPath $TempWeights -DestinationPath $WeightsZip -CompressionLevel Optimal
}
finally {
    Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-AssetTable -Path $OutputDir
Write-Host ""
Write-Host "Upload weights.zip to a GitHub Release named/tagged weights-pretrained."
Write-Host "OutputDir: $OutputDir"
