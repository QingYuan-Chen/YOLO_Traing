[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [Parameter(Mandatory = $true)]
    [string]$CalibrationImages,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$WorkName,
    [ValidateRange(1, 10000)]
    [int]$CalibrationCount = 1024,
    [string]$Container = 'hailo-suite'
)

$ErrorActionPreference = 'Stop'

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "docker 命令失败，退出码：$LASTEXITCODE" }
}

$resolvedOnnx = (Resolve-Path -LiteralPath $OnnxPath).Path
$resolvedCalibration = (Resolve-Path -LiteralPath $CalibrationImages).Path
if (-not (Test-Path -LiteralPath $resolvedOnnx -PathType Leaf)) { throw "ONNX 文件不存在：$resolvedOnnx" }
if (-not (Test-Path -LiteralPath $resolvedCalibration -PathType Container)) { throw "校准图片目录不存在：$resolvedCalibration" }

$images = @(Get-ChildItem -LiteralPath $resolvedCalibration -File | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp)$' } | Sort-Object FullName)
if ($images.Count -eq 0) { throw "校准目录中没有支持的图片：$resolvedCalibration" }
$takeCount = [Math]::Min($CalibrationCount, $images.Count)
$selected = for ($i = 0; $i -lt $takeCount; $i++) {
    if ($takeCount -eq 1) { $index = 0 } else { $index = [Math]::Round($i * ($images.Count - 1) / ($takeCount - 1)) }
    $images[$index]
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop 未启动或 Docker 引擎不可用。' }
Invoke-Docker start $Container | Out-Null

$hailoUid = (& docker exec $Container id -u hailo).Trim()
if ($LASTEXITCODE -ne 0 -or $hailoUid -notmatch '^\d+$') {
    throw "无法读取容器内 hailo 用户的 UID：$hailoUid"
}
$hailoGid = (& docker exec $Container id -g hailo).Trim()
if ($LASTEXITCODE -ne 0 -or $hailoGid -notmatch '^\d+$') {
    throw "无法读取容器内 hailo 用户的 GID：$hailoGid"
}
$hailoOwner = "${hailoUid}:${hailoGid}"

$workDir = "/home/hailo/model-conversion/$WorkName"
$calibDir = "$workDir/calib_images"
$onnxName = [IO.Path]::GetFileName($resolvedOnnx)
$staging = Join-Path ([IO.Path]::GetTempPath()) ("hailo-calib-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    for ($i = 0; $i -lt $selected.Count; $i++) {
        $destination = Join-Path $staging ('calib_{0:D4}{1}' -f $i, $selected[$i].Extension.ToLowerInvariant())
        Copy-Item -LiteralPath $selected[$i].FullName -Destination $destination
    }

    Invoke-Docker exec -u root $Container bash -lc "rm -rf -- '$workDir' && mkdir -p '$calibDir' && chown -R '$hailoOwner' '$workDir'"
    Invoke-Docker cp $resolvedOnnx "${Container}:$workDir/$onnxName"
    Invoke-Docker cp "$staging/." "${Container}:$calibDir/"
    Invoke-Docker exec -u root $Container bash -lc "chown -R '$hailoOwner' '$workDir'"
    Invoke-Docker exec $Container bash -lc "test -r '$workDir/$onnxName' && test -w '$workDir'"
    $containerCalibrationFiles = @(& docker exec $Container find $calibDir -maxdepth 1 -type f)
    if ($LASTEXITCODE -ne 0) {
        throw '无法读取容器内校准图片列表。'
    }
    if ($containerCalibrationFiles.Count -ne $takeCount) {
        throw "容器内校准图片数量不正确：期望 $takeCount，实际 $($containerCalibrationFiles.Count)"
    }

    Write-Host 'Hailo 工作区准备完成，尚未开始编译。'
    Write-Host "容器工作区：$workDir"
    Write-Host "ONNX：$onnxName"
    Write-Host "校准图片：$takeCount / $($images.Count)"
    Write-Host '手动编译命令：'
    Write-Host ".\Model_Conversion\Hailo\compile_yolov8n_hailo8l.ps1 -WorkName '$WorkName' -OnnxFileName '$onnxName' -OutputPath '<输出.hef>' -Classes 1"
}
finally {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
