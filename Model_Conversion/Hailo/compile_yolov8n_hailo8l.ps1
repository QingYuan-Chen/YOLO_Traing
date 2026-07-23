[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$WorkName,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[^/\\]+\.onnx$')]
    [string]$OnnxFileName,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [ValidateRange(1, 10000)]
    [int]$Classes = 1,
    [string]$Container = 'hailo-suite',
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'

function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "docker 命令失败，退出码：$LASTEXITCODE" }
}

& docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop 未启动或 Docker 引擎不可用。' }
Invoke-Docker start $Container | Out-Null

$outputFullPath = [IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFullPath
if (-not (Test-Path -LiteralPath $outputDirectory)) { New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null }
if (Test-Path -LiteralPath $outputFullPath) { throw "输出文件已存在，为避免覆盖已停止：$outputFullPath" }

$workDir = "/home/hailo/model-conversion/$WorkName"
$containerOnnx = "$workDir/$OnnxFileName"
$modelStem = [IO.Path]::GetFileNameWithoutExtension($OnnxFileName)
$containerHef = "$workDir/${modelStem}_hailo8l.hef"
$containerLog = "$workDir/${modelStem}_compile.log"

Invoke-Docker exec $Container bash -lc "test -r '$containerOnnx' && test -w '$workDir'"
$containerCalibrationFiles = @(& docker exec $Container find "$workDir/calib_images" -maxdepth 1 -type f)
if ($LASTEXITCODE -ne 0) {
    throw '无法读取容器内校准图片列表。'
}
if ($containerCalibrationFiles.Count -eq 0) {
    throw "容器校准目录为空：$workDir/calib_images"
}
if ($PreflightOnly) {
    $containerPwd = @(Invoke-Docker exec -w $workDir $Container pwd)[-1].Trim()
    if ($containerPwd -ne $workDir) {
        throw "Docker 工作目录转发测试失败：期望 $workDir，实际 $containerPwd"
    }
    Write-Host 'Hailo 编译预检通过，未启动编译。'
    Write-Host "容器工作区：$workDir"
    Write-Host "ONNX：$containerOnnx"
    Write-Host "校准图片：$($containerCalibrationFiles.Count)"
    return
}

$endNodes = @(
    '/model.22/cv2.0/cv2.0.2/Conv',
    '/model.22/cv3.0/cv3.0.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv'
)
$endNodeText = $endNodes -join ' '
$compile = "set -o pipefail; source /local/workspace/hailo_virtualenv/bin/activate; " +
    "hailomz compile yolov8n --ckpt '$containerOnnx' --calib-path '$workDir/calib_images' " +
    "--hw-arch hailo8l --classes $Classes --performance --end-node-names $endNodeText " +
    "2>&1 | tee '$containerLog' && mv -f '$workDir/yolov8n.hef' '$containerHef' && sha256sum '$containerHef'"

Write-Host '开始 Hailo-8L 编译；该步骤可能需要较长时间。'
Invoke-Docker exec -w $workDir $Container bash -lc $compile
Invoke-Docker cp "${Container}:$containerHef" $outputFullPath

$containerHash = ((& docker exec $Container sha256sum $containerHef) -split '\s+')[0].ToUpperInvariant()
if ($LASTEXITCODE -ne 0) { throw '无法读取容器内 HEF 哈希。' }
$hostHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputFullPath).Hash
if ($containerHash -ne $hostHash) { throw "复制后 SHA-256 不一致：container=$containerHash, host=$hostHash" }

Write-Host "转换完成：$outputFullPath"
Write-Host "容器日志：$containerLog"
Write-Host "SHA-256：$hostHash"
