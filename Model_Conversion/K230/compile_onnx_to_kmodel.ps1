[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OnnxPath,
    [Parameter(Mandatory = $true)]
    [string]$CalibrationImages,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [ValidateRange(1, 8192)]
    [int]$InputWidth = 320,
    [ValidateRange(1, 8192)]
    [int]$InputHeight = 320,
    [ValidateRange(1, 10000)]
    [int]$CalibrationCount = 10,
    [string]$Container = 'k230-converter',
    [string]$ValidationPython = 'E:\Anaconda_envs\envs\yolo\python.exe',
    [switch]$KeepWorkspace
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
if (-not (Test-Path -LiteralPath $ValidationPython -PathType Leaf)) { throw "验证 Python 不存在：$ValidationPython" }

$outputFullPath = [IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $outputFullPath
if (-not (Test-Path -LiteralPath $outputDirectory)) { New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null }
if (Test-Path -LiteralPath $outputFullPath) { throw "输出文件已存在，为避免覆盖已停止：$outputFullPath" }

$preflight = @'
import sys
import onnx
path = sys.argv[1]
model = onnx.load(path)
onnx.checker.check_model(model)
opsets = {item.domain or "ai.onnx": item.version for item in model.opset_import}
default_opset = opsets.get("ai.onnx", 0)
if default_opset > 14:
    raise SystemExit(f"K230 容器只兼容到较旧 opset；当前 ai.onnx opset={default_opset}，请重新导出 opset 12")
inputs = []
for value in model.graph.input:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "dynamic")
    inputs.append((value.name, dims))
if not inputs or any(not isinstance(dim, int) or dim <= 0 for _, shape in inputs for dim in shape):
    raise SystemExit(f"K230 转换要求静态输入；当前输入={inputs}")
print(f"ONNX preflight OK: ir={model.ir_version}, opsets={opsets}, inputs={inputs}")
'@
& $ValidationPython -c $preflight $resolvedOnnx
if ($LASTEXITCODE -ne 0) { throw 'ONNX 预检查失败。' }

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

$token = [Guid]::NewGuid().ToString('N').Substring(0, 10)
$workspace = "/workspace/model-conversion-$token"
$containerOnnx = "$workspace/model.onnx"
$containerCalibration = "$workspace/calibration"
$containerKmodel = "$workspace/model.kmodel"

try {
    Invoke-Docker exec $Container sh -lc "mkdir -p '$containerCalibration'"
    Invoke-Docker cp $resolvedOnnx "${Container}:$containerOnnx"
    for ($i = 0; $i -lt $selected.Count; $i++) {
        $target = ('calib_{0:D4}{1}' -f $i, $selected[$i].Extension.ToLowerInvariant())
        Invoke-Docker cp $selected[$i].FullName "${Container}:$containerCalibration/$target"
    }

    Invoke-Docker exec -w $workspace $Container python3 /home/user/model_converter/convert_model.py `
        --model $containerOnnx --dataset_path $containerCalibration `
        --input_width $InputWidth --input_height $InputHeight --target k230 --ptq_option 0

    Invoke-Docker exec -e 'PATH=/home/user/.local/lib/python3.8/site-packages:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' `
        -w $workspace $Container python3 -c "import nncase; s=nncase.Simulator(); s.load_model(open('model.kmodel','rb').read()); print('KModel simulator load OK')"
    Invoke-Docker cp "${Container}:$containerKmodel" $outputFullPath

    $containerHash = ((& docker exec $Container sha256sum $containerKmodel) -split '\s+')[0].ToUpperInvariant()
    if ($LASTEXITCODE -ne 0) { throw '无法读取容器内 KModel 哈希。' }
    $hostHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputFullPath).Hash
    if ($containerHash -ne $hostHash) { throw "复制后 SHA-256 不一致：container=$containerHash, host=$hostHash" }

    Write-Host "转换完成：$outputFullPath"
    Write-Host "校准图片：$takeCount / $($images.Count)"
    Write-Host "SHA-256：$hostHash"
}
finally {
    if (-not $KeepWorkspace -and $workspace -like '/workspace/model-conversion-*') {
        & docker exec $Container sh -lc "rm -rf -- '$workspace'" *> $null
    }
}
