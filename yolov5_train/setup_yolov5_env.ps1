[CmdletBinding()]
param(
  [string]$YoloV5Commit = "70b964b6d5067fff621f724c85d0e39e6b4c8e4e"
)

# 禁用 CONDA 插件以及取消频道提示
$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_NUMBER_CHANNEL_NOTICES = "0"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path

# 指定 conda 的环境路径和包下载缓存路径
$env:CONDA_ENVS_PATH = Join-Path $repoRoot ".conda_envs"
$env:CONDA_PKGS_DIRS = Join-Path $repoRoot ".conda_pkgs"

# 设定虚拟环境的存放路径以及 yolov5 源码仓库路径
$envPath = Join-Path $repoRoot ".conda_envs\yolov5"
$repoPath = Join-Path $PSScriptRoot "yolov5"

# 若环境不存在则创建基于 python 3.9 的新环境
if (!(Test-Path $envPath)) {
  conda create --solver=classic -p $envPath python=3.9 -y
}

# 激活新创建或已存在的 conda 虚拟环境
conda activate $envPath

# 若源码仓库不存在则克隆并锁定经过验证的 YOLOv5 提交
if (!(Test-Path $repoPath)) {
  git clone https://github.com/ultralytics/yolov5.git $repoPath
  git -C $repoPath checkout $YoloV5Commit
} else {
  $currentCommit = git -C $repoPath rev-parse HEAD
  if ($currentCommit -ne $YoloV5Commit) {
    Write-Warning "Existing YOLOv5 checkout is $currentCommit; verified commit is $YoloV5Commit. No automatic checkout was performed."
  }
}

# 切换工作目录到 yolov5 源码目录，并安装所需的所有依赖
Set-Location $repoPath
pip install -r requirements.txt
