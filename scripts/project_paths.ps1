function Resolve-YoloWorkspaceRoot {
    [CmdletBinding()]
    param([string]$WorkspaceRoot)

    if ($WorkspaceRoot) {
        return [IO.Path]::GetFullPath($WorkspaceRoot)
    }
    if ($env:YOLO_WORKSPACE_ROOT) {
        return [IO.Path]::GetFullPath($env:YOLO_WORKSPACE_ROOT)
    }
    return Join-Path $env:USERPROFILE 'Desktop\YOLOTraining'
}
