[CmdletBinding()]
param(
    [string]$FreeCADRoot = "",
    [string]$ProjectPython = "",
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [switch]$SkipImportCheck,
    [switch]$Json
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

try {
    $repoRoot = Get-S1RepoRoot
    if (-not $FreeCADRoot) { $FreeCADRoot = Find-S1FreeCADRoot }
    if (-not $ProjectPython) { $ProjectPython = Join-Path $repoRoot ".venv\Scripts\python.exe" }
    $layout = Resolve-S1FreeCADLayout -FreeCADRoot $FreeCADRoot
    Set-S1ProcessEnvironment -Layout $layout -RepoRoot $repoRoot
    Assert-S1PortAvailable -Port $Port
    Assert-S1ProjectImports -ProjectPython $ProjectPython
    if (-not $SkipImportCheck) {
        Assert-S1FreeCADImports -Layout $layout
    }
    $result = [ordered]@{
        Root = $layout.Root
        App = $layout.App
        Python = $layout.Python
        Lib = $layout.Lib
        ProjectPython = [IO.Path]::GetFullPath($ProjectPython)
        Port = $Port
    }
    if ($Json) {
        [pscustomobject]$result | ConvertTo-Json -Compress
    }
    else {
        Write-Output "Preflight OK: FreeCAD $($layout.App); project Python $ProjectPython; port $Port available."
    }
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
