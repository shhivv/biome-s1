[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [switch]$Ui,
    [string]$FreeCADRoot = "",
    [string]$ProjectPython = "",
    [int]$Level = 3,
    [string]$Split = "iid",
    [int]$Seed = 7,
    [string]$Out = "runs\gui_demo",
    [int]$Port = 8765,
    [switch]$SkipImportCheck,
    [string]$FreeCADApp = "",
    [string]$MacroPath = "",
    [string]$RunnerPath = "",
    [string]$CapturePath = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$freecadProcess = $null
$exitCode = 0
try {
    $repoRoot = Get-S1RepoRoot
    if (-not $FreeCADRoot) { $FreeCADRoot = Find-S1FreeCADRoot }
    if (-not $ProjectPython) { $ProjectPython = Join-Path $repoRoot ".venv\Scripts\python.exe" }
    $preflightParams = @{ FreeCADRoot = $FreeCADRoot; ProjectPython = $ProjectPython; Port = $Port }
    if ($SkipImportCheck) { $preflightParams.SkipImportCheck = $true }
    & (Join-Path $PSScriptRoot "preflight.ps1") @preflightParams | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Windows preflight failed." }

    $layout = Resolve-S1FreeCADLayout -FreeCADRoot $FreeCADRoot
    Set-S1ProcessEnvironment -Layout $layout -RepoRoot $repoRoot
    if ($Ui) { $env:FREECAD_S1_UI = "1" } else { Remove-Item Env:FREECAD_S1_UI -ErrorAction SilentlyContinue }
    if ($CapturePath) { $env:S1_TEST_CAPTURE = [IO.Path]::GetFullPath($CapturePath) }

    if (-not $FreeCADApp) { $FreeCADApp = $layout.App }
    if (-not $MacroPath) { $MacroPath = Join-Path $repoRoot "scripts\freecad_gui_server.FCMacro" }
    if (-not $RunnerPath) { $RunnerPath = Join-Path $repoRoot "scripts\gui_demo.py" }
    $modelPath = $Model
    if (-not [IO.Path]::IsPathRooted($modelPath)) { $modelPath = Join-Path $repoRoot $modelPath }
    $outPath = $Out
    if (-not [IO.Path]::IsPathRooted($outPath)) { $outPath = Join-Path $repoRoot $outPath }

    $freecadProcess = Start-S1ChildProcess -FilePath $FreeCADApp -ArgumentList @([IO.Path]::GetFullPath($MacroPath))
    Wait-S1Port -Process $freecadProcess -Port $Port

    & $ProjectPython ([IO.Path]::GetFullPath($RunnerPath)) --model ([IO.Path]::GetFullPath($modelPath)) `
        --level $Level --split $Split --seed $Seed --port $Port --out ([IO.Path]::GetFullPath($outPath))
    $exitCode = $LASTEXITCODE
}
catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    $exitCode = 1
}
finally {
    if ($null -ne $freecadProcess) {
        Stop-S1ProcessTree -RootProcessId $freecadProcess.Id
        $freecadProcess.Dispose()
    }
}
exit $exitCode
