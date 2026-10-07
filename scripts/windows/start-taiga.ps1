[CmdletBinding()]
param(
    [string]$FreeCADRoot = "",
    [string]$ProjectPython = "",
    [int]$Level = 3,
    [string]$Split = "iid",
    [int]$Seed = 7,
    [string]$Out = "runs\windows\taiga",
    [ValidateRange(1, 65535)][int]$Port = 8765,
    [switch]$SkipImportCheck,
    [string]$FreeCADApp = "",
    [string]$MacroPath = "",
    [string]$RunnerPath = "",
    [string]$CapturePath = ""
)

& (Join-Path $PSScriptRoot "start-demo.ps1") @PSBoundParameters -Model "release\hf"
exit $LASTEXITCODE
