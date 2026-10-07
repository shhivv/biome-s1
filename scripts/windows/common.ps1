Set-StrictMode -Version Latest

function Get-S1RepoRoot {
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
}

function Resolve-S1FreeCADLayout {
    param([Parameter(Mandatory = $true)][string]$FreeCADRoot)

    $root = [IO.Path]::GetFullPath($FreeCADRoot)
    $candidates = @($root)
    if (Test-Path -LiteralPath $root -PathType Container) {
        $candidates += @(Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { $_.FullName })
    }
    foreach ($candidate in $candidates) {
        foreach ($bin in @((Join-Path $candidate "bin"), $candidate)) {
            $app = Join-Path $bin "FreeCAD.exe"
            $python = Join-Path $bin "python.exe"
            $module = Join-Path $bin "FreeCAD.pyd"
            if ((Test-Path -LiteralPath $app -PathType Leaf) -and
                (Test-Path -LiteralPath $python -PathType Leaf) -and
                (Test-Path -LiteralPath $module -PathType Leaf)) {
                return [pscustomobject]@{
                    Root = [IO.Path]::GetFullPath($candidate)
                    App = [IO.Path]::GetFullPath($app)
                    Python = [IO.Path]::GetFullPath($python)
                    Lib = [IO.Path]::GetFullPath($bin)
                }
            }
        }
    }

    $expectedBin = Join-Path $root "bin"
    $missing = @("FreeCAD.exe", "python.exe", "FreeCAD.pyd") |
        Where-Object { -not (Test-Path -LiteralPath (Join-Path $expectedBin $_) -PathType Leaf) }
    throw "FreeCAD layout is incomplete under '$root'; missing: $($missing -join ', ')."
}

function Find-S1FreeCADRoot {
    $candidates = New-Object 'System.Collections.Generic.List[string]'
    if ($env:FREECAD_APP) {
        $bin = Split-Path -Parent ([IO.Path]::GetFullPath($env:FREECAD_APP))
        $candidates.Add((Split-Path -Parent $bin))
    }
    foreach ($base in @($env:ProgramW6432, $env:ProgramFiles)) {
        if ($base -and (Test-Path -LiteralPath $base -PathType Container)) {
            Get-ChildItem -LiteralPath $base -Directory -Filter "FreeCAD*" -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { $candidates.Add($_.FullName) }
        }
    }
    if ($env:LOCALAPPDATA) {
        $programs = Join-Path $env:LOCALAPPDATA "Programs"
        if (Test-Path -LiteralPath $programs -PathType Container) {
            Get-ChildItem -LiteralPath $programs -Directory -Filter "FreeCAD*" -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                ForEach-Object { $candidates.Add($_.FullName) }
        }
    }
    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        try {
            return (Resolve-S1FreeCADLayout -FreeCADRoot $candidate).Root
        }
        catch {
            continue
        }
    }
    throw "FreeCAD not found on Windows; install FreeCAD 1.1 or pass -FreeCADRoot."
}

function Set-S1ProcessEnvironment {
    param(
        [Parameter(Mandatory = $true)]$Layout,
        [string]$RepoRoot = (Get-S1RepoRoot)
    )

    $env:FREECAD_APP = $Layout.App
    $env:FREECAD_PYTHON = $Layout.Python
    $env:FREECAD_LIB = $Layout.Lib
    $env:FREECAD_S1_REPO = [IO.Path]::GetFullPath($RepoRoot)
}

function Assert-S1PortAvailable {
    param([int]$Port = 8765)

    $client = New-Object Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        throw "TCP port $Port is already in use; stop the existing listener before starting Biome-S1."
    }
    catch [Net.Sockets.SocketException] {
        return
    }
    finally {
        $client.Dispose()
    }
}

function Assert-S1ProjectImports {
    param([Parameter(Mandatory = $true)][string]$ProjectPython)

    $python = [IO.Path]::GetFullPath($ProjectPython)
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Project Python is missing: $python. Create .venv with Python 3.11 and install -e `".[dev]`"."
    }
    $output = & $python -c "import freecad_s1, torch; print(torch.__version__)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Project Python could not import freecad_s1 and torch:`n$($output -join [Environment]::NewLine)"
    }
}

function Assert-S1FreeCADImports {
    param([Parameter(Mandatory = $true)]$Layout)

    Set-S1ProcessEnvironment -Layout $Layout
    $output = & $Layout.Python -c "import FreeCAD, Part; print(FreeCAD.Version())" 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "FreeCAD's bundled Python could not import FreeCAD and Part:`n$($output -join [Environment]::NewLine)"
    }
}

function Start-S1ChildProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = [IO.Path]::GetFullPath($FilePath)
    $info.UseShellExecute = $false
    $quoted = foreach ($argument in $ArgumentList) {
        if ($argument -match '[\s"]') {
            '"' + $argument.Replace('"', '\"') + '"'
        }
        else {
            $argument
        }
    }
    $info.Arguments = $quoted -join " "
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $info
    if (-not $process.Start()) {
        throw "Could not start process: $FilePath"
    }
    return $process
}

function Wait-S1Port {
    param(
        [Parameter(Mandatory = $true)]$Process,
        [int]$Port = 8765,
        [int]$TimeoutSeconds = 120
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "FreeCAD exited with code $($Process.ExitCode) before opening TCP port $Port."
        }
        $client = New-Object Net.Sockets.TcpClient
        try {
            $client.Connect("127.0.0.1", $Port)
            return
        }
        catch [Net.Sockets.SocketException] {
            Start-Sleep -Milliseconds 200
        }
        finally {
            $client.Dispose()
        }
    }
    throw "FreeCAD did not open TCP port $Port within $TimeoutSeconds seconds."
}

function Stop-S1ProcessTree {
    param([Parameter(Mandatory = $true)][int]$RootProcessId)

    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $ids = New-Object 'System.Collections.Generic.List[int]'
    $ids.Add($RootProcessId)
    do {
        $before = $ids.Count
        foreach ($process in $processes) {
            if ($ids.Contains([int]$process.ParentProcessId) -and -not $ids.Contains([int]$process.ProcessId)) {
                $ids.Add([int]$process.ProcessId)
            }
        }
    } while ($ids.Count -gt $before)
    for ($index = $ids.Count - 1; $index -ge 0; $index--) {
        Stop-Process -Id $ids[$index] -Force -ErrorAction SilentlyContinue
    }
}
