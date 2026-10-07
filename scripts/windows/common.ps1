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
        $app = [IO.Path]::GetFullPath($env:FREECAD_APP)
        if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
            throw "FREECAD_APP does not name an existing executable: '$app'."
        }
        $bin = Split-Path -Parent $app
        try {
            $layout = Resolve-S1FreeCADLayout -FreeCADRoot (Split-Path -Parent $bin)
        }
        catch {
            throw "FREECAD_APP does not belong to a complete FreeCAD layout: '$app'. $($_.Exception.Message)"
        }
        if (-not [string]::Equals($layout.App, $app, [StringComparison]::OrdinalIgnoreCase)) {
            throw "FREECAD_APP must point to the FreeCAD.exe selected by its installation layout: '$app'."
        }
        return $layout.Root
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

function Get-S1OwnedProcessTree {
    param(
        [Parameter(Mandatory = $true)][object[]]$Processes,
        [Parameter(Mandatory = $true)][int]$RootProcessId
    )

    $byId = @{}
    foreach ($process in $Processes) { $byId[[int]$process.ProcessId] = $process }
    if (-not $byId.ContainsKey($RootProcessId)) { return @() }

    $owned = New-Object 'System.Collections.Generic.List[object]'
    $owned.Add($byId[$RootProcessId])
    $accepted = @{ $RootProcessId = $byId[$RootProcessId] }
    do {
        $before = $owned.Count
        foreach ($process in $Processes) {
            $processId = [int]$process.ProcessId
            $parentPid = [int]$process.ParentProcessId
            if ($accepted.ContainsKey($parentPid) -and -not $accepted.ContainsKey($processId)) {
                $parentCreated = ([datetime]$accepted[$parentPid].CreationDate).ToUniversalTime()
                $childCreated = ([datetime]$process.CreationDate).ToUniversalTime()
                if ($childCreated -ge $parentCreated) {
                    $accepted[$processId] = $process
                    $owned.Add($process)
                }
            }
        }
    } while ($owned.Count -gt $before)
    return $owned.ToArray()
}

function Stop-S1ProcessTree {
    param(
        [Parameter(Mandatory = $true)][int]$RootProcessId,
        [Parameter(Mandatory = $true)][long]$RootStartTimeUtcFileTime
    )

    $root = Get-Process -Id $RootProcessId -ErrorAction SilentlyContinue
    if ($null -eq $root -or $root.StartTime.ToUniversalTime().ToFileTimeUtc() -ne $RootStartTimeUtcFileTime) {
        return
    }

    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $owned = @(Get-S1OwnedProcessTree -Processes $processes -RootProcessId $RootProcessId)
    $identities = @(foreach ($process in $owned) {
        $current = Get-Process -Id ([int]$process.ProcessId) -ErrorAction SilentlyContinue
        if ($null -ne $current) {
            [pscustomobject]@{
                Id = [int]$process.ProcessId
                StartTimeUtcFileTime = $current.StartTime.ToUniversalTime().ToFileTimeUtc()
            }
        }
    })
    for ($index = $identities.Count - 1; $index -ge 0; $index--) {
        $identity = $identities[$index]
        $current = Get-Process -Id $identity.Id -ErrorAction SilentlyContinue
        if ($null -ne $current -and
            $current.StartTime.ToUniversalTime().ToFileTimeUtc() -eq $identity.StartTimeUtcFileTime) {
            try { $current.Kill() } catch { }
        }
    }
}
