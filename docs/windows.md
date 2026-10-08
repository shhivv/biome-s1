# Running Biome-S1 on Windows

Windows support is community-tested and experimental. The steps below were
verified on 64-bit Windows with installed FreeCAD 1.1.4. They use Biome-S1's
existing internal FreeCAD/Qt paths; the macOS accessibility and experimental
hands paths are not supported on Windows.

## Requirements

- 64-bit Windows 10 or 11.
- FreeCAD 1.1.x installed from the official FreeCAD release.
- Python 3.11 and [`uv`](https://docs.astral.sh/uv/) available from PowerShell.
- About 1 GB of free disk space for the Python environment and model files.

A discrete GPU is not required. Do not install PyTorch into FreeCAD's bundled
Python: FreeCAD and the model intentionally run in separate processes.

## Install the Python environment

Open PowerShell in the repository and run:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

This creates files only inside `.venv`; it does not require administrator
rights or change the system `PATH`.

## Check the installation

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1
```

Preflight checks the installed FreeCAD layout, both Python environments, the
required imports, and whether the local server port is free. If FreeCAD is in
a nonstandard location, pass its installation directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\preflight.ps1 `
  -FreeCADRoot "D:\Applications\FreeCAD 1.1"
```

A valid installation contains `bin\FreeCAD.exe`, `bin\python.exe`, and
`bin\FreeCAD.pyd`. The Python runtime also recognizes `FREECAD_APP`,
`FREECAD_PYTHON`, and `FREECAD_LIB`; the PowerShell launchers derive those
values from `-FreeCADRoot` and set them only for their own process tree.
When set directly, `FREECAD_APP` must point to `bin\FreeCAD.exe`.

## Run the demos

Taiga-S1 selects FreeCAD commands and lets the runtime fill their parameters:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start-taiga.ps1
```

Mesa-S1 operates the visible FreeCAD interface through its internal Qt view:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\start-mesa.ps1
```

Both commands default to level 3, split `iid`, seed 7. Override these with
`-Level`, `-Split`, and `-Seed`; use `-FreeCADRoot` for a nonstandard FreeCAD
location. Results are written to `runs\windows\taiga` and
`runs\windows\mesa`, including a `.FCStd` model and PNG preview. A successful
run prints `result: SUCCESS` and exits with code 0.

The launchers reserve `127.0.0.1:8765`, start only the selected FreeCAD
process, and stop that process tree when the demo finishes or fails. They do
not persist environment variables or stop unrelated FreeCAD sessions by name.

## Troubleshooting

- **FreeCAD not found:** rerun with `-FreeCADRoot` pointing to the directory
  above `bin`.
- **Import failure:** confirm that FreeCAD is 1.1.x and rerun preflight; do not
  copy `FreeCAD.pyd` into `.venv`.
- **Port 8765 is already in use:** close the previous Biome-S1 server or choose
  the same free `-Port` value for the launcher.
- **PowerShell blocks scripts:** use the complete commands above. Their
  execution-policy override applies only to that PowerShell process.
- **The demo exits nonzero:** read the final `result` line and the preceding
  error. A saved file alone does not mean the requested part matched.

To remove the project environment and generated demos, delete `.venv` and the
relevant directory under `runs\windows`. FreeCAD is an independent installed
application and is not modified or removed by these scripts.
