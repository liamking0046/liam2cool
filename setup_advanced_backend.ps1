param(
    [string]$BackendDir = ".\external\ComfyUI",
    [switch]$SkipCustomNodes
)

$ErrorActionPreference = 'Stop'

function Ensure-Git {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required. Install Git for Windows and reopen PowerShell."
    }
}

Ensure-Git

$backendPath = Resolve-Path -Path (Split-Path -Parent $BackendDir) -ErrorAction SilentlyContinue
if (-not $backendPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $BackendDir) | Out-Null
}

if (-not (Test-Path $BackendDir)) {
    Write-Host "Cloning ComfyUI into $BackendDir ..."
    git clone https://github.com/comfyanonymous/ComfyUI.git $BackendDir
} else {
    Write-Host "ComfyUI already exists at $BackendDir. Pulling latest changes..."
    git -C $BackendDir pull --ff-only
}

if (-not $SkipCustomNodes) {
    $customNodes = Join-Path $BackendDir "custom_nodes"
    New-Item -ItemType Directory -Force -Path $customNodes | Out-Null

    $repos = @(
        "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git",
        "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
        "https://github.com/ltdrdata/ComfyUI-Manager.git"
    )

    foreach ($repo in $repos) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($repo)
        $target = Join-Path $customNodes $name

        if (-not (Test-Path $target)) {
            Write-Host "Cloning $name ..."
            git clone $repo $target
        } else {
            Write-Host "$name already present. Pulling latest changes..."
            git -C $target pull --ff-only
        }
    }
}

Write-Host ""
Write-Host "Advanced backend repo setup complete."
Write-Host "Next:"
Write-Host "  1) cd $BackendDir"
Write-Host "  2) python -m venv .venv"
Write-Host "  3) .\.venv\Scripts\Activate.ps1"
Write-Host "  4) pip install -r requirements.txt"
Write-Host "  5) python main.py"
Write-Host ""
Write-Host "Then load AnimateDiff/ControlNet workflows in ComfyUI for higher-realism motion generation."
