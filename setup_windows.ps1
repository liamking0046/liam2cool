param(
    [string]$PythonVersion = "3.11",
    [switch]$SkipSystemInstalls,
    [switch]$SkipNodeSetup
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Ensure-Command {
    param(
        [string]$Name,
        [string]$WingetId,
        [string]$InstallHint
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "$Name found: $($cmd.Source)"
        return $true
    }

    if ($SkipSystemInstalls) {
        throw "$Name is missing. Install it first. $InstallHint"
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$Name is missing and winget is not available. $InstallHint"
    }

    Write-Host "$Name not found. Installing with winget ($WingetId)..."
    winget install --id $WingetId -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install $Name with winget. $InstallHint"
    }

    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name install was attempted but the command is still unavailable. Open a new terminal and rerun this script."
    }

    return $true
}

Write-Step "Checking required tools"
Ensure-Command -Name git -WingetId Git.Git -InstallHint "Install Git for Windows from https://git-scm.com/download/win"
Ensure-Command -Name py -WingetId "Python.Python.$PythonVersion" -InstallHint "Install Python $PythonVersion+ from https://www.python.org/downloads/windows/"

if (-not $SkipNodeSetup) {
    Ensure-Command -Name node -WingetId OpenJS.NodeJS.LTS -InstallHint "Install Node.js LTS from https://nodejs.org/"
}

Write-Step "Creating/updating virtual environment"
py -$PythonVersion -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed with py -$PythonVersion. Retrying with default py launcher..." -ForegroundColor Yellow
    py -m venv .venv
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "Virtual environment creation failed."
}

$pythonExe = Resolve-Path .\.venv\Scripts\python.exe

Write-Step "Installing Python dependencies"
& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt

if (-not $SkipNodeSetup) {
    Write-Step "Installing npm dependencies"
    npm install
}

Write-Step "Setup complete"
Write-Host "Run one of these commands:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  npm start -- --source .\assets\source.jpg --reference .\assets\reference.mp4 --output .\outputs\motion_control.mp4"
Write-Host "  .\run_motion_control.ps1 -Source .\assets\source.jpg -Reference .\assets\reference.mp4 -Output .\outputs\motion_control.mp4"
