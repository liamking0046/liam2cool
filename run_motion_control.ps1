param(
    [Parameter(Mandatory = $true)]
    [Alias('Image')]
    [string]$Source,

    [Parameter(Mandatory = $true)]
    [Alias('Video')]
    [string]$Reference,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [double]$FlowScale = 0.8,
    [double]$GlobalScale = 1.0,
    [double]$BlendStrength = 0.15,
    [int]$MaxFrames = 0
)

function Test-VideoPath([string]$Path) {
    $videoExt = @('.mp4', '.mov', '.avi', '.mkv', '.webm')
    return $videoExt -contains [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
}

function Test-ImagePath([string]$Path) {
    $imageExt = @('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    return $imageExt -contains [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
}

if ((Test-VideoPath $Source) -and (Test-ImagePath $Reference)) {
    throw "Inputs appear swapped. -Source must be an image and -Reference must be a video. Example: -Source .\assets\source.jpg -Reference .\assets\reference.mp4"
}

$pythonArgs = @(
    'motion_control.py',
    '--source', $Source,
    '--reference', $Reference,
    '--output', $Output,
    '--flow-scale', $FlowScale,
    '--global-scale', $GlobalScale,
    '--blend-strength', $BlendStrength
)

if ($MaxFrames -gt 0) {
    $pythonArgs += @('--max-frames', $MaxFrames)
}

python @pythonArgs
