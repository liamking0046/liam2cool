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
    [int]$MaxFrames = 0,
    [ValidateSet('natural', 'cgi')]
    [string]$Look = 'natural',
    [double]$LookStrength = 0.45,
    [double]$MotionFocus = 0.6,
    [double]$HandBoost = 0.35,
    [double]$TemporalSmooth = 0.2,
    [double]$IdentityLock = 0.45,
    [double]$StructureLock = 0.55,
    [double]$LightingTransfer = 0.35,
    [double]$FlowMomentum = 0.45,
    [int]$QualityPasses = 2,
    [double]$UpperbodyStickiness = 0.75,
    [double]$MicroMotion = 0.25,
    [double]$TemporalReproject = 0.55,
    [double]$OcclusionProtect = 0.65
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
    '--blend-strength', $BlendStrength,
    '--look', $Look,
    '--look-strength', $LookStrength,
    '--motion-focus', $MotionFocus,
    '--hand-boost', $HandBoost,
    '--temporal-smooth', $TemporalSmooth,
    '--identity-lock', $IdentityLock,
    '--structure-lock', $StructureLock,
    '--lighting-transfer', $LightingTransfer,
    '--flow-momentum', $FlowMomentum,
    '--quality-passes', $QualityPasses,
    '--upperbody-stickiness', $UpperbodyStickiness,
    '--micro-motion', $MicroMotion,
    '--temporal-reproject', $TemporalReproject,
    '--occlusion-protect', $OcclusionProtect
)

if ($MaxFrames -gt 0) {
    $pythonArgs += @('--max-frames', $MaxFrames)
}

python @pythonArgs
