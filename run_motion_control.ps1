param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [Parameter(Mandatory = $true)]
    [string]$Reference,

    [Parameter(Mandatory = $true)]
    [string]$Output,

    [double]$FlowScale = 0.8,
    [double]$GlobalScale = 1.0,
    [double]$BlendStrength = 0.15,
    [int]$MaxFrames = 0
)

$maxFramesArg = ""
if ($MaxFrames -gt 0) {
    $maxFramesArg = "--max-frames $MaxFrames"
}

python motion_control.py `
    --source "$Source" `
    --reference "$Reference" `
    --output "$Output" `
    --flow-scale $FlowScale `
    --global-scale $GlobalScale `
    --blend-strength $BlendStrength `
    $maxFramesArg

