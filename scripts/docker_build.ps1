param(
    [string]$Platform = "",
    [string]$Tag = "usv-faults:dev",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu",
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Platform) {
    $args = @("buildx", "build", "--platform", $Platform, "--load", "-t", $Tag, "--build-arg", "TORCH_INDEX_URL=$TorchIndexUrl")
    if ($NoCache) {
        $args += "--no-cache"
    }
    $args += "."
    docker @args
} else {
    $args = @("compose", "build", "--build-arg", "TORCH_INDEX_URL=$TorchIndexUrl")
    if ($NoCache) {
        $args += "--no-cache"
    }
    docker @args
}
