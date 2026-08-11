param(
    [int]$CurrentTrainingPid = 0,
    [int]$PollHours = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$yolo = "C:\Users\youyk\.conda\envs\yolo\Scripts\yolo.exe"
$data = "datasets/M3FD_6ch/rgbt6.yaml"
$runsDirectory = Join-Path $workspace "runs\detect"
$queueLog = Join-Path $runsDirectory "m3fd_ms_queue.log"

Set-Location -LiteralPath $workspace

function Write-QueueLog {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $queueLog -Value $line
    Write-Host $line
}

function Get-CompletedEpochs {
    param([string]$RunName)

    $results = Join-Path $runsDirectory "$RunName\results.csv"
    if (-not (Test-Path -LiteralPath $results)) {
        return 0
    }
    return @(Import-Csv -LiteralPath $results).Count
}

function Test-RunComplete {
    param([string]$RunName)

    $best = Join-Path $runsDirectory "$RunName\weights\best.pt"
    return (Get-CompletedEpochs $RunName) -ge 200 -and (Test-Path -LiteralPath $best)
}

try {
    if ($PollHours -lt 1) {
        throw "PollHours must be at least 1."
    }

    if ($CurrentTrainingPid -gt 0) {
        $currentRun = "m3fd_dual_ms_drr_200"
        while (-not (Test-RunComplete $currentRun)) {
            $completedEpochs = Get-CompletedEpochs $currentRun
            $currentProcess = Get-Process -Id $CurrentTrainingPid -ErrorAction SilentlyContinue
            if (-not $currentProcess) {
                throw "$currentRun process stopped at $completedEpochs/200 epochs; queue halted."
            }
            Write-QueueLog "$currentRun is at $completedEpochs/200 epochs; checking again in $PollHours hour(s)."
            Start-Sleep -Seconds ($PollHours * 3600)
        }
        Write-QueueLog "$currentRun completed successfully."
    }

    $queue = @(
        @{ Pattern = "rrc"; Model = "custom_models/yolov8n-dual-ms-rrc.yaml" },
        @{ Pattern = "drc"; Model = "custom_models/yolov8n-dual-ms-drc.yaml" }
    )

    foreach ($experiment in $queue) {
        $runName = "m3fd_dual_ms_{0}_200" -f $experiment.Pattern
        if (Test-RunComplete $runName) {
            Write-QueueLog "$runName already completed; skipping."
            continue
        }

        $runDirectory = Join-Path $runsDirectory $runName
        if (Test-Path -LiteralPath $runDirectory) {
            $completedEpochs = Get-CompletedEpochs $runName
            throw "$runName already exists with $completedEpochs/200 epochs; queue halted to avoid a duplicate run."
        }

        $trainLog = Join-Path $runsDirectory "$runName.console.log"
        Write-QueueLog "Starting $runName with model $($experiment.Model)."
        & $yolo detect train "model=$($experiment.Model)" "data=$data" imgsz=640 batch=2 epochs=200 workers=0 seed=0 "name=$runName" 2>&1 | Tee-Object -FilePath $trainLog
        $trainExitCode = $LASTEXITCODE
        if ($trainExitCode -ne 0) {
            throw "$runName exited with code $trainExitCode. See $trainLog."
        }
        if (-not (Test-RunComplete $runName)) {
            $completedEpochs = Get-CompletedEpochs $runName
            throw "$runName exited without completing 200 epochs ($completedEpochs/200)."
        }
        Write-QueueLog "$runName completed successfully."
    }

    Write-QueueLog "All queued experiments completed."
} catch {
    Write-QueueLog "ERROR: $($_.Exception.Message)"
    exit 1
}
