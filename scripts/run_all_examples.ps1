<#!
.SYNOPSIS
Runs all example run_ga.py scripts (optionally multiple times) in parallel using PowerShell jobs.

.DESCRIPTION
Discovers subdirectories of the ./examples folder containing a run_ga.py and launches each as a background job.
You can specify a repetition count (-Count) to execute each example multiple times for statistical analyses.
Each repetition is a separate job (Name pattern: <example>#<k>). Optionally throttle concurrent jobs with -Jobs.

.PARAMETER Count
Number of times to run each example (default 1).

.PARAMETER Jobs
Maximum number of concurrent jobs. If omitted or 0, all jobs launch immediately.

.PARAMETER Runner
The command prefix used to invoke Python (default 'uv run python'). You may change to 'python' or a venv path.

.PARAMETER ExamplesPath
Path to the examples directory (default ./examples relative to this script's parent directory).

.EXAMPLE
pwsh ./scripts/run_all_examples.ps1

.EXAMPLE
pwsh ./scripts/run_all_examples.ps1 -Count 5 -Jobs 4

.EXAMPLE
pwsh ./scripts/run_all_examples.ps1 -Count 3 -Runner "python"

.NOTES
Return code: 0 if all jobs succeed, non‑zero if any fail. Summary printed at end.
#>
param(
    [int]$Count = 1,
    [int]$Jobs = 0,
    [string]$Runner = 'uv run python',
    [string]$ExamplesPath = '',
    [switch]$Group,          # If set, one job per example; loop Count times inside that job
    [switch]$GroupRuns       # Alias for -Group (user convenience)
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Resolve examples directory
if ([string]::IsNullOrWhiteSpace($ExamplesPath)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
    $repoRoot   = Split-Path -Parent $scriptRoot
    $ExamplesPath = Join-Path $repoRoot 'examples'
}

if (-not (Test-Path $ExamplesPath)) {
    Write-Error "Examples directory not found: $ExamplesPath"
    exit 1
}

if ($Count -lt 1) {
    Write-Error "Count must be >= 1"
    exit 2
}

# Discover examples
$examples = Get-ChildItem -Path $ExamplesPath -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'run_ga.py') } | Sort-Object Name
if (-not $examples) {
    Write-Error 'No examples with run_ga.py found.'
    exit 3
}

if ($GroupRuns) { $Group = $true }

Write-Host "Discovered $($examples.Count) examples. Scheduling $Count run(s) each. (Group mode: $Group)" -ForegroundColor Cyan

# Split runner into tokens; naive split on space (simple). For more complex quoting, user should pass pre-split.
$runnerTokens = $Runner -split ' '

<#
Build job definitions.
If grouping: one job per example (Name = example name). The job will internally loop Count times.
Else: one job per repetition (existing behavior).
#>
if ($Group) {
    $allJobs = @()
    foreach ($ex in $examples) {
        $scriptPath = Join-Path $ex.FullName 'run_ga.py'
        $allJobs += [PSCustomObject]@{ Name = $ex.Name; Script = $scriptPath; BaseName = $ex.Name; Index = 1 }
    }
} else {
    $allJobs = @()
    foreach ($ex in $examples) {
        for ($k = 1; $k -le $Count; $k++) {
            $label = if ($Count -gt 1) { "$($ex.Name)#$k" } else { $ex.Name }
            $scriptPath = Join-Path $ex.FullName 'run_ga.py'
            $allJobs += [PSCustomObject]@{ Name = $label; Script = $scriptPath; BaseName = $ex.Name; Index = $k }
        }
    }
}

$totalJobs = $allJobs.Count

# Auto-derive concurrency if -Jobs not specified (0): min(number of examples, logical cores)
if ($Jobs -le 0) {
    try {
        $logical = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
        if (-not $logical -or $logical -le 0) { $logical = 1 }
    } catch { $logical = 1 }
    $Jobs = [Math]::Min($totalJobs, $logical)
    Write-Host "Auto-selected concurrency: $Jobs (examples=$totalJobs, logicalCores=$logical)" -ForegroundColor DarkCyan
}

Write-Host "Total jobs: $totalJobs (Concurrency limit: $Jobs)" -ForegroundColor Cyan

function Start-ExampleJob($jobDef, $groupMode, $count, $runnerTokens) {
    $cmd = @($runnerTokens + $jobDef.Script)
    if ($groupMode) {
        Start-Job -Name $jobDef.Name -ScriptBlock {
            param($command, $runs)
            $failed = 0
            for ($i = 1; $i -le $runs; $i++) {
                Write-Host "[$($command[-1])] Run #$i" -ForegroundColor Cyan
                & $command[0] $command[1..($command.Count-1)]
                $code = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
                if ($code -ne 0) {
                    Write-Host "Run #$i failed with code $code" -ForegroundColor Red
                    $failed = 1
                }
            }
            if ($failed -ne 0) { exit 1 } else { exit 0 }
        } -ArgumentList (,$cmd), $count | Out-Null
    } else {
        Start-Job -Name $jobDef.Name -ScriptBlock {
            param($command)
            & $command[0] $command[1..($command.Count-1)]
            $code = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
            exit $code
        } -ArgumentList (,$cmd) | Out-Null
    }
}

$launched = 0
foreach ($jobDef in $allJobs) {
    if ($Jobs -gt 0) {
        while ( (@(Get-Job | Where-Object { $_.State -eq 'Running' })).Count -ge $Jobs ) {
            Start-Sleep -Seconds 1
            $running = (@(Get-Job | Where-Object { $_.State -eq 'Running' })).Count
            Write-Host ("Throttling: {0}/{1} running..." -f $running, $Jobs) -ForegroundColor DarkGray
        }
    }
    Start-ExampleJob $jobDef $Group $Count $runnerTokens
    $launched++
}

Write-Host "Launched $launched jobs." -ForegroundColor Green

<#
Reworked wait loop to avoid interactive Prompt:
We use Wait-Job -Any to wait for at least one job to finish, then Receive-Job (no -Wait) for completed jobs.
We capture per-job exit codes by inspecting the child job's Error/State and using $job.ChildJobs[0].JobStateInfo.ExitCode when available.
This prevents PowerShell from prompting for Job[] indices.
#>

$jobResults = @{}
while ($true) {
    $allCurrent = @(Get-Job)
    if ($allCurrent.Count -eq 0) { break }
    $done = $allCurrent | Where-Object { $_.State -in 'Completed','Failed','Stopped' }
    foreach ($j in $done) {
        if ($jobResults.ContainsKey($j.Name)) { continue }
        $null = Receive-Job -Job $j -ErrorAction SilentlyContinue
        # Manually remove job to keep list small
        try { Remove-Job -Job $j -Force -ErrorAction SilentlyContinue } catch { }
        $exit = 0
        try {
            if ($j.ChildJobs.Count -gt 0 -and $j.ChildJobs[0].JobStateInfo.ExitCode -ne $null) {
                $exit = [int]$j.ChildJobs[0].JobStateInfo.ExitCode
            }
        } catch { }
        if ($j.State -ne 'Completed' -and $exit -eq 0) { $exit = 999 }
        $jobResults[$j.Name] = [PSCustomObject]@{ Name = $j.Name; State = $j.State; ExitCode = $exit }
    }
    Start-Sleep -Milliseconds 500
}

# Build results list preserving launch order
$results = @()
foreach ($jobDef in $allJobs) {
    if ($jobResults.ContainsKey($jobDef.Name)) {
        $results += $jobResults[$jobDef.Name]
    } else {
        # If somehow missing, mark unknown
        $results += [PSCustomObject]@{ Name = $jobDef.Name; State = 'Unknown'; ExitCode = 999 }
    }
}

Write-Host "\nSummary:" -ForegroundColor Cyan
$fail = 0
foreach ($r in ($results | Sort-Object Name)) {
    $status = if ($r.ExitCode -eq 0) { 'OK' } else { 'FAIL' }
    if ($status -eq 'FAIL') { $fail++ }
    Write-Host ("  {0,-25} {1}" -f $r.Name, $status)
}

if ($fail -gt 0) {
    Write-Host "Failures: $fail" -ForegroundColor Red
    exit 5
} else {
    Write-Host "All jobs completed successfully." -ForegroundColor Green
}
