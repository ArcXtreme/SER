<#
E4 CREMA-D restart under a RAM guard - meant to be run OUTSIDE Claude Code (own PowerShell window).

  -Step run       (default) preflight, then ONLY:  python -u scripts\stage5_e4_run.py --datasets CREMA-D
  -Step followup  after a successful run: summarise (both datasets), controls RAVDESS, controls CREMA-D, audit
  -Step selftest  guard self-test with a tiny dummy worker (no experiment code, logs to %TEMP%)

Guard (one process, one CSV writer):
  - samples every 1 s while the worker is opening the feature store or loading a HuBERT layer, else every 5 s
  - CSV columns: time, worker working set (GB), system free RAM (GB), current [stage] line, last log line
  - if system free RAM < -MinFreeGB (default 1.0): kills the worker tree, writes <tag>_ABORT.txt with the stage,
    exits 2. NEVER restarts anything.
  - exit 3 if the worker wrote anything to stderr (Python error), exit 4 if protected files changed.
Preflight (run): refuses to start if another stage5 python process is running, if protected files differ from
results/E4/snapshot_before_cremad_restart.json or the E1-E3 snapshot, or if free RAM < -MinStartFreeGB (default 3).
#>
param([ValidateSet("run", "followup", "selftest")][string]$Step = "run",
      [double]$MinFreeGB = 1.0, [double]$MinStartFreeGB = 3.0)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$Py = "C:\SER_data\.venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"

function FreeGB { (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB }

function Invoke-Guarded([string]$Tag, [string[]]$PyArgs, [string]$Dir) {
    $log = Join-Path $Dir "$Tag.log"; $err = "$log.err"; $csv = Join-Path $Dir "ram_monitor_$Tag.csv"
    foreach ($f in $log, $err, $csv) { if (Test-Path $f) { throw "refusing to overwrite $f" } }
    "time,worker_ws_gb,system_free_gb,stage,last_line" | Out-File $csv -Encoding utf8
    Write-Host "[$Tag] start $(Get-Date -Format s): $Py -u $($PyArgs -join ' ')"
    $p = Start-Process -FilePath $Py -ArgumentList (@("-u") + $PyArgs) -RedirectStandardOutput $log `
         -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $minFree = 99.0; $maxWs = 0.0; $stage = "starting"; $n = 0
    while (-not $p.HasExited) {
        $fast = $stage -match "opening feature store|loading HuBERT layer|starting"
        Start-Sleep -Milliseconds ($(if ($fast) { 1000 } else { 5000 }))
        $ws = 0.0
        $ids = @($p.Id) + @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue | ForEach-Object { $_.ProcessId })
        foreach ($i in $ids) { try { $w = (Get-Process -Id $i -ErrorAction Stop).WorkingSet64 / 1GB; if ($w -gt $ws) { $ws = $w } } catch {} }
        $free = FreeGB
        $tail = @(Get-Content $log -Tail 30 -ErrorAction SilentlyContinue)
        $st = @($tail | Where-Object { $_ -match "\[stage\]" }); if ($st.Count) { $stage = $st[-1].Trim() }
        $last = if ($tail.Count) { $tail[-1].Trim() } else { "" }
        $q = { param($s) '"' + ($s -replace '"', "'") + '"' }
        "{0},{1:N3},{2:N3},{3},{4}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fff"), $ws, $free, (& $q $stage), (& $q $last) |
            Add-Content $csv -Encoding utf8
        if ($free -lt $minFree) { $minFree = $free }; if ($ws -gt $maxWs) { $maxWs = $ws }
        $n++; if ($n % 60 -eq 0) { Write-Host ("[{0}] {1} free {2:N2} GB, worker {3:N2} GB | {4}" -f $Tag, (Get-Date -Format T), $free, $ws, $stage) }
        if ($free -lt $MinFreeGB) {
            & taskkill /PID $p.Id /T /F | Out-Null
            $msg = "RAM GUARD STOP ($Tag) at $(Get-Date -Format s): system free {0:N2} GB < {1} GB; worker {2:N2} GB`r`nstage: {3}`r`nlast log line: {4}" -f $free, $MinFreeGB, $ws, $stage, $last
            $msg | Out-File (Join-Path $Dir "${Tag}_ABORT.txt") -Encoding utf8
            Write-Host $msg -ForegroundColor Red
            return 2
        }
    }
    $p.WaitForExit()
    Write-Host ("[{0}] finished {1}; min free RAM {2:N2} GB; max worker WS {3:N2} GB; last stage: {4}" -f $Tag, (Get-Date -Format s), $minFree, $maxWs, $stage)
    $e = Get-Content $err -ErrorAction SilentlyContinue
    if ($e) { Write-Host "--- stderr ($Tag) ---" -ForegroundColor Red; $e | Select-Object -Last 25 | Write-Host; return 3 }
    return 0
}

function Assert-Protected {
    & $Py -I scripts\stage5_e4_protect.py verify
    if ($LASTEXITCODE -ne 0) { Write-Host "Protected files changed - stopping." -ForegroundColor Red; exit 4 }
}

if ($Step -eq "selftest") {
    $dir = Join-Path $env:TEMP "e4_guard_selftest_$Stamp"; New-Item -ItemType Directory $dir | Out-Null
    $code = "import time;print('  [stage] CREMA-D: opening feature store',flush=True);time.sleep(2);" +
            "print('  [stage] loading HuBERT layer 7',flush=True);time.sleep(4);print('    loaded HuBERT layer 7',flush=True);" +
            "print('  [stage] CREMA-D fold 0: phase D xattn_HMP seed 0 (training)',flush=True);time.sleep(11)"
    $r = Invoke-Guarded "selftest" @("-c", "`"$code`"") $dir
    Get-Content (Join-Path $dir "ram_monitor_selftest.csv")
    exit $r
}

$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "stage5_" }
if ($busy) { Write-Host "Another stage5 python process is running (PID $($busy.ProcessId -join ', ')) - not starting." -ForegroundColor Red; exit 5 }
Assert-Protected
$dir = Join-Path $Root "results\E4"

if ($Step -eq "run") {
    $f = FreeGB
    Write-Host ("Free RAM before start: {0:N2} GB (required >= {1} GB; stop threshold {2} GB)" -f $f, $MinStartFreeGB, $MinFreeGB)
    if ($f -lt $MinStartFreeGB) { Write-Host "Not enough free RAM to start - close applications and retry." -ForegroundColor Red; exit 6 }
    $r = Invoke-Guarded "full_CREMA-D_restart_$Stamp" @("scripts\stage5_e4_run.py", "--datasets", "CREMA-D") $dir
    Assert-Protected
    Write-Host "CREMA-D run exit code: $r"
    exit $r
}

if ($Step -eq "followup") {
    $steps = @(@("summarise", @("scripts\stage5_e4_run.py", "--datasets", "RAVDESS", "CREMA-D", "--summarise")),
               @("controls_RAVDESS", @("scripts\stage5_e4_controls.py", "--datasets", "RAVDESS")),
               @("controls_CREMA-D", @("scripts\stage5_e4_controls.py", "--datasets", "CREMA-D")),
               @("audit", @("scripts\stage5_e4_audit.py")))
    foreach ($s in $steps) {
        $r = Invoke-Guarded "$($s[0])_$Stamp" $s[1] $dir
        if ($r -ne 0) { Write-Host "STOPPED at $($s[0]) (exit $r). Nothing restarted." -ForegroundColor Red; Assert-Protected; exit $r }
    }
    Assert-Protected
    Write-Host "FOLLOW-UP COMPLETE"
    exit 0
}
