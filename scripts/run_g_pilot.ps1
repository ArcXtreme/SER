<#
G-study guarded runner (report section 8.8). One worker process at a time, ONE writer of each RAM CSV.

  -Step selftest   guard self-test with a dummy worker (logs to %TEMP%)
  -Step gates      reproduction gates R1-R6, determinism probes, B0 input check (fold 0; validation only)
  -Step pilot      precompute + all pilot groups (fold 0; validation only)   [requires team approval]

GUARD (team decision 2026-11-07)
  - stop if AVAILABLE memory (free + reclaimable standby/cache; Win32_PerfFormattedData_PerfOS_Memory.AvailableBytes)
    < 1 GB, or if the worker's working set > 6 GB. Free RAM, available RAM and worker memory are logged separately.
  - sampling every 1 s while a stage line says opening/loading/streaming/precompute, otherwise every 5 s.
  - on a stop: kill the worker tree, write <tag>_ABORT.txt with the stage, exit 2. NEVER restarts anything.
  - exit 3 if the worker wrote to stderr; exit 4 if protected files changed; the chain stops at the first non-zero step.
#>
param([ValidateSet("selftest", "gates", "pilot")][string]$Step = "gates",
      [double]$MinAvailableGB = 1.0, [double]$MaxWorkerGB = 6.0, [double]$MinStartAvailableGB = 3.0)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$Py = "C:\SER_data\.venv\Scripts\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Dir = Join-Path $Root "results\G"
New-Item -ItemType Directory -Force $Dir | Out-Null

# AVAILABLE physical memory (free + standby/cache, i.e. reclaimable) from GlobalMemoryStatusEx (ullAvailPhys): fast and
# locale-independent (the WMI performance-counter class took 4-9 s per query in the self-test). FREE memory from WMI.
Add-Type -TypeDefinition @"
using System; using System.Runtime.InteropServices;
public static class GMem {
  [StructLayout(LayoutKind.Sequential)] public class MS { public uint dwLength = (uint)Marshal.SizeOf(typeof(MS)); public uint dwMemoryLoad;
    public ulong ullTotalPhys, ullAvailPhys, ullTotalPageFile, ullAvailPageFile, ullTotalVirtual, ullAvailVirtual, ullAvailExtendedVirtual; }
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool GlobalMemoryStatusEx([In, Out] MS m);
  public static double AvailableGB() { var m = new MS(); if (!GlobalMemoryStatusEx(m)) throw new Exception("GlobalMemoryStatusEx failed"); return m.ullAvailPhys / 1073741824.0; }
}
"@
function Mem {
    $os = Get-CimInstance Win32_OperatingSystem
    [pscustomobject]@{ free = $os.FreePhysicalMemory / 1MB; available = [GMem]::AvailableGB() }
}

function Invoke-Guarded([string]$Tag, [string[]]$PyArgs, [string]$OutDir) {
    $log = Join-Path $OutDir "$Tag.log"; $err = "$log.err"; $csv = Join-Path $OutDir "ram_monitor_$Tag.csv"
    foreach ($f in $log, $err, $csv) { if (Test-Path $f) { throw "refusing to overwrite $f" } }
    "time,worker_ws_gb,system_free_gb,system_available_gb,stage,last_line" | Out-File $csv -Encoding utf8
    Write-Host "[$Tag] start $(Get-Date -Format s): $Py -u $($PyArgs -join ' ')"
    $p = Start-Process -FilePath $Py -ArgumentList (@("-u") + $PyArgs) -RedirectStandardOutput $log `
         -RedirectStandardError $err -WindowStyle Hidden -PassThru
    $minAvail = 99.0; $minFree = 99.0; $maxWs = 0.0; $stage = "starting"; $n = 0
    while (-not $p.HasExited) {
        $fast = $stage -match "opening|loading|streaming|precompute|starting|ZW"
        Start-Sleep -Milliseconds ($(if ($fast) { 1000 } else { 5000 }))
        $ws = 0.0
        $ids = @($p.Id) + @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue | ForEach-Object { $_.ProcessId })
        foreach ($i in $ids) { try { $w = (Get-Process -Id $i -ErrorAction Stop).WorkingSet64 / 1GB; if ($w -gt $ws) { $ws = $w } } catch {} }
        $m = Mem
        $tail = @(Get-Content $log -Tail 30 -ErrorAction SilentlyContinue)
        $st = @($tail | Where-Object { $_ -match "\[stage\]" }); if ($st.Count) { $stage = $st[-1].Trim() }
        $last = if ($tail.Count) { $tail[-1].Trim() } else { "" }
        $q = { param($s) '"' + ($s -replace '"', "'") + '"' }
        "{0},{1:N3},{2:N3},{3:N3},{4},{5}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fff"), $ws, $m.free, $m.available, (& $q $stage), (& $q $last) |
            Add-Content $csv -Encoding utf8
        if ($m.available -lt $minAvail) { $minAvail = $m.available }; if ($m.free -lt $minFree) { $minFree = $m.free }; if ($ws -gt $maxWs) { $maxWs = $ws }
        $n++; if ($n % 60 -eq 0) { Write-Host ("[{0}] {1} available {2:N2} GB, free {3:N2} GB, worker {4:N2} GB | {5}" -f $Tag, (Get-Date -Format T), $m.available, $m.free, $ws, $stage) }
        $why = $null
        if ($m.available -lt $MinAvailableGB) { $why = "available memory {0:N2} GB < {1} GB" -f $m.available, $MinAvailableGB }
        elseif ($ws -gt $MaxWorkerGB) { $why = "worker working set {0:N2} GB > {1} GB" -f $ws, $MaxWorkerGB }
        if ($why) {
            & taskkill /PID $p.Id /T /F | Out-Null
            $msg = "RESOURCE GUARD STOP ($Tag) at $(Get-Date -Format s): $why (free {0:N2} GB, available {1:N2} GB, worker {2:N2} GB)`r`nstage: {3}`r`nlast log line: {4}" -f $m.free, $m.available, $ws, $stage, $last
            $msg | Out-File (Join-Path $OutDir "${Tag}_ABORT.txt") -Encoding utf8
            Write-Host $msg -ForegroundColor Red
            return 2
        }
    }
    $p.WaitForExit()
    Write-Host ("[{0}] finished {1}; min available {2:N2} GB; min free {3:N2} GB; max worker WS {4:N2} GB; last stage: {5}" -f $Tag, (Get-Date -Format s), $minAvail, $minFree, $maxWs, $stage)
    $e = Get-Content $err -ErrorAction SilentlyContinue
    if ($e) { Write-Host "--- stderr ($Tag) ---" -ForegroundColor Red; $e | Select-Object -Last 25 | Write-Host; return 3 }
    return 0
}

function Assert-Protected([switch]$Quick) {
    $a = @("-I", "scripts\stage5_g_protect.py", "verify"); if ($Quick) { $a += "--quick" }
    & $Py @a
    if ($LASTEXITCODE -ne 0) { Write-Host "Protected files changed - stopping." -ForegroundColor Red; exit 4 }
    & $Py -I scripts\stage5_e4_protect.py verify
    if ($LASTEXITCODE -ne 0) { Write-Host "E1-E4 snapshots changed - stopping." -ForegroundColor Red; exit 4 }
}

if ($Step -eq "selftest") {
    $tdir = Join-Path $env:TEMP "g_guard_selftest_$Stamp"; New-Item -ItemType Directory $tdir | Out-Null
    $code = "import time;print('  [stage] X: opening feature store',flush=True);time.sleep(3);print('  [stage] X: phase S Z (training)',flush=True);time.sleep(11)"
    $r = Invoke-Guarded "selftest" @("-c", "`"$code`"") $tdir
    Get-Content (Join-Path $tdir "ram_monitor_selftest.csv")
    exit $r
}

$busy = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "stage5_" }
if ($busy) { Write-Host "Another stage5 python process is running (PID $($busy.ProcessId -join ', ')) - not starting." -ForegroundColor Red; exit 5 }
$m0 = Mem
Write-Host ("Before start: available {0:N2} GB, free {1:N2} GB (start requires available >= {2} GB)" -f $m0.available, $m0.free, $MinStartAvailableGB)
if ($m0.available -lt $MinStartAvailableGB) { Write-Host "Not enough available RAM to start." -ForegroundColor Red; exit 6 }
Assert-Protected

if ($Step -eq "gates") {
    $steps = @(@("gates_RAVDESS_pooled", @("scripts\stage5_g_run.py", "--pilot", "--gates-only", "--dataset", "RAVDESS", "--group", "pooled")),
               @("gates_RAVDESS_frameL", @("scripts\stage5_g_run.py", "--pilot", "--gates-only", "--dataset", "RAVDESS", "--group", "frameL")),
               @("gates_CREMA-D_pooled", @("scripts\stage5_g_run.py", "--pilot", "--gates-only", "--dataset", "CREMA-D", "--group", "pooled")),
               @("gates_CREMA-D_frameL", @("scripts\stage5_g_run.py", "--pilot", "--gates-only", "--dataset", "CREMA-D", "--group", "frameL")))
} else {
    $steps = @(@("precompute", @("scripts\stage5_g_precompute.py", "--datasets", "RAVDESS", "CREMA-D")))
    foreach ($ds in "RAVDESS", "CREMA-D") {
        foreach ($g in "pooled", "frameL", "frameUM") { $steps += ,@("pilot_${ds}_$g", @("scripts\stage5_g_run.py", "--pilot", "--dataset", $ds, "--group", $g)) }
    }
    $steps += ,@("pilot_CREMA-D_zw", @("scripts\stage5_g_run.py", "--pilot", "--dataset", "CREMA-D", "--group", "zw"))
}
foreach ($s in $steps) {
    $r = Invoke-Guarded "$($s[0])_$Stamp" $s[1] $Dir
    if ($r -ne 0) { Write-Host "STOPPED at $($s[0]) (exit $r). Nothing restarted." -ForegroundColor Red; Assert-Protected -Quick; exit $r }
    Assert-Protected -Quick
}
Assert-Protected
Write-Host "$Step COMPLETE"
exit 0
