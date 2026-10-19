# Keep Windows from sleeping / entering Modern Standby while the given parent process
# is alive (the display may still turn off). Exits as soon as the parent is gone, so
# the request is released even if the evaluation is killed and its EXIT trap never runs.
# Started by scripts/eval.sh for the duration of an evaluation run.
# ASCII only: Windows PowerShell 5.1 reads BOM-less files in the ANSI code page.
param([Parameter(Mandatory = $true)][int]$ParentPid)
Add-Type -Namespace TestPilot -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[TestPilot.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
while (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 10 }
