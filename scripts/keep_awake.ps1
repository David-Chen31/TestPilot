# Keep Windows from sleeping / entering Modern Standby while this process is alive
# (the display may still turn off). The request is released automatically when the
# process exits. Started by scripts/eval.sh for the duration of an evaluation run.
# ASCII only: Windows PowerShell 5.1 reads BOM-less files in the ANSI code page.
Add-Type -Namespace TestPilot -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED = [uint32]"0x00000001"
[TestPilot.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
while ($true) { Start-Sleep -Seconds 60 }
