# Dot-source in a PowerShell session to put uv and dulwich on PATH:  . .\scripts\dev.ps1
$env:Path = "$env:USERPROFILE\.local\bin;$env:APPDATA\Python\Python314\Scripts;$env:Path"
