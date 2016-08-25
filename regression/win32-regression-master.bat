@echo off

%comspec% /c ""e:\regress\draco\regression\win32-regress.bat"" x86 > e:\regress\logs\regression-master.log
rem %comspec% /k ""e:\regress\draco\regression\win32-regress.bat"" x86

rem c:\myscript.%date:~-4%%date:~4,2%%date:~7,2%.%time::=%.log 2>&1
