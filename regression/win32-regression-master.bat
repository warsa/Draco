@echo off
rem ---------------------------------------------------------------------------
rem File  : regression/win32-regression-master.bat
rem Date  : Tuesday, May 31, 2016, 14:48 pm
rem Author: Kelly Thompson
rem Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
rem         All rights are reserved.
rem ---------------------------------------------------------------------------

rem Use Task Scheduler to create a task that runs this script every night at 
rem midnight.

set logdir=e:\regress\logs

%comspec% /c ""e:\regress\draco\regression\update_regression_dir.bat"" x86 > %logdir%\update_regression_dir.log 2>&1
%comspec% /c ""e:\regress\draco\regression\win32-regress.bat"" x86 > %logdir%\regression-master.log 2>&1
rem %comspec% /k ""e:\regress\draco\regression\win32-regress.bat"" x86

rem c:\myscript.%date:~-4%%date:~4,2%%date:~7,2%.%time::=%.log 2>&1
