@echo off
rem ---------------------------------------------------------------------------
rem File  : regression/win32-regress.bat
rem Date  : Tuesday, May 31, 2016, 14:48 pm
rem Author: Kelly Thompson
rem Note  : Copyright (C) 2016, Los Alamos National Security, LLC.
rem         All rights are reserved.
rem ---------------------------------------------------------------------------

rem This file copied from c:\program files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat.  
rem It establishes a Visual Studio environment in a command prompt.  The 
rem Windows shortcut runs the following command:
rem
rem %comspec% /[k|c] ""C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"" x86
rem
rem This fill is called from win32-regression-master.bat so that all outpout
rem can be captured in a log file.

if "%1" == "" goto x86
if not "%2" == "" goto usage

if /i %1 == x86       goto x86
if /i %1 == amd64     goto amd64
if /i %1 == x64       goto amd64
if /i %1 == arm       goto arm
if /i %1 == x86_arm   goto x86_arm
if /i %1 == x86_amd64 goto x86_amd64
if /i %1 == amd64_x86 goto amd64_x86
if /i %1 == amd64_arm goto amd64_arm
goto usage

:x86
if not exist "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" goto missing
call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat"
goto :SetVisualStudioVersion

:SetVisualStudioVersion
set VisualStudioVersion=12.0
goto :vendorsetup

:usage
echo Error in script usage. The correct usage is:
echo     %0 [option]
echo where [option] is: x86 ^| amd64 ^| arm ^| x86_amd64 ^| x86_arm ^| amd64_x86 ^| amd64_arm
echo:
echo For example:
echo     %0 x86_amd64
goto :eof

:missing
echo The specified configuration type is missing.  The tools for the
echo configuration might not be installed.
goto :eof

rem -------------------------------------------------------------------------------------------
rem The main regression script starts here.
rem -------------------------------------------------------------------------------------------

:vendorsetup
call e:\work\vendors\setupvendors.bat
set USE_GITHUB=1

:cdash
rem set dashboard_type=Experimental
set dashboard_type=Nightly
set base_dir=e:\regress
set comp=cl
set script_dir=e:\regress\draco\regression
set script_name=Draco_Win32.cmake
set ctestparts=Configure,Build,Test,Submit

:dracodebug

set subproj=draco
set build_type=Debug
set work_dir=%base_dir%\cdash\%subproj%\%dashboard_type%_%comp%\%build_type%

rem print some information
echo Environment:
echo .
set
echo .
echo -----     -----     -----     -----     -----

rem navigate to the workdir
if not exist %work_dir% mkdir %work_dir%
cd /d %work_dir%

rem clear the build directory (need to do this here to avoid a hang).
rem if exist %work_dir%\build rmdir /s /q build
if not exist %work_dir%\build mkdir build
if not exist %work_dir%\source mkdir source
if not exist %work_dir%\target mkdir target

rem goto :jayennerelease

echo "ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\draco-%build_type%-cbts.log"
ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\draco-%build_type%-cbts.log

:dracorelease

set subproj=draco
set build_type=Release
set work_dir=%base_dir%\cdash\%subproj%\%dashboard_type%_%comp%\%build_type%

rem navigate to the workdir
if not exist %work_dir% mkdir %work_dir%
cd /d %work_dir%

rem clear the build directory (need to do this here to avoid a hang).
rem if exist %work_dir%\build rmdir /s /q build
if not exist %work_dir%\build mkdir build
if not exist %work_dir%\source mkdir source
if not exist %work_dir%\target mkdir target

rem run the ctest script

echo "ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\draco-%build_type%-cbts.log"
ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\draco-%build_type%-cbts.log

rem --------------------------------------------------------------------------

:jayennerelease

set script_dir=e:\regress\jayenne\regression
set script_name=Jayenne_Win32.cmake

set subproj=jayenne
set build_type=Release
set work_dir=%base_dir%\cdash\%subproj%\%dashboard_type%_%comp%\%build_type%

rem navigate to the workdir
if not exist %work_dir% mkdir %work_dir%
cd /d %work_dir%

rem clear the build directory (need to do this here to avoid a hang).
rem if exist %work_dir%\build rmdir /s /q build
if not exist %work_dir%\build mkdir build
if not exist %work_dir%\source mkdir source
if not exist %work_dir%\target mkdir target

rem goto :jayennedebug

rem run the ctest script

echo "ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\%subproj%-%build_type%-cbts.log"
ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\%subproj%-%build_type%-cbts.log

:jayennedebug

set build_type=Debug
set work_dir=%base_dir%\cdash\%subproj%\%dashboard_type%_%comp%\%build_type%

rem navigate to the workdir
if not exist %work_dir% mkdir %work_dir%
cd /d %work_dir%

rem clear the build directory (need to do this here to avoid a hang).
rem if exist %work_dir%\build rmdir /s /q build
if not exist %work_dir%\build mkdir build
if not exist %work_dir%\source mkdir source
if not exist %work_dir%\target mkdir target

rem run the ctest script

echo "ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\%subproj%-%build_type%-cbts.log"
ctest -VV -S %script_dir%\%script_name%,%dashboard_type%,%build_type%,%ctestparts% > %base_dir%\logs\%subproj%-%build_type%-cbts.log


:done
echo You need to remove -k from script launch to let this window close automatically.
